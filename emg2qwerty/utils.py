# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
import emg2qwerty.ssm_models as ssm_models
import emg2qwerty.modules as modules
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from typing import Any, Callable, Dict, Optional, Tuple
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import wandb

from pytorch_lightning import _logger as log

def instantiate_optimizer_and_scheduler(
    params: Iterator[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }



def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def cpus_per_task(gpus_per_node: int, tasks_per_node: int, num_workers: int) -> int:
    """Number of CPUs to request per task per node taking into account
    the number of GPUs and dataloading workers."""
    gpus_per_task = gpus_per_node // tasks_per_node
    if gpus_per_task <= 0:
        return num_workers + 1
    else:
        return (num_workers + 1) * gpus_per_task




class MyCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing schedule.
        heavily inspired by MyCosineAnnealingWarmRestarts in https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/lr_scheduler.py#L1046

        - removed T_mult (1 always) -> T_i = T_0 always (removed T_i)
        - added decayment: lr_decay is a list with the moltiplicative factors for each iteration: [1, 1/2...]  # always referred to baseline learning rates
    """

    def __init__(
        self,
        optimizer: torch.optim.lr_scheduler.Optimizer,
        T_0: int,
        warmup_epochs: int,
        lr_decay: list,
        eta_min: float = 0.0,
        last_epoch: int = -1,

    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}"
            )
        
        self.T_0 = T_0      # epochs per iteration
        self.eta_min = eta_min
        self.T_cur = last_epoch         # will be replaced by current epoch % T_0
        self.Iter_cur = 0                 # will be replaced by current epoch // T_0
        self.warmup_epochs = warmup_epochs
        self.lr_decay = lr_decay
        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        # torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)
        if not self._get_lr_called_within_step:
            print("To get the last learning rate computed by the scheduler, please use `get_last_lr()`. Exiting...")
            exit()

        decay_factor = self.lr_decay[self.Iter_cur] if self.Iter_cur<len(self.lr_decay) else self.lr_decay[-1]

        # linear warmup
        if self.T_cur < self.warmup_epochs:
            return [
                self.eta_min + (base_lr*decay_factor - self.eta_min) * (self.T_cur / self.warmup_epochs)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.eta_min + (base_lr*decay_factor - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_0))/2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:                                           # iterative mode
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1         
            if self.T_cur >= self.T_0:
                self.T_cur = self.T_cur - self.T_0      # this assumes T_mult==1
                self.Iter_cur += 1                        # this assumes T_mult==1

        else:                                                       # called on specific epoch
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                self.T_cur = epoch % self.T_0           # this assumes T_mult==1
                self.Iter_cur = epoch // self.T_0         # this assumes T_mult==1
            else:
                self.T_cur = epoch
                self.Iter_cur = 0
        self.last_epoch = math.floor(epoch)

        with torch.optim.lr_scheduler._enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):         # call to the get_lr function
                param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class MyEarlyStopping(EarlyStopping):
    ''' check the improvement on monitored quantity at the end of each scheduler period only:
            - set patience to 1
            - set lr_scheduler_period equal to the scheduler period 
    '''

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,

        lr_scheduler_period: int = None,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )
       
        self.lr_scheduler_period = lr_scheduler_period
        log.info(f"INFO: Early stopping configured with patience {self.patience} and lr_scheduler_period {self.lr_scheduler_period}")

    def on_validation_end(self, trainer, pl_module):
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        # checks early stopping at the end of each scheduler period only
        if trainer.current_epoch == 0 or (trainer.current_epoch+1) % self.lr_scheduler_period == 0:
            self._run_early_stopping_check(trainer)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        if trainer.current_epoch == 0 or (trainer.current_epoch+1) % self.lr_scheduler_period == 0:
            self._run_early_stopping_check(trainer)






def plot_timescales(model):
    ''' model should be an instance of S4D model '''
    assert isinstance(model, modules.S4Model), f"model should be an instance of S4D model but is {type(model)}"


    S4_layers = model.s4_layers

    S4D_kernel_dtA = {}
    S4D_kernel_dtA_real = {}
    min_dt = np.inf 
    max_dt = 0
    min_dt_real = np.inf 
    max_dt_real = 0
    for i_layer, layer in enumerate(S4_layers):
        dt = np.exp(layer.kernel.log_dt.data.detach().numpy())              # (H)    a dt for each input channel
        log_A_real = layer.kernel.log_A_real.data.detach().numpy()
        A_imag = layer.kernel.A_imag.data.detach().numpy()
        A = -np.exp(log_A_real) + 1j * A_imag                    # (H N)  a complex A diagonal matrix for each input channel   # note: A has negative real components!
        A_real = np.exp(log_A_real)             # this is - the real part of A (A_real is strictly positive)

        S4D_kernel_dtA[i_layer] = A * dt[:, None]   # (H N)  a complex A diagonal matrix for each input channel      # np.exp(layer.kernel.log_dt.data.detach().numpy())
        S4D_kernel_dtA_real[i_layer] =  A_real * dt[:, None]
    
    S4D_kernel_time_scales_real = {} 
    S4D_kernel_time_scales = {}     # tau = -1/(log(a_bar)) = 1/(mod(a)*dt)   #intuitively if a and a_bar are real: a_bar=exp(a*dt)<1 -> a<0 (hence the minus disappear when i consider the abs)
    S4D_kernel_spectrums = {}
    for i_layer, layer in enumerate(S4_layers):
        time_scale = 1/ (np.abs(S4D_kernel_dtA[i_layer])).flatten() # (H N)  a real A diagonal matrix for each input channel
        time_scale_real = 1/ (np.abs(S4D_kernel_dtA_real[i_layer])).flatten() # (H N)  a real A diagonal matrix for each input channel
        spectrum = np.exp(S4D_kernel_dtA[i_layer])       # (H N)  a complex A diagonal matrix for each input channel

        S4D_kernel_time_scales[i_layer] = time_scale
        S4D_kernel_time_scales_real[i_layer] = time_scale_real
        S4D_kernel_spectrums[i_layer] = spectrum

        min_dt = min(np.min(S4D_kernel_time_scales[i_layer]), min_dt)
        max_dt = max(np.max(S4D_kernel_time_scales[i_layer]), max_dt)

        min_dt_real = min(np.min(S4D_kernel_time_scales_real[i_layer]), min_dt_real)
        max_dt_real = max(np.max(S4D_kernel_time_scales_real[i_layer]), max_dt_real)


    # plot timescales
    fig,axs = plt.subplots(1,2)
    bins = np.logspace(np.log10(min_dt), np.log10(max_dt), 100)
    bins_real = np.logspace(np.log10(min_dt_real), np.log10(max_dt_real), 100)

    for i_layer in S4D_kernel_time_scales.keys():
        axs[0].hist(S4D_kernel_time_scales[i_layer],      bins=bins,      label=f'Layer {i_layer}', histtype='step', cumulative=True, color=f'C{i_layer}')
        axs[1].hist(S4D_kernel_time_scales_real[i_layer], bins=bins_real, label=f'Layer {i_layer}', histtype='step', cumulative=True, color=f'C{i_layer}')
    axs[0].set_xlabel('1/abs(dt*A)')
    axs[1].set_xlabel('1/abs(dt*A_real)')
    axs[0].set_ylabel('cdf')
    plt.legend()
    plt.show()


    fig,axs = plt.subplots(1,2)
    bins = np.logspace(np.log10(min_dt), np.log10(max_dt), 12)
    bins_real = np.logspace(np.log10(min_dt_real), np.log10(max_dt_real), 12)
    for i_layer in S4D_kernel_time_scales.keys():
        axs[0].hist(S4D_kernel_time_scales[i_layer],      bins=bins,      label=f'Layer {i_layer}', histtype='step', cumulative=False, color=f'C{i_layer}')
        axs[1].hist(S4D_kernel_time_scales_real[i_layer], bins=bins_real, label=f'Layer {i_layer}', histtype='step', cumulative=False, color=f'C{i_layer}')

    axs[0].set_xlabel('1/abs(dt*A)')
    axs[1].set_xlabel('1/abs(dt*A_real)')
    axs[0].set_ylabel('pdf')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(int(np.ceil(len(S4D_kernel_spectrums)/4)), min(len(S4D_kernel_spectrums), 4))
    # draw the unit circle
    theta = np.linspace(0, 2 * np.pi, 100)  # 100 points from 0 to 2*pi
    x = np.cos(theta)
    y = np.sin(theta)

    axes = axes.flatten() if len(S4D_kernel_spectrums) > 1 else [axes]
    
    # plot the spectrum
    for i in range(len(S4_layers)):
        ax = axes[i]
        spectrum = S4D_kernel_spectrums[i]
        ax.plot(x, y, 'r', linewidth=1)
        ax.scatter(np.real(spectrum), np.imag(spectrum), marker='.', alpha=0.8, s=1)
    
        # format axis
        ax.set_title(f'Layer {i}')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.show()
    


def get_wandb_run_by_id(run_id: str, project: str):
    """
    Retrieve a specific wandb run using its ID.
    
    Args:
        run_id: The unique ID of the run (e.g., 'abc123de')
        project: The name of the wandb project
        entity: The wandb username or team name (optional)
        
    Returns:
        The wandb run object
    """
    api = wandb.Api()
    run_path = f"{project}/{run_id}"
    
    try:
        run = api.run(run_path)
        print(f"Retrieved run: {run.name} (ID: {run.id})")
        return run
    except Exception as e:
        print(f"Error retrieving run: {e}")
        return None
