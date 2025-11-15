from collections.abc import Sequence, Iterator
from pathlib import Path
from typing import Any, ClassVar

import os
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection
from omegaconf import DictConfig, ListConfig, OmegaConf
import platform 
from spikingjelly.activation_based import functional


from torch.optim import Optimizer

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates, get_target_len_from_metrics
from emg2qwerty.modules import (
    SpectrogramNorm,
    TDSConvEncoder,
    MultiBandRotationInvariantMLP,
    S4Model,
    GRULayer,
    SJ_SNN,
    transposed_AvgPool1d,
    GRULayerPooled,
)
from emg2qwerty.transforms import Transform

import pickle
import time


persistent_workers = False

class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
        loss: str,

        train_on_longer_every: int = -1,  # if >0, train on longer windows every train_on_longer_every epochs
        train_on_longer_factor: int = 1,  # if >0, train on longer windows every train_on_longer_every epochs
    
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.loss = loss

        self.train_on_longer_every = train_on_longer_every
        self.train_on_longer_factor = train_on_longer_factor
        self.use_longer_window_length = False

    def setup(self, stage: str | None = None, output_metadata: bool = False) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    loss = self.loss,
                    transform=self.train_transform,
                    window_length=self.window_length,   # in samples
                    padding=self.padding,               # in samples
                    jitter=True,
                    output_metadata=output_metadata,
                    # stride is left to None: it will be set to window_length
                )
                for hdf5_path in self.train_sessions
            ]
        )
        
        if self.train_on_longer_every > 0:
            assert self.train_on_longer_factor > 0, "train_on_longer_factor must be > 0 if train_on_longer_every is > 0"
            self.train_dataset_long = ConcatDataset(
                [
                    WindowedEMGDataset(
                        hdf5_path,
                        loss = self.loss,
                        transform=self.train_transform,
                        window_length=self.window_length*self.train_on_longer_factor,   # in samples
                        padding=self.padding,               # in samples
                        jitter=True,
                        output_metadata=output_metadata,
                        # stride is left to None: it will be set to window_length
                    )
                    for hdf5_path in self.train_sessions
                ]
            )

        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    loss = self.loss,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                    output_metadata=output_metadata,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        self.val_dataset_continuous = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    loss = self.loss,
                    transform=self.val_transform,
                    window_length=self.window_length*20,  # longer window lenghts
                    padding=self.padding,
                    jitter=False,
                    output_metadata=output_metadata,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    loss = self.loss,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                    output_metadata=output_metadata,
                )
                for hdf5_path in self.test_sessions
            ]
        )

        self.test_dataset_chunk = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    loss = self.loss,
                    transform=self.test_transform,
                    window_length=self.window_length*20,  # longer window lenghts,
                    padding=self.padding,  # (0,0)
                    jitter=False,
                    output_metadata=output_metadata,
                )
                for hdf5_path in self.test_sessions
            ]
        )


    def train_dataloader(self) -> DataLoader:
        print(f' {self.trainer.current_epoch}   train dataloader called: use_longer_window_length:', self.use_longer_window_length)
        return DataLoader(
            self.train_dataset if not self.use_longer_window_length else self.train_dataset_long,
            batch_size=self.batch_size if not self.use_longer_window_length else max(4, self.batch_size//self.train_on_longer_factor), 
            shuffle=True,
            num_workers=self.num_workers,              # useful to load data in parallel
            collate_fn = WindowedEMGDataset.collateFalse,
            pin_memory=True,
            persistent_workers=persistent_workers if not self.use_longer_window_length else False,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn = WindowedEMGDataset.collateFalse,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        val_continuous_dataloader = DataLoader(
            self.val_dataset_continuous,
            batch_size=self.batch_size//8,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn = WindowedEMGDataset.collateTrue,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        return [val_dataloader, val_continuous_dataloader]

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return [DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,   # self.num_workers,
            collate_fn = WindowedEMGDataset.collateFalse,
            pin_memory=False,
            persistent_workers=False,
        )]
    
    def test_dataloader_chunk(self) -> DataLoader:
        return DataLoader(
            self.test_dataset_chunk,
            batch_size=self.batch_size//8,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn = WindowedEMGDataset.collateTrue,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
    
    



class BaseRecurrentModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_transform: str,
        in_features: int,
        mlp_features: Sequence[int] | str,
        loss : str,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        model_HF: DictConfig,
        model_LF: DictConfig,
        decoder: DictConfig,

        train_on_longer_every: int = -1,

        non_linearity: str='relu',          # rot invariant MLP
        pooling: str='mean',                # rot invariant MLP
        offsets: Sequence[int]=[-1,0,1],    # rot invariant MLP
        sparsity_RotInvMLP_after_non_linearity: float=0.0,  # sparsity after non-linearity for the rot invariant MLP
        norm: str='none',                # normalization for the rot invariant MLP  

        input_mean_pooling: int=1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.train_on_longer_every = train_on_longer_every

        self.input_mean_pooling = input_mean_pooling
        if 'spectrogram' in in_transform:
            assert input_mean_pooling==1, "input_mean_pooling must be 1 if in_transform is spectrogram based"
        self.input_mean_pooling_module = transposed_AvgPool1d(kernel_size=input_mean_pooling, stride=input_mean_pooling) if input_mean_pooling>1 else nn.Identity()
        
        # extracting mlp_features in case of 'auto_factor_LB_UB' format
        if isinstance(mlp_features, str):
            assert mlp_features.startswith('auto_') and mlp_features.count('_')==3, "mlp_features must be a sequence of integers or have format 'auto_factor_LB_UB'"
            factor = float(mlp_features.split('_')[1])
            lower_bound = 0      if mlp_features.split('_')[2]=='none' else int(mlp_features.split('_')[2]) 
            upper_bound = np.inf if mlp_features.split('_')[3]=='none' else int(mlp_features.split('_')[3])
            assert lower_bound <= upper_bound, f"lower_bound ({lower_bound}) must be less than or equal to upper_bound ({upper_bound})"
            if model_HF._target_ == 'emg2qwerty.modules.GRULayer':
                mlp_features = int(model_HF.d_hidden/factor)   # in this way GRU input has shape 2*d_hidden/factor
                mlp_features = [ min(max(mlp_features, lower_bound), upper_bound) ] 
            elif model_HF._target_ == 'emg2qwerty.modules.S4Model':
                mlp_features = int(model_HF.d_model/factor)     # in this way S4 input has shape 2*d_model/factor (then encoded in d_model) -- maybe not necessary
                mlp_features = [ min(max(mlp_features, lower_bound), upper_bound) ] 
            elif model_HF._target_ == 'emg2qwerty.modules.TemporalConvolution':
                mlp_features = int(model_HF.out_channels[0]/factor)       # in this way TemporalConvolution input has shape 2*out_channels[0]/factor
                mlp_features = [ min(max(mlp_features, lower_bound), upper_bound) ] 
            else:
                raise NotImplementedError(f"automatic mlp_features is not implemented for model {model_HF._target_} in lightning.py")

        if len(mlp_features)>0:
            num_features = self.NUM_BANDS * mlp_features[-1]
        else:
            num_features = self.NUM_BANDS * in_features
            assert non_linearity == 'none', "non_linearity must be 'none' if mlp_features is empty"

        self.loss = loss

        self.sparsity_RotInvMLP_after_non_linearity = sparsity_RotInvMLP_after_non_linearity
        self.RotInvMLP_layers = len(mlp_features)
        self.reg_loss_fn_RotInvMLP = nn.MSELoss(reduction='mean')

        if model_HF._target_ == 'identity':
            self.model_HF = nn.Identity()
            self.HLfreq_ratio = 1
            self.apply_reg_loss_HF = 'identity'
        else:
            self.model_HF = instantiate(model_HF, d_input=num_features)

            self.HLfreq_ratio = self.model_HF.HLfreq_ratio
            if self.HLfreq_ratio > 1:
                assert self.loss == 'ctc_loss', f"{self.loss} is not compatible with HLfreq_ratio > 1"
            
            if model_HF._target_ == 'emg2qwerty.modules.GRULayer':
                self.apply_reg_loss_HF = 'GRULayer'
                self.reg_loss_fn_HF = nn.MSELoss(reduction='mean')
            elif model_HF._target_ == 'emg2qwerty.modules.S4Model':
                self.apply_reg_loss_HF = 'S4Model'
                self.reg_loss_fn_HF = nn.MSELoss(reduction='mean')
            elif model_HF._target_ == 'emg2qwerty.modules.SJ_SNN':
                self.apply_reg_loss_HF = 'SJ_SNN'
                self.reg_loss_fn_HF = nn.MSELoss(reduction='mean')
            elif model_HF._target_ == 'emg2qwerty.modules.TemporalConvolution':
                self.apply_reg_loss_HF = 'TemporalConvolution'
            else:
                raise NotImplementedError(f"Unknown HF model {model_HF} in lightning.py")
            
        if model_LF._target_ == 'emg2qwerty.modules.S4Model':
            d_input = num_features if model_HF._target_ == 'identity' else self.model_HF.d_output
            d_output = model_LF.d_output if model_LF.d_output > 0 else d_input
            self.model_LF = instantiate(model_LF, d_input=d_input, d_output=d_output)
            self.apply_reg_loss_LF = 'S4Model'
            self.reg_loss_fn_LF = nn.MSELoss(reduction='mean')
        elif model_LF._target_ == 'emg2qwerty.modules.TDSConvEncoder':
            d_input = num_features if model_HF._target_ == 'identity' else self.model_HF.d_output
            self.model_LF = instantiate(model_LF, d_input=d_input)
            self.apply_reg_loss_LF = 'TDSConvEncoder'
        elif model_LF._target_ == 'emg2qwerty.modules.GRULayer':
            d_input = num_features if model_HF._target_ == 'identity' else self.model_HF.d_output
            self.model_LF = instantiate(model_LF, d_input=d_input)
            self.apply_reg_loss_LF = 'GRULayer'
        elif model_LF._target_ == 'emg2qwerty.modules.GRULayerPooled':
            d_input = num_features if model_HF._target_ == 'identity' else self.model_HF.d_output
            self.model_LF = instantiate(model_LF, d_input=d_input)
            self.apply_reg_loss_LF = 'GRULayerPooled'
        else:
            raise NotImplementedError(f"Unknown LF model: {model_LF._target_} in lightning.py")
        

        # Define Model:
        # inputs: (T, N, bands=2, electrode_channels=16, possibly freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, electrode_channels=16, possibly freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS) if 'spectrogram' in in_transform else nn.Identity(),

            self.input_mean_pooling_module,
            
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(      # is compatible with and without frequency dimension; if mlp_features is empty, performs only the rotation invariant "augmentation"
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
                non_linearity=non_linearity,
                pooling=pooling,
                offsets=offsets,
                sparsity_after_non_linearity=sparsity_RotInvMLP_after_non_linearity,
                norm=norm, 
            ),
            # (T, N, num_features)      # flattening over bands (dim=2) and channels
            nn.Flatten(start_dim=2),
            
            self.model_HF,      # takes (T_HF, N, C) and returns (T_LF, N, C_out_HF=C_in_LF)
            self.model_LF,      # takes (T_LF, N, C_in_LF) and returns (T_LF, N, C_out_LF)

            # (T, N, num_classes)
            nn.Linear(self.model_LF.d_output, charset().num_classes),
            nn.LogSoftmax(dim=-1) if self.loss=='ctc_loss' else nn.Identity(), 
        )

        # Decoder
        self.decoder = instantiate(decoder)

        # Criterion
        if self.loss == 'ctc_loss':
            self.loss_fn = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        elif self.loss == 'cross_entropy_loss':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Unknown loss function: {loss}")

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "val_continuous", "test", "test_cont"]
            }
        )

        self.i_batch = -1
        self.save_calibration_data = None

    
    def on_before_optimizer_step(self, optimizer: Optimizer, opt_idx: int) -> None:
        """Called before ``optimizer.step()``."""  

        nan_found = False
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    nan_found = True
                    break
        if nan_found:
            # Zero out gradients so optimizer.step() becomes a no-op
            optimizer.zero_grad(set_to_none=True)
            print(f"Skipped optimizer step due to NaN/Inf gradients {nan_found}  -  ep: {self.trainer.current_epoch}", flush=True)

            device = next(self.model.parameters()).device
            rank = self.global_rank if hasattr(self, 'global_rank') else getattr(self.trainer, 'global_rank', 0)
            path_to_ckpt = self.trainer.checkpoint_callback.last_model_path

            # reload last checkpoint
            checkpoint = torch.load(path_to_ckpt, map_location=device)
            state_dict = checkpoint['state_dict']
            self.trainer.lightning_module.load_state_dict(state_dict, strict=True)
            for i_opt in range(len(self.trainer.optimizers)):
                self.trainer.optimizers[i_opt].load_state_dict(checkpoint['optimizer_states'][i_opt])

            print(f"rank {rank}   device {device} loaded checkpoint from {path_to_ckpt}", flush=True)
        

    def on_train_epoch_start(self):
        # Determine whether to change window length this epoch
        if self.train_on_longer_every >0:
            epoch = self.trainer.current_epoch
            if self.save_calibration_data is None:
                assert self.train_on_longer_every>1

            datamodule = self.trainer.datamodule
            if (epoch+1)%self.train_on_longer_every == 0:
                datamodule.use_longer_window_length = True
                self.trainer.reset_train_dataloader()
                print('Info: setting use_longer_window_length to True')
            elif epoch>0 and epoch%self.train_on_longer_every == 0:
                datamodule.use_longer_window_length = False
                self.trainer.reset_train_dataloader()
                print('Info: setting use_longer_window_length to False')
        else:
            print(f'Info: train_on_longer_every is {self.train_on_longer_every}')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        sparsity_dict = {}
        for i_layer, layer in enumerate(self.model):
            if self.save_calibration_data is not None:
                dir_path = f'temp_input_data/{self.save_calibration_data}'
                if i_layer == 0:
                    self.i_batch += 1
                if not os.path.isdir(f'{dir_path}/{i_layer}_{layer.__class__.__name__}'):
                    os.makedirs(f'{dir_path}/{i_layer}_{layer.__class__.__name__}')
                np.savez(f'{dir_path}/{i_layer}_{layer.__class__.__name__}/{self.i_batch}_input.npz', x=x.detach().cpu().numpy())
                
            x = layer(x)
            if isinstance(x, torch.Tensor):
                pass
            else:
                x, temp_dict = x
                for key, value in temp_dict.items():
                    if key not in sparsity_dict:
                        sparsity_dict[key] = value
                    else:
                        raise ValueError(f"Key {key} already exists in sparsity_dict [layer {layer.__class__.__name__}]")
        
        return x, sparsity_dict
    

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
    
        if phase=='test':
            torch.cuda.empty_cache()
            time.sleep(5)

        inputs = batch["inputs"]                            # has shape T,N
        targets = batch["targets"]                          # has shape T,N
        input_lengths = batch["input_lengths"]              # has shape N: for each element in the batch, the lenght of the input sequence
        target_lengths = batch["target_lengths"]            # has shape N: for each element in the batch, the lenght of the target sequence (also output sequnce in the case of cross entropy loss)
        continuous_val = batch["continuous_val"]            # this is used for longer chunks in validation AND for chunk based testing
        batch_size = len(input_lengths)  # batch_size

        if phase == 'val':
            phase_string = f"val_continuous" if continuous_val else f"val"
        elif phase == 'test':
            phase_string = f"test" if continuous_val else f"test_cont"
        else:
            phase_string = phase

        rank = self.global_rank if hasattr(self, 'global_rank') else getattr(self.trainer, 'global_rank', 0)

        emissions, sparsity_dict = self.forward(inputs)        # has shape (T, batch_size, num_classes)

        try:
            LowFreq_HLfreq_ratio = self.model_LF.HLfreq_ratio
        except AttributeError:
            LowFreq_HLfreq_ratio = 1
        try:
            upscale_last_layer = self.model_LF.upscale_last_layer
        except AttributeError:
            upscale_last_layer = 1

        if self.HLfreq_ratio == 1:
            # Shrink input lengths by an amount equivalent to the conv encoder's
            # temporal receptive field to compute output activation lengths for CTCLoss.
            # NOTE: This assumes the encoder doesn't perform any temporal downsampling
            # such as by striding.
            T_diff = inputs.shape[0]//(LowFreq_HLfreq_ratio*self.input_mean_pooling) - emissions.shape[0]//upscale_last_layer       # number (in LF_timesteps, not upscaled)     
            emission_lengths = (input_lengths//(LowFreq_HLfreq_ratio*self.input_mean_pooling) - T_diff)*upscale_last_layer          # number for each input in the batch [vector]
        else:
            T_diff = inputs.shape[0]//(self.HLfreq_ratio*LowFreq_HLfreq_ratio*self.input_mean_pooling) - emissions.shape[0]//upscale_last_layer  
            emission_lengths = (input_lengths//(self.HLfreq_ratio*LowFreq_HLfreq_ratio*self.input_mean_pooling) - T_diff)*upscale_last_layer

        if self.loss == 'ctc_loss':
            loss = self.loss_fn(
                log_probs=emissions,  # (T, batch_size, num_classes)
                targets=targets.transpose(0, 1),  # (T, batch_size) -> (batch_size, T)
                input_lengths=emission_lengths,  # (batch_size,)
                target_lengths=target_lengths,  # (batch_size,)
            )
        elif self.loss == 'cross_entropy_loss':
            loss = self.loss_fn(
                emissions.view(-1, emissions.size(-1)),        # (T*batch_size, num_classes)
                targets.view(-1),                              # (T*batch_size)
            )
            targets = targets.detach().cpu()
            targets = nn.functional.one_hot(targets, num_classes=charset().num_classes).numpy().astype(np.float32)  # (T, batch_size, num_classes)
            targets = self.decoder.decode_batch(
                emissions=targets,
                emission_lengths=target_lengths.detach().cpu().numpy(),
            )

        if self.sparsity_RotInvMLP_after_non_linearity>0:
            for i_layer in range(self.RotInvMLP_layers):
                sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_RotInvMLP_after_non_linearity_{i_layer}']
                assert isinstance(sparsity, torch.Tensor), f"sparsity is not a tensor: {sparsity} {type(sparsity)}"
                assert np.abs(sparsity.item()-(1-num_zeros/num_elements))<0.001, f"sparsity is not correct: {sparsity} {num_zeros/num_elements} [sparse quantity must be binary]"
                reg_loss = self.sparsity_RotInvMLP_after_non_linearity * self.reg_loss_fn_RotInvMLP(sparsity, torch.zeros_like(sparsity))
                loss = loss + reg_loss
                self.log(f"{phase_string}/sparsity_RotInvMLP_after_non_linearity_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)
        else:
            for i_layer in range(self.RotInvMLP_layers):
                sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_RotInvMLP_after_non_linearity_{i_layer}']
                sparsity = 1-num_zeros/num_elements   # sparsity is the fraction of non_zero elements: corrected on 21/05/2025  from B300
                self.log(f"{phase_string}/sparsity_NZ_RotInvMLP_after_non_linearity_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)

        if self.apply_reg_loss_HF == 'GRULayer':
            for i_layer in range(self.model_HF.num_layers):

                ''' sparsity_GRU_after_non_linearity  '''
                if self.model_HF.sparsity_after_non_linearity>0:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_GRU_after_non_linearity_{i_layer}']
                    assert isinstance(sparsity, torch.Tensor), f"sparsity is not a tensor: {sparsity} {type(sparsity)}"     # must be a tensor for differentiation
                    assert np.abs(sparsity.item()-(1-num_zeros/num_elements))<0.001, f"GRU sparsity is not correct: {sparsity} {num_zeros/num_elements} [sparse quantity must be binary]"       # mean must be coherent with the number of zeros
                    reg_loss = self.model_HF.sparsity_after_non_linearity * self.reg_loss_fn_HF(sparsity, torch.zeros_like(sparsity))
                    loss = loss + reg_loss
                    self.log(f"{phase_string}/sparsity_GRU_after_non_linearity_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)  
                else:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_GRU_after_non_linearity_{i_layer}']
                    sparsity = 1-num_zeros/num_elements
                    self.log(f"{phase_string}/sparsity_NZ_GRU_after_non_linearity_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)  

                ''' sparsity_EGRU '''
                if self.model_HF.gru_model=='EventBasedGRU':
                    if self.model_HF.sparsity_hidden_EGRU>0:
                        sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_hidden_EGRU_{i_layer}']
                        assert isinstance(sparsity, torch.Tensor), f"sparsity is not a tensor: {sparsity} {type(sparsity)}"
                        assert np.abs(sparsity.item()-(1-num_zeros/num_elements))<0.001, f"EventBasedGRU sparsity is not correct: {sparsity} {num_zeros/num_elements} [sparse quantity must be binary]"       # mean must be coherent with the number of zeros
                        reg_loss = self.model_HF.sparsity_hidden_EGRU * self.reg_loss_fn_HF(sparsity, torch.zeros_like(sparsity))
                        loss = loss + reg_loss
                        self.log(f"{phase_string}/sparsity_hidden_EGRU_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)
                    else:
                        sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_hidden_EGRU_{i_layer}']
                        sparsity = 1-num_zeros/num_elements   
                        self.log(f"{phase_string}/sparsity_NZ_hidden_EGRU_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)
        
        if self.apply_reg_loss_HF == 'S4Model':
            if self.model_HF.reg_timescale>0:
                timescales = self.model_HF.get_timescales()
                reg_loss = self.model_HF.reg_timescale * self.reg_loss_fn_HF(timescales, torch.zeros_like(timescales))
                loss = loss + reg_loss
            
            for i_layer in range(self.model_HF.num_layers):
                if self.model_HF.sparsity_post_actv_S4D>0:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_post_actv_S4D_HF_{i_layer}']
                    assert isinstance(sparsity, torch.Tensor), f"sparsity is not a tensor: {sparsity} {type(sparsity)}"
                    assert np.abs(sparsity.item()-(1-num_zeros/num_elements))<0.001, f"S4Model (HF) sparsity is not correct: {sparsity} {num_zeros/num_elements} [sparse quantity must be binary]"
                    reg_loss = self.model_HF.sparsity_post_actv_S4D * self.reg_loss_fn_HF(sparsity, torch.zeros_like(sparsity))
                    loss = loss + reg_loss
                    self.log(f"{phase_string}/sparsity_post_actv_S4D_HF_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)
                else:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_post_actv_S4D_HF_{i_layer}']
                    sparsity = 1-num_zeros/num_elements
                    self.log(f"{phase_string}/sparsity_NZ_post_actv_S4D_HF_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)      # NZ is for not zeros based loss [equal to sparsity in case of binary quantities]

        if self.apply_reg_loss_HF == 'SJ_SNN':
            for i_layer in range(self.model_HF.num_layers-1):   # not considering the last one
                if self.model_HF.sparsity_spiking_activity>0:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_spiking_activity_{i_layer}']
                    assert isinstance(sparsity, torch.Tensor), f"sparsity is not a tensor: {sparsity} {type(sparsity)}"
                    assert np.abs(sparsity.item()-(1-num_zeros/num_elements))<0.001, f"SJ_SNN sparsity is not correct: {sparsity} {num_zeros/num_elements} [sparse quantity must be binary]"
                    reg_loss = self.model_HF.sparsity_spiking_activity * self.reg_loss_fn_HF(sparsity, torch.zeros_like(sparsity))
                    loss = loss + reg_loss
                    self.log(f"{phase_string}/sparsity_spiking_activity_HF_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)
                else:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_spiking_activity_{i_layer}']
                    sparsity = 1-num_zeros/num_elements
                    self.log(f"{phase_string}/sparsity_NZ_spiking_activity_HF_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)        

                sparsity, num_elements, num_zeros  = sparsity_dict[f'sparsity_SJSNN_post_reduction']
                sparsity = 1-num_zeros/num_elements
                self.log(f"{phase_string}/sparsity_NZ_SJSNN_post_reduction", sparsity, batch_size=batch_size, sync_dist=True)        

        if self.apply_reg_loss_LF == 'S4Model':
            if self.model_LF.reg_timescale>0:
                timescales = self.model_LF.get_timescales()
                reg_loss = self.model_LF.reg_timescale * self.reg_loss_fn_LF(timescales, torch.zeros_like(timescales))
                loss = loss + reg_loss
            
            for i_layer in range(self.model_LF.num_layers):
                if self.model_LF.sparsity_post_actv_S4D>0:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_post_actv_S4D_{i_layer}']
                    assert isinstance(sparsity, torch.Tensor), f"sparsity is not a tensor: {sparsity} {type(sparsity)}"
                    assert np.abs(sparsity.item()-(1-num_zeros/num_elements))<0.001, f"S4Model (LF) sparsity is not correct: {sparsity} {num_zeros/num_elements} [sparse quantity must be binary]"
                    reg_loss = self.model_LF.sparsity_post_actv_S4D * self.reg_loss_fn_LF(sparsity, torch.zeros_like(sparsity))
                    loss = loss + reg_loss
                    self.log(f"{phase_string}/sparsity_post_actv_S4D_LF_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)
                else:
                    sparsity, num_elements, num_zeros = sparsity_dict[f'sparsity_post_actv_S4D_{i_layer}']
                    sparsity = 1-num_zeros/num_elements
                    self.log(f"{phase_string}/sparsity_NZ_post_actv_S4D_LF_{i_layer}", sparsity, batch_size=batch_size, sync_dist=True)

                
            if torch.isnan(loss) or torch.isinf(loss):
                loss = 0
                for i_b in range(batch_size):
                    temp_loss = self.loss_fn(
                        log_probs=emissions[:,i_b:i_b+1,:],  # (T, batch_size, num_classes)
                        targets=targets.transpose(0, 1)[i_b:i_b+1],  # (T, batch_size) -> (batch_size, T)
                        input_lengths=emission_lengths[i_b:i_b+1],  # (batch_size,)
                        target_lengths=target_lengths[i_b:i_b+1],  # (batch_size,)
                    )
                    if torch.isnan(temp_loss) or torch.isinf(temp_loss):
                        print(f"Rank {rank}: Batch {i_b} loss is NaN/Inf")
                        continue
                    else:
                        loss += temp_loss
                loss = loss/batch_size
                if type(loss) is not torch.Tensor:
                    loss = self.loss_fn(
                        log_probs=emissions,  # (T, batch_size, num_classes)
                        targets=targets.transpose(0, 1),  # (T, batch_size) -> (batch_size, T)
                        input_lengths=emission_lengths,  # (batch_size,)
                        target_lengths=target_lengths,  # (batch_size,)
                    )
                    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

            # If still problematic, gather diagnostic info from all processes
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"CRITICAL: Persistent NaN/Inf on rank {rank}")
                
                # For debugging layer by layer
                with torch.no_grad():
                    x = inputs
                    for i, layer in enumerate(self.model):
                        x = layer(x)
                        if type(x) is tuple:
                            x, _ = x
                        print(f'Rank {rank}, Layer {i} ({layer.__class__.__name__}): {x.shape} (nans: {torch.isnan(x).sum()}/{x.numel()})')

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase_string}_metrics"]
        if self.loss == 'ctc_loss':
            targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(batch_size):
            if self.loss == 'ctc_loss':
                # Unpad targets (T, batch_size) for batch entry
                target_temp = [ targ for targ in targets[:target_lengths[i], i] if targ!=charset().null_class ] # this is useful only in test when REP_TARGET>1
                target = LabelData.from_labels(target_temp)
            elif self.loss == 'cross_entropy_loss':
                target = targets[i]
            metrics.update(prediction=predictions[i], target=target)


        self.log(f"{phase_string}/loss", 0. if torch.isnan(loss) or torch.isinf(loss) else loss, batch_size=batch_size, sync_dist=True)       # this is called at every batch, the log shoud use on_epoch = True for valid and test and False for training....
        return None if torch.isnan(loss) or torch.isinf(loss) else loss



    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        trg_len = get_target_len_from_metrics(metrics)
        if phase == 'test':
            updated = trg_len > 0
        else:
            updated = True
        if updated:
            self.log_dict(metrics.compute(), sync_dist=True)        # in all cases on_epoch = True shoud be used
            metrics.reset()

        if phase == 'val':
            continuous_metrics = self.metrics["val_continuous_metrics"]
            self.log_dict(continuous_metrics.compute(), sync_dist=True) 
            continuous_metrics.reset()
        elif phase == 'test':
            continuous_metrics = self.metrics["test_cont_metrics"]
            trg_len = get_target_len_from_metrics(continuous_metrics)
            updated = trg_len > 0
            if updated:       
                self.log_dict(continuous_metrics.compute(), sync_dist=True) 
                continuous_metrics.reset() 
        

    def on_validation_batch_start(self, *args, **kwargs) -> None:
        self._batch_start("val")
    def on_train_batch_start(self, *args, **kwargs) -> None:
        self._batch_start("train")
    def on_test_batch_start(self, *args, **kwargs) -> None:
        self._batch_start("test")

    def _batch_start(self, phase: str) -> None:
        if isinstance(self.model_HF, SJ_SNN):
            # SJ_SNN has a specific reset method
            functional.reset_net(self.model_HF)


    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)


    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        # this is called after the two calls of validation_step
        self._epoch_end("val")
        
    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")


    def configure_optimizers(self) -> dict[str, Any]:
        # configure optimizers is called by trainer.fit

        if isinstance(self.model_HF, S4Model) or isinstance(self.model_LF, S4Model):
            # S4 requires a specific optimizer setup
            return setup_optimizer_S4Model(
                self.model,
                optimizer_config=self.hparams.optimizer,
                lr_scheduler_config=self.hparams.lr_scheduler,
            )
        elif isinstance(self.model_HF, TDSConvEncoder) or isinstance(self.model_LF, TDSConvEncoder) or isinstance(self.model_LF, GRULayer) or isinstance(self.model_LF, GRULayerPooled):
            # this should just be an else actually.....
            return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
            )
        else:
            raise NotImplementedError(
                f"Optimizer setup not implemented in lightining.py"
            )




def setup_optimizer_S4Model(
        model: nn.Module,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
    ) -> dict[str, Any]:
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C?, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """
    # Get all parameters in the model
    params = model.parameters()
    all_parameters = list(params)

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = instantiate(optimizer_config, params)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)           # this set up the scheduler and its parameters
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)        # here lr_scheduler_config has NOT a _target_ key. lr_scheduler is a dictionary with the lr_scheduler_config keys (interval) and sheduler overriding the lr_scheduler_config ones

    # Print optimizer info
    print("setup_optimizer_S4Model:")
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }
