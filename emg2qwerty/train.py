
import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import time
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from platform import platform
import os
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from glob import glob

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform

os.environ['WANDB_INIT_TIMEOUT'] = '300'


log = logging.getLogger(__name__)

wandb_mode = 'online'
if wandb_mode != 'disabled':
    wandb.login(key=os.environ["WANDB_API_KEY"], timeout=300)



@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):

    log_dir = HydraConfig.get().run.dir
    print(f'main called: logging in dir **{log_dir}**')

    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    wandb_logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name,
        group=config.wandb.group,
        tags=config.wandb.tags,
        mode=config.wandb.mode,
    )
    run_id_wandb = wandb_logger.experiment.id
    log.info(f"Current WandB run ID: {run_id_wandb}")

    if config.autoTestOnly != '':
        log.info(f"Auto test only mode enabled")
        assert not config.train, "Auto test only mode is enabled, but config.train is set to True. Please set config.train to False."
        # copy checkpoints to log dir
        dir_checkpoint = f"multirun_log/{config.autoTestOnly}/checkpoints"
        available_checkpoints = glob(f"{dir_checkpoint}/*.ckpt")
        
        val_present = False
        for f in available_checkpoints:
            if 'val_' in f:
                val_present = True
        
        if not val_present:
            assert len(available_checkpoints) == 2

        os.mkdir(f"{log_dir}/checkpoints")
        for f in available_checkpoints:
            if 'last.ckpt' in f:
                destination_name = f'last.ckpt'
            else:
                if not val_present:
                    epoch = f.split('/')[-1]
                    epoch = epoch.split('=')[1].split('-')[0]
                    destination_name = f'val_{epoch}.ckpt'
                else:
                    destination_name = f.split('/')[-1]
            log.info(f"Copying checkpoint {f} to log dir {log_dir}/checkpoints/{destination_name}")
            os.system(f'cp {f} {log_dir}/checkpoints/{destination_name}')

    wandb_logger.log_hyperparams(config)

     # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    
    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    # Note: determinism is not effective with high values of train_on_longer_factor 
    # due to the non-deterministic behaviour of the CTC loss
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])
    

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.model}")

    # check if input is transformed in periodogram
    transf_dict_train = OmegaConf.to_container(config.transforms, resolve=True)['train']
    in_transf = 'raw_signal'
    for tr in transf_dict_train:
        if 'Spectrogram' in tr['_target_']:
            in_transf = 'spectrogram'
        if config.loss=='cross_entropy_loss':
            if 'TemporalAlignmentJitter' in tr['_target_']:
                raise NotImplementedError("TemporalAlignmentJitter is not compatible with cross_entropy_loss")
            if in_transf == 'spectrogram':
                raise NotImplementedError("Spectrogram is not compatible with cross_entropy_loss")

    module = instantiate(
        config.model,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        in_transform=in_transf,
        loss=config.loss,
        _recursive_=False,
        model_HF=config.model_HF,
        model_LF=config.model_LF,
        train_on_longer_every=config.train_on_longer_every,
    )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        loss = config.loss,
        train_on_longer_every = config.train_on_longer_every,
        train_on_longer_factor = config.train_on_longer_factor,
        _convert_="object",
    )

    if config.checkpoint is not None:

        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = module.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )
        assert module.loss == config.loss, f"Losses do not match: {module.loss} (loaded) != {config.loss} (config)"
        assert module.hparams.in_transform == in_transf, f"Input transforms do not match: {module.hparams.in_transform} (loaded) != {in_transf} (config)"


    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    extra_callback_configs = config.get("extra_callbacks", [])
    callback_configs = callback_configs + extra_callback_configs
    callbacks = [instantiate(cfg) for cfg in callback_configs]
    
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=False) if config.trainer.devices > 1 else None,
    )


    if config.train:
        # Train
        trainer.fit(module, datamodule)

        # Load best checkpoint
        print(f'loading checkpoint from trainer.best_model_path: {trainer.checkpoint_callback.best_model_path}')
        module = module.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    else:
        datamodule.setup()


    test_metrics = trainer.test(module, dataloaders=datamodule.test_dataloader_chunk())

    results = {
        "test_metrics": test_metrics,
    }

    pprint.pprint(results, sort_dicts=False)

    log.info("Finishing WandB run")
    wandb.finish()

    log.info(f"Saved WandB run to log dir {log_dir}")



if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    
    main()
    print('Ended job')