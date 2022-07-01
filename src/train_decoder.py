# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed amp_level to be dependent on CPU to permit running on CPU.
# Removed testing and irrelevant code (such as unsupported methods from the previous codebase)
# Removed resume training, caching.
# Added wandb code, removed Tensorboard.
# Added imports as needed
# Changed comparsion -> comparison_mode

import os
from argparse import ArgumentParser, Namespace
import yaml
import pytorch_lightning as pl

import util
from enumeration import Task
from model import Model, Tagger, BaseVAE, VAE

import torch

# added
import wandb
# Added

def main(hparams):
    
    model = VAE(hparams)
    os.makedirs(
        os.path.join(hparams.default_save_path, hparams.exp_name), exist_ok=True
    )
    logger = pl.loggers.WandbLogger(
        name = util.get_model_path_section(trainer, hparams)
        save_dir = hparams.default_save_path, name=hparams.exp_name
    )
    args = { 'name' : util.get_model_path_section(logger.version, hparams) }
    if hparams.wandb_group: args['group'] = hparams.wandb_group
    if hparams.wandb_job_type: args['job_type'] = hparams.wandb_job_type
    wandb.init(**args)
    logger.watch(model, log_freq=5)

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=model.selection_criterion,
        min_delta=hparams.min_delta,
        patience=hparams.patience,
        verbose=True,
        mode=model.comparison_mode,
        strict=True,
    )
    base_dir = os.path.join(
        hparams.default_save_path,
        hparams.exp_name,
        f"version_{logger.version}" if logger.version is not None else "",
    )
    model.base_dir = base_dir
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base_dir, "ckpts"),
        filename="ckpts_{epoch}-{%s:.3f}" % model.selection_criterion,
        monitor=model.selection_criterion,
        verbose=True,
        save_last=hparams.save_last,
        save_top_k=hparams.save_top_k,
        mode=model.comparison_mode,
    )
    logging_callback = util.Logging(base_dir)
    lr_logger = pl.callbacks.LearningRateMonitor()
    callbacks = [early_stopping, checkpoint_callback, logging_callback, lr_logger]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=hparams.default_save_path,
        gradient_clip_val=hparams.gradient_clip_val,
        num_nodes=hparams.num_nodes,
        gpus=hparams.gpus,
        auto_select_gpus=True,
        overfit_batches=hparams.overfit_batches,
        track_grad_norm=hparams.track_grad_norm,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=hparams.fast_dev_run,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        max_steps=hparams.max_steps,
        min_steps=hparams.min_steps,
        val_check_interval=int(hparams.val_check_interval)
        if hparams.val_check_interval > 1
        else hparams.val_check_interval,
        log_every_n_steps=hparams.log_every_n_steps,
        accelerator=hparams.accelerator,
        precision=hparams.precision,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        replace_sampler_ddp=True,
        terminate_on_nan=True,
        amp_backend=hparams.amp_backend,
        amp_level=hparams.amp_level,
    )
    if hparams.do_train:
        trainer.fit(model)
    # Added below if/printout
    if hparams.do_test:
        print('Will not perform testing, as this script does not test.')


if __name__ == "__main__":
    parser = ArgumentParser()
    # Moved below logic to util.py
    parser = util.add_training_arguments(parser)
    # Moved argument adding logic to model-internal
    parser = VAE.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)