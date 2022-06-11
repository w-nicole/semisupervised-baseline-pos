# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed amp_level to be dependent on CPU to permit running on CPU.
# Removed testing and irrelevant code (such as unsupported methods from the previous codebase)
# Added checkpoint load for decoder and changed to finetune decoder
# Added logic to load encoder checkpoint path so BaseVAE can be declared
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

# Added
def add_encoder_checkpoint_hparams(hparams):
    hparam_dict = vars(hparams)
    folder = util.get_folder_from_checkpoint_path(hparams.decoder_checkpoint)
    with open(os.path.join(folder, 'hparams.yaml')) as f:
        base_hparams = yaml.safe_load(f)
    encoder_checkpoint = base_hparams['encoder_checkpoint']
    hparam_dict['encoder_checkpoint'] = encoder_checkpoint
    return Namespace(**hparam_dict)
# end added

def main(hparams):
    hparams = add_encoder_checkpoint_hparams(hparams) # Added
    if hparams.cache_dataset:
        if not hparams.cache_path:
            hparams.cache_path = os.path.join(os.path.expanduser("~"), ".cache/clnlp")
        os.makedirs(hparams.cache_path, exist_ok=True)

    if hparams.do_train:
        model = VAE(hparams)
    else:
        assert os.path.isfile(hparams.checkpoint)
        model = VAE.load_from_checkpoint(hparams.checkpoint)

    os.makedirs(
        os.path.join(hparams.default_save_path, hparams.exp_name), exist_ok=True
    )
    logger = pl.loggers.TensorBoardLogger(
        hparams.default_save_path, name=hparams.exp_name, version=None
    )

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
    # Added the below lines until the divider.
    parser.add_argument("--decoder_checkpoint", default="", type=str)
    ############################################################################
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--min_delta", default=1e-3, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--save_last", default=False, type=util.str2bool)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--do_train", default=True, type=util.str2bool)
    # Below: changed do_test to False as this script doesn't support testing.
    parser.add_argument("--do_test", default=False, type=util.str2bool)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--cache_dataset", default=False, type=util.str2bool)
    parser.add_argument("--cache_path", default="", type=str)
    ############################################################################
    parser.add_argument("--default_save_path", default="./", type=str)
    parser.add_argument("--gradient_clip_val", default=0, type=float)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--overfit_batches", default=0.0, type=float)
    parser.add_argument("--track_grad_norm", default=-1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=util.str2bool)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--min_steps", default=None, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--log_every_n_steps", default=10, type=int)
    parser.add_argument("--accelerator", default=None, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--amp_backend", default="native", type=str)
    # only used for non-native amp
    # Changed to below to permit running on CPU
    parser.add_argument("--amp_level", default="01" if torch.cuda.is_available() else None, type=str)
    ############################################################################
    parser = Model.add_model_specific_args(parser)
    parser = Tagger.add_model_specific_args(parser)
    parser = BaseVAE.add_model_specific_args(parser)
    parser = VAE.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)