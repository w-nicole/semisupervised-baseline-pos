
# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

import argparse
from argparse import ArgumentParser, Namespace
import json
import os
import re
import yaml
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback

import collections.abc as container_abcs 
from torch._six import string_classes

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, RandomSampler, Sampler

import wandb
from enumeration import Task

apply_gpu = lambda item : item.cuda() if torch.cuda.is_available() else item
remove_from_gpu = lambda tensor : tensor.detach().cpu() if torch.cuda.is_available() else tensor.detach()

def get_folder_from_checkpoint_path(checkpoint_path):
    path_components = checkpoint_path.split('/')
    folder = '/'.join(path_components[:-2])
    return folder
    
def get_subset_model(model_class, is_masked, subset_count = -1):
    model_args_dict = {
        'data_dir' : "../../ud-treebanks-v1.4",
        'trn_langs' : 'English',
        'val_langs' : 'English',
        'masked' : 'y' if is_masked else 'n',
        'subset_count' : str(subset_count)
    }
    parser = model_class.add_model_specific_args(ArgumentParser())
    args = [ arg for arg_name, arg_value in model_args_dict.items() for arg in [f'--{arg_name}', arg_value] ]
    model_args = parser.parse_args(args)
    model = model_class(model_args)
    return model
    
def assert_if_is_full(phase, subset_model):
    if phase != 'train': return
    if not subset_model.hparams.subset_ratio == 1 and subset_model.hparams.subset_count == -1:
        print("model provided will not provide entire train set. Is this deliberate?")
        
def get_subset_dataloader(subset_model, lang, split):
    assert_if_is_full(split, subset_model)
    return subset_model.get_unshuffled_dataloader(lang, split)
    
def get_subset_labels(subset_model, lang, split):
    return subset_model.get_flat_labels(lang, split)
    
def train_call(model_class):
    parser = ArgumentParser()
    parser = add_training_arguments(parser)
    parser = model_class.add_model_specific_args(parser)
    hparams = parser.parse_args()
    train_main(hparams, model_class)
    
def train_main(raw_hparams, model_class):
    
    raw_hparams_dict = vars(raw_hparams)
    if raw_hparams.hyperparameter_names:
        hyperparam_names = raw_hparams.hyperparameter_names.split()
        raw_hparams_dict['name'] += '_'.join([
            f'{key}={raw_hparams_dict[key]}'
            for key in hyperparam_names
        ])
    else:
        if not raw_hparams_dict['name']:
            raw_hparams_dict['name'] = 'default'
    raw_hparams = Namespace(**raw_hparams_dict)

    wandb_dir = os.path.join(raw_hparams.default_save_path, raw_hparams.name)
    if not os.path.exists(wandb_dir): os.makedirs(wandb_dir)
    args = {
        'name' : raw_hparams.name,
        'job_type' : raw_hparams.job_type,
        'group' : raw_hparams.group,
        'config' : raw_hparams,
        'dir' : wandb_dir
    }
    wandb.init(**args)
    hparams = Namespace(**wandb.config)
    model = model_class(hparams)

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=model.selection_criterion,
        min_delta=hparams.min_delta,
        patience=hparams.patience,
        verbose=True,
        mode=model.comparison_mode,
        strict=True,
    )
    
    version_name = f"version_{wandb.run.id}"
    base_dir = os.path.join(wandb_dir, version_name)
        
    logger = pl.loggers.WandbLogger(
        name = raw_hparams.name,
        save_dir = hparams.default_save_path,
        version = version_name
    )
    
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    yaml_path = os.path.join(base_dir, 'hparams.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(hparams, f)

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
    logging_callback = Logging(base_dir)
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
    
    trainer.validate(model)
    if hparams.prep_termination:
        wandb.mark_preempting()
    if hparams.do_train:
        trainer.fit(model)
    # Added below if/printout
    if hparams.do_test:
        print('Will not perform testing, as this script does not test.')
    
def add_training_arguments(parser):
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--min_delta", default=1e-3, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--save_last", default=False, type=str2bool)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--do_train", default=True, type=str2bool)
    # Below: changed do_test to False as this script doesn't support testing.
    parser.add_argument("--do_test", default=False, type=str2bool)
    parser.add_argument("--checkpoint", default="", type=str)
    ############################################################################
    parser.add_argument("--default_save_path", default="./", type=str)
    parser.add_argument("--gradient_clip_val", default=0, type=float)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--overfit_batches", default=0.0, type=float)
    parser.add_argument("--track_grad_norm", default=-1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=str2bool)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--min_steps", default=None, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--log_every_n_steps", default=1, type=int)
    parser.add_argument("--accelerator", default=None, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--amp_backend", default="native", type=str)
    # only used for non-native amp
    parser.add_argument("--amp_level", default="01" if torch.cuda.is_available() else None, type=str)
    parser.add_argument("--log_wandb", default=True, type=str2bool)
    parser.add_argument("--log_frequency", default=1, type=int)
    parser.add_argument("--group", default="", type=str)
    parser.add_argument("--job_type", default="", type=str)
    parser.add_argument("--hyperparameter_names", default="", type=str)
    parser.add_argument("--prep_termination", default=False, type=str2bool)
    return parser
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Logging(Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.filename = os.path.join(save_dir, "results.jsonl")

    def on_run_end(self, trainer, pl_module, phase):
        """Called when the validation loop ends."""
        with open(self.filename, "a") as fp:
            logs = dict()
            for k, v in trainer.callback_metrics.items():
                if k.startswith(phase):
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    logs[k] = v
            logs["step"] = trainer.global_step
            print(json.dumps(logs), file=fp)
            
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.reset_metrics('train')
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.on_run_end(trainer, pl_module, 'train')
    
    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pl_module.reset_metrics('val')

    def on_validation_end(self, trainer, pl_module):
        self.on_run_end(trainer, pl_module, 'val')

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        pl_module.reset_metrics('tst')

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        with open(self.filename, "a") as fp:
            logs = dict()
            for k, v in trainer.callback_metrics.items():
                if k.startswith("tst_") or k == "select":
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    logs[k] = v
            print(json.dumps(logs), file=fp)

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def get_inverse_square_root_schedule_with_warmup(
    optimizer, warmup_steps, last_epoch=-1
):
    """
    Create a schedule with linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def lr_lambda(step):
        decay_factor = warmup_steps ** 0.5
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return decay_factor * step ** -0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, warmup_steps, training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        ratio = (training_steps - step) / max(1, training_steps - warmup_steps)
        return max(ratio, 0)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def pad_batch(batch, padding=-1):
    max_len = max([len(b) for b in batch])
    new_batch = []
    for b in batch:
        b_ = np.zeros(max_len, dtype=b.dtype) + padding
        b_[: len(b)] = b
        new_batch.append(b_)
    return new_batch


def default_collate(batch, padding):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        assert all([ torch.all(elem.long() == elem) for elem in batch ])
        return torch.stack([ elem.long() for elem in batch ] , 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            assert len(elem.shape) == 1, f"Only conceptually supports 1d arrays. Current element shape: {elem.shape}"
            return default_collate(
                [torch.as_tensor(b) for b in pad_batch(batch, padding)], padding
            )  # auto padding
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {
            key: default_collate([d[key] for d in batch], padding[key]) for key in elem
        }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(default_collate(samples, padding) for samples in zip(*batch))
        )
    elif isinstance(elem, container_abcs.Sequence):
        if isinstance(elem[0], string_classes):
            return batch
        transposed = zip(*batch)
        return [default_collate(samples, padding) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class ConcatSampler(Sampler):
    def __init__(self, concat_dataset: ConcatDataset, samples_per_dataset: int):
        assert isinstance(concat_dataset, ConcatDataset)
        self.concat_dataset = concat_dataset
        self.nb_datasets = len(concat_dataset.datasets)
        self.samples_per_dataset = samples_per_dataset

        weight = torch.tensor([len(d) for d in concat_dataset.datasets]).float()
        self.weight = weight / weight.sum()

    def sample_dataset(self):
        return torch.multinomial(self.weight, 1, replacement=True).item()

    def __iter__(self):
        iterators = [iter(RandomSampler(d)) for d in self.concat_dataset.datasets]
        done = np.array([False] * self.nb_datasets)
        while not done.all():
            dataset_id = self.sample_dataset()
            if done[dataset_id]:
                continue
            batch = []
            for _ in range(self.samples_per_dataset):
                try:
                    idx = next(iterators[dataset_id])
                except StopIteration:
                    done[dataset_id] = True
                    break
                if dataset_id > 0:
                    idx += self.concat_dataset.cumulative_sizes[dataset_id - 1]
                batch.append(idx)

            if len(batch) == self.samples_per_dataset:
                yield from batch

    def __len__(self):
        n = self.samples_per_dataset
        return sum([len(d) // n * n for d in self.concat_dataset.datasets])
    
        