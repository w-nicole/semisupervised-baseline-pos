
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original include:
# Changed hyperparameters to match those in the paper,
# Added averaging behavior.
# Changed to not support weighted features, but instead a concatenation of all hidden representations.
# Changed to support single hidden layer MLP.
# Changed `comparsion` to `comparison_mode`
# Added support for logging train metrics and non-accuracy metrics.
# Changed forward to __call__.
# Changed logging methodology, added train logging, and moved to wandb.
# Removed epochwise train metrics.
# Changed checkpointing metric for compatibility with wandb logging.
# Removed ._metric
# Added manual dump of yaml.
# Plateau monitor changed to be epoch, on target language.
# Added extra custom logging maintenance and logic
# Added third return for __call__ for compatibility.
# Removed dropout

import hashlib
import json
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer

import constant
import util
from dataset.base import Dataset
from enumeration import Schedule, Split, Task
from metric import Metric, POSMetric, AverageMetric, LABEL_PAD_ID

from dataset import collate
import wandb
import math
import yaml
import plotly.express as px

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.optimizer = None
        self.scheduler = None
        self.metric_names = None
        # Changed below to account for train and other metrics.
        self.metrics: Dict[str, Dict[str, Dict[str, Metric]]] = defaultdict(dict)
        self.trn_datasets: List[Dataset] = None
        self.val_datasets: List[Dataset] = None
        self.tst_datasets: List[Dataset] = None
        self.padding: Dict[str, int] = {}
        self.base_dir: str = ""
        
        # below line: added
        self.optimization_loss = None

        self._batch_per_epoch: int = -1
        self._comparison_mode: Optional[str] = None
        self._selection_criterion: Optional[str] = None
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        pl.seed_everything(hparams.seed)
        
        # Added the following
        self.run_phases = [Split.train, 'val', Split.test]
        self.concat_all_hidden_states = self.hparams.concat_all_hidden_states
        
        if not self.hparams.target_language:
            one_other_target_language = (len(self.hparams.val_langs) == 2 and constant.SUPERVISED_LANGUAGE in self.hparams.val_langs)
            valid_val_langs = len(self.hparams.val_langs) == 1 or one_other_target_language
            assert valid_val_langs, "target_language/checkpoint was designed with at most 1 non-source language."
            # If it's a pure optimization, then use that language
            if len(self.hparams.trn_langs) == 1:
                self.target_language = self.hparams.trn_langs[0]
            # Otherwise, if mixed training, favor the non-English language
            elif one_other_target_language:
                not_supervised_languages = list(filter(lambda lang : lang != constant.SUPERVISED_LANGUAGE, self.hparams.val_langs))
                assert len(not_supervised_languages) == 1
                self.target_language = not_supervised_languages[0]
            else:
                assert False, "This case for checkpoint language was not considered."
        else:
            self.target_language = self.hparams.target_language
         # end additions

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrain)
        # Changed below to correspond to classmethod
        self.encoder_mbert = self.build_model(self.hparams.pretrain)
        self.freeze_layers()
         # Override layer specification if instead should freeze the whole thing
        if self.hparams.freeze_mbert:
            self.freeze_bert(self)

        # Removed dropout
        
        # Added below
        self.name_to_metric = {
            'pos_acc' : POSMetric('pos'),
            'token_acc' : POSMetric('token')
        }
        self.monitor_acc_key = 'pos'
        
        # Structure: dict[phase][lang] = [{metric_key : value}]
        self.custom_logs = defaultdict(dict)

    # Changed below to accept pretrain as argument so classmethod works.
    @classmethod
    def build_model(self, pretrain):
        config = AutoConfig.from_pretrained(
            pretrain, output_hidden_states=True
        )
        model = AutoModel.from_pretrained(pretrain, config=config)
        return model
        
    def build_linear(self, input_size, output_size, hidden_size, hidden_layers):
        return nn.Linear(input_size, output_size)
            
    def build_mlp(self, input_size, output_size, hidden_size, hidden_layers):
        assert hidden_layers >= 0
        layers = []
        # input layer
        layers.extend([
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU()
        ])
        for _ in range(hidden_layers):
            layers.extend([
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU()
            ])
        layers.extend([torch.nn.Linear(hidden_size, output_size)])
        return nn.Sequential(*tuple(layers))

    def freeze_layers(self):
        if self.hparams.freeze_layer == -1:
            return
        elif self.hparams.freeze_layer >= 0:
            for i in range(self.hparams.freeze_layer + 1):
                if i == 0:
                    print("freeze embeddings")
                    self.freeze_embeddings()
                else:
                    print(f"freeze layer {i}")
                    self.freeze_layer(i)

    def freeze_embeddings(self):
        util.freeze(self.encoder_mbert.embeddings)
            
    def freeze_bert(self, encoder):
        # Adapted from model/base.py by taking the logic to freeze up to and including a certain layer
        # Doesn't freeze the pooler, but encode_sent excludes pooler correctly.
        encoder.freeze_embeddings()
        for index in range(encoder.encoder_mbert.config.num_hidden_layers + 1):
            encoder.freeze_layer(index)
        # end adapted
   
    def freeze_layer(self, layer):
        util.freeze(self.encoder_mbert.encoder.layer[layer - 1])
        
    @property
    def mbert_output_size(self):
        # hidden_size = the input to the classifier
        # Added logic for concatenated embeddings
        single_layer_size = self.encoder_mbert.config.hidden_size
        if not self.concat_all_hidden_states:
            return single_layer_size
        else:
            return single_layer_size * (self.encoder_mbert.config.num_hidden_layers + 1)
        # end added
        
    @property
    def batch_per_epoch(self):
        if self.trn_datasets is None:
            self.trn_datasets = self.prepare_datasets(Split.train)

        if self._batch_per_epoch < 0:
            total_datasize = sum([len(d) for d in self.trn_datasets])
            self._batch_per_epoch = np.ceil(total_datasize / self.hparams.batch_size)

        return self._batch_per_epoch

    @property
    def selection_criterion(self):
        assert self._selection_criterion is not None
        return self._selection_criterion

    @property
    def comparison_mode(self):
        assert self._comparison_mode is not None
        return self._comparison_mode

    # Below: changes due to 3d metric dict and no ._metric
    def setup_metrics(self):
        langs = self.hparams.trn_langs + self.hparams.val_langs + self.hparams.tst_langs
        langs = sorted(list(set(langs)))
        for phase in self.run_phases:
            for lang in langs:
                self.metrics[phase][lang] = {}
                self.custom_logs[phase][lang] = []
                for metric_key in self.metric_names:
                    metric = deepcopy(self.name_to_metric[metric_key]) if metric_key in self.name_to_metric else AverageMetric(metric_key)
                    self.metrics[phase][lang][metric_key] = metric 

    # Below: changed to permit train logging
    def reset_metrics(self, phase):
        for all_metrics in self.metrics[phase].values():
            for metric in all_metrics.values():
                metric.reset()

    def get_mask(self, sent: Tensor):
        mask = (sent != self.tokenizer.pad_token_id).long()
        return mask
    
    # Changed below to make the model explicitly specified.
    def encode_sent(
        self,
        mbert: transformers.PreTrainedModel,
        sent: Tensor,
        # added below
        start_indices : Tensor,
        end_indices : Tensor,
        # end changes
        langs: Optional[List[str]] = None,
        segment: Optional[Tensor] = None
    ):
        mask = self.get_mask(sent)
        import time; start_time = time.time()
        output = mbert(input_ids=sent, attention_mask=mask, token_type_ids=segment)
        print(f'mbert forward: {time.time() - start_time}')
        hs = self.process_feature(output['hidden_states'])
        # Below: added averaging.
        import time; start_time = time.time()
        averaged_hs = collate.average_embeddings(hs, start_indices, end_indices)
        print(f'average embeddings: {time.time() - start_time}')
        return averaged_hs
        # end changes

    def process_feature(self, hidden_states: List[Tensor]):
        if not isinstance(hidden_states, tuple):
            assert len(hidden_states[0].shape) == 2, hidden_states.shape[0]
        if self.concat_all_hidden_states:
            hs: Tensor = torch.cat(hidden_states, dim = -1)
        else:
            hs = hidden_states[self.hparams.feature_layer]
        return hs

    # Renamed variables, function, direct return of loss_dict, no self.log for loss
    # Updated assert message and metrics indexing
    def step_helper(self, batch, prefix):
        import time
        start_time = time.time()
        loss_dict, encoder_outputs, _ = self.__call__(batch)
        assert (
            len(set(batch["lang"])) == 1
        ), "batch should contain only one language"
        lang = batch["lang"][0]
        
        for acc_key, encoder_outputs in encoder_outputs.items():
            labels_key = f'{acc_key}_labels'
            accuracy_type_metric_args = (batch[labels_key], encoder_outputs)
            pos_metric_args = (batch[labels_key], encoder_outputs)
            import time; start_time = time.time()
            self.metrics[prefix][lang][f'{acc_key}_acc'].add(*accuracy_type_metric_args)
            print('after the add', time.time() - start_time)
            
        number_of_true_labels = (batch['pos_labels'] != LABEL_PAD_ID).sum()

        assert all(map(lambda s : 'acc' not in s, loss_dict.keys())), loss_dict.keys()
        start_time = time.time()
        for metric_key in loss_dict:
            if metric_key in 'lang': continue
            value = loss_dict[metric_key]
            self.metrics[prefix][lang][metric_key].add(value, number_of_true_labels)
        print('adding non-accuracy', time.time() - start_time)
        return loss_dict
    
    def detensor_results(self, metrics):
        return {
            k : v.cpu().item()
            if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

                
    def log_wandb(self, phase, lang_list, loss_dict, batch_idx, dataloader_idx):
        assert len(set(lang_list)) == 1, lang_list
        lang = lang_list[0]
        modifier = f'{phase}_{lang}'
        if not self.trainer.sanity_checking:
            modify_metric = lambda modifier, metric : f"{modifier}_{metric}" if not metric.startswith(modifier) else metric
            running_loss_dict = {}
            for metric_group in self.metric_names:
                if metric_group in {'lang'}: continue
                running_loss_dict.update({
                    modify_metric(modifier, metric) : value
                    for metric, value in self.metrics[phase][lang][metric_group].get_metric().items()
                })
            if phase == 'train':
                self.custom_logs[phase][lang].append(self.detensor_results(running_loss_dict))
            modified_loss_dict = {
                modify_metric(modifier, key) if key in self.metric_names else key : value
                for key, value in running_loss_dict.items()
            }
            for metric_key, metric_value in modified_loss_dict.items():
                self.log(metric_key, metric_value)

    # Moved from model/tagger.py.    
    # Changed below to be compatible with later models' loss_dict
    # and to do accuracy updates and wandb logging
    # Removed train self.log of loss
    def training_step(self, batch, batch_idx):
        import time
        start_time = time.time()
        loss_dict = self.step_helper(batch, 'train')
        print(f'step helper: {time.time() - start_time}')
        loss_dict['loss'] = loss_dict[self.optimization_loss]
        # Detach per the warning. Double detach is safe.
        loss_dict = {
            key : value.detach() if key not in ['lang', 'loss'] else value
            for key, value in loss_dict.items()
        }
        start_time = time.time()
        self.log_wandb('train', batch['lang'], loss_dict, batch_idx, None)
        print(f'log wandb: {time.time() - start_time}')
        import pdb; pdb.set_trace()
        return loss_dict
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Do not log initial validation batchwise because would be overwritten by epoch 0 metrics.
        loss_dict = self.step_helper(batch, "val")
        return loss_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step_helper(batch, "tst")
        
    # Added this function
    def add_language_to_batch_output(self, loss_dict, batch):
        assert all(batch['lang'][0] == elem for elem in batch['lang']), set(batch['lang'])
        loss_dict.update({'lang' : batch['lang'][0]})

    # Changed training_epoch_end
    # Removed aggregate_outputs and moved logic to be weighted losses on aggregate_metrics.

    # Changed prefix to phase to mark meaning
    # Added global train step
    def aggregate_metrics(self, langs: List[str], phase: str):
        aver_metric = defaultdict(list)
        for lang in langs:
            current_metrics = { 'trainer/global_step' : self.global_step }
            for metric_key in self.metric_names:
                metric = self.metrics[phase][lang][metric_key]
                for key, val in metric.get_metric().items():
                    logging_key = f"{phase}_{lang}_{key}_epoch"
                    current_metrics.update({logging_key : val})
                    aver_metric[key].append(val)
                    self.log(logging_key, val)
                    if f'{self.monitor_acc_key}_acc' in logging_key:
                        best_key = f'best_{logging_key}'
                        current_val = wandb.summary[best_key] if best_key in wandb.summary.keys() else -float('inf')
                        wandb.summary[best_key] = max(current_val, val)
            # if phase == 'val': # Don't log for train, as it is per-step.
            #     custom_log_dict = self.detensor_results(current_metrics)
            #     self.custom_logs[phase][lang].append(custom_log_dict)
        for key, vals in aver_metric.items():
            self.log(f"{phase}_{key}_all_epoch", torch.stack(vals).mean())

    def training_epoch_end(self, outputs):
        print('in training epoch end')
        self.aggregate_metrics(self.hparams.trn_langs, 'train')
        
    # Added skip sanity check in logging
    # Below functions: fixed strings to phase names
    def validation_epoch_end(self, outputs):
        if self.trainer.sanity_checking: return
        self.aggregate_metrics(self.hparams.val_langs, 'val')

    def test_epoch_end(self, outputs):
        self.aggregate_metrics(self.hparams.tst_langs, Split.test)
        return
    # end changes

    def get_warmup_and_total_steps(self):
        if self.hparams.max_steps is not None:
            max_steps = self.hparams.max_steps
        else:
            max_steps = self.hparams.max_epochs * self.batch_per_epoch
        assert not (
            self.hparams.warmup_steps != -1 and self.hparams.warmup_portion != -1
        )
        if self.hparams.warmup_steps != -1:
            assert self.hparams.warmup_steps > 0
            warmup_steps = self.hparams.warmup_steps
        elif self.hparams.warmup_portion != -1:
            assert 0 < self.hparams.warmup_portion < 1
            warmup_steps = int(max_steps * self.hparams.warmup_portion)
        else:
            warmup_steps = 1
        return warmup_steps, max_steps
        
    # Split configure_optimizers to add two different learning rates
    # but preserve no decay on certain parameters.
    # Split logic to reuse it by moving into split_parameters.
    
    def split_parameters(self, named_parameters, match_templates):
        assert isinstance(match_templates, list), f"Check if {match_templates} is a string."
        not_matches = [
            (n, p)
            for n, p in named_parameters
            if not any(nd in n for nd in match_templates)
        ]
       
        matches = [
            (n, p)
            for n, p in named_parameters
            if any(nd in n for nd in match_templates)
        ]
        return matches, not_matches
        
    
    def split_weight_decay_params(self, model_parameters):
        no_decay = ["bias", "LayerNorm.weight"]
        with_weight_decay_params, no_weight_decay_params = self.split_parameters(model_parameters, no_decay)
        with_weight_decay = {
            "params": with_weight_decay_params,
            "weight_decay": self.hparams.weight_decay,
        }
        no_weight_decay = {
            "params": no_weight_decay_params,
            "weight_decay": 0.0,
        }
        return with_weight_decay, no_weight_decay

    # Split up the learning rates and optimization of bert vs not below
    def configure_optimizers(self):
        named_optimizer_grouped_parameters = []
        other_split_hparams = {
            'mbert' : { 'lr' : self.hparams.mbert_learning_rate },
            'default' : { 'lr' : self.hparams.default_learning_rate },
        }
        mbert_params, default_params = self.split_parameters(list(self.named_parameters()), ['encoder_mbert'])
        split_params = {
            'mbert' : mbert_params,
            'default' : default_params
        }
        for key in split_params:
            by_model_params = split_params[key]
            other_hparams = other_split_hparams[key]
            by_weight_with_decay, by_weight_no_decay = self.split_weight_decay_params(by_model_params)
            
            by_weight_with_decay.update(other_hparams)
            by_weight_no_decay.update(other_hparams)
            
            named_optimizer_grouped_parameters.extend([
                by_weight_with_decay,
                by_weight_no_decay
            ])
        remove_names = lambda param_dict : {
            k : [ p for n, p in v ] if k == 'params' else v
            for k, v in param_dict.items()
        }
        optimizer_grouped_parameters = list(map(remove_names, named_optimizer_grouped_parameters))
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )
        warmup_steps, max_steps = self.get_warmup_and_total_steps()
        if self.hparams.schedule == Schedule.invsqroot:
            scheduler = util.get_inverse_square_root_schedule_with_warmup(
                optimizer, warmup_steps
            )
            interval = "step"
        elif self.hparams.schedule == Schedule.linear:
            scheduler = util.get_linear_schedule_with_warmup(
                optimizer, warmup_steps, max_steps
            )
            interval = "step"
        elif self.hparams.schedule == Schedule.reduceOnPlateau:
            # Below: changed min_lr to 1e-10
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=0, min_lr=1e-10, mode="min"
            )
            interval = "epoch"
        else:
            raise ValueError(self.hparams.schedule)

        self.optimizer = optimizer
        self.scheduler = scheduler
        scheduler_dict = {"scheduler": scheduler, "interval": interval}
        if self.hparams.schedule == Schedule.reduceOnPlateau:
            # Changed below key
            scheduler_dict["monitor"] = f"val_{self.target_language}_{self.optimization_loss}_epoch"
        return [optimizer], [scheduler_dict]
        

    def _get_signature(self, params: Dict):
        def md5_helper(obj):
            return hashlib.md5(str(obj).encode()).hexdigest()

        signature = dict()
        for key, val in params.items():
            if key == "tokenizer" and isinstance(val, transformers.PreTrainedTokenizer):
                signature[key] = md5_helper(list(val.get_vocab().items()))
            else:
                signature[key] = str(val)

        md5 = md5_helper(list(signature.items()))
        return md5, signature

    def prepare_datasets(self, split: str) -> List[Dataset]:
        raise NotImplementedError

    def prepare_datasets_helper(
        self,
        data_class: Type[Dataset],
        langs: List[str],
        split: str,
        max_len: int,
        **kwargs,
    ):
        datasets = []

        for lang in langs:
            filepath = data_class.get_file(self.hparams.data_dir, lang, split)
            if filepath is None:
                print(f"skipping {split} language: {lang}")
                continue
            params = {}
            params["task"] = self.hparams.task
            params["tokenizer"] = self.tokenizer
            params["filepath"] = filepath
            params["lang"] = lang
            params["split"] = split
            params["max_len"] = max_len
            if split == Split.train:
                params["subset_ratio"] = self.hparams.subset_ratio
                params["subset_count"] = self.hparams.subset_count
                params["subset_seed"] = self.hparams.subset_seed
            params.update(kwargs)
            md5, signature = self._get_signature(params)
            del params["task"]
            cache_file = f"{self.hparams.cache_path}/{md5}"
            if self.hparams.cache_dataset and os.path.isfile(cache_file):
                print(f"load from cache {filepath} with {self.hparams.pretrain}")
                dataset = torch.load(cache_file)
            else:
                dataset = data_class(**params)
                if self.hparams.cache_dataset:
                    print(f"save to cache {filepath} with {self.hparams.pretrain}")
                    torch.save(dataset, cache_file)
                    with open(f"{cache_file}.json", "w") as fp:
                        json.dump(signature, fp)
            datasets.append(dataset)
        return datasets

    def train_dataloader(self):
        if self.trn_datasets is None:
            self.trn_datasets = self.prepare_datasets(Split.train)

        # Renamed collate to collate_fn due to import
        collate_fn = partial(util.default_collate, padding=self.padding)
        if len(self.trn_datasets) == 1:
            dataset = self.trn_datasets[0]
            sampler = RandomSampler(dataset)
        else:
            dataset = ConcatDataset(self.trn_datasets)
            # Removed mix_sampling logic.
            sampler = util.ConcatSampler(dataset, self.hparams.batch_size)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.number_of_workers,
        )

    def val_dataloader(self):
            
        if self.val_datasets is None:
            self.val_datasets = self.prepare_datasets(Split.dev)

        # Renamed collate to collate_fn due to import
        collate_fn = partial(util.default_collate, padding=self.padding)
        return [
            DataLoader(
                val_dataset,
                batch_size=self.hparams.eval_batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=self.hparams.number_of_workers,
            )
            for val_dataset in self.val_datasets
        ]

    def test_dataloader(self):
        if self.tst_datasets is None:
            self.tst_datasets = self.prepare_datasets(Split.test)

        # Renamed collate to collate_fn due to import
        collate_fn = partial(util.default_collate, padding=self.padding)
        return [
            DataLoader(
                tst_dataset,
                batch_size=self.hparams.eval_batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=1,
            )
            for tst_dataset in self.tst_datasets
        ]
        
    @classmethod
    def add_layer_stack_args(cls, parser, modifier):
        # Manually made these the same as default latent.
        parser.add_argument(f"--{modifier}_hidden_layers", default=1, type=int)
        parser.add_argument(f"--{modifier}_hidden_size", default=64, type=int)
        parser.add_argument(f"--{modifier}_model_type", default='linear', type=str)
        return parser

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        # Changes: Removed irrelevant arguments generally here.
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        # shared
        # below: changed to default
        parser.add_argument("--task", default="udpos", type=str)
        parser.add_argument("--data_dir", required=True, type=str)
        parser.add_argument("--trn_langs", required=True, nargs="+", type=str)
        parser.add_argument("--val_langs", required=True, nargs="+", type=str)
        parser.add_argument("--tst_langs", default=[], nargs="*", type=str)
        parser.add_argument("--max_trn_len", default=128, type=int)
        parser.add_argument("--max_tst_len", default=128, type=int)
        parser.add_argument("--subset_ratio", default=1.0, type=float)
        parser.add_argument("--subset_count", default=-1, type=int)
        parser.add_argument("--subset_seed", default=42, type=int)
        # Removed mix_sampling.
        # encoder
        # Changed pretrain to be set to default.
        parser.add_argument("--pretrain", default="bert-base-multilingual-cased", type=str)
        parser.add_argument("--freeze_layer", default=-1, type=int)
        parser.add_argument("--feature_layer", default=-1, type=int)
        # Below additions
        parser.add_argument("--use_hidden_layer", default=False, type=util.str2bool)
        parser.add_argument("--hidden_layer_size", default=-1, type=int)
        # end additions
        # Below line: changed from providing weighted features to a concatenated all hidden states
        parser.add_argument("--concat_all_hidden_states", default=False, type=util.str2bool)
        # Changed to remove all types of dropout
        # misc
        parser.add_argument("--seed", default=42, type=int)
        # Split the learning rates below
        parser.add_argument("--mbert_learning_rate", default=5e-5, type=float)
        parser.add_argument("--default_learning_rate", default=5e-5, type=float)
        # Changed below beta2 parameter to match the paper
        parser.add_argument("--adam_beta2", default=0.999, type=float)
        parser.add_argument("--adam_eps", default=1e-8, type=float)
        # Changed below weight_decay parameter to match the paper
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--schedule", default=Schedule.linear, choices=Schedule().choices(), type=str)
        parser.add_argument("--warmup_steps", default=-1, type=int)
        # Changed below warmup portion to match the paper.
        parser.add_argument("--warmup_portion", default=0.1, type=float)
        # fmt: on
        # Added below
        parser.add_argument("--number_of_workers", default=1, type=int)
        parser.add_argument("--freeze_mbert", default=False, type=util.str2bool)
        return parser
