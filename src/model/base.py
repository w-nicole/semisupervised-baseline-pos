
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed hyperparameters to match those in the paper,
# Added averaging behavior.
# Changed to not support weighted features, but instead a concatenation of all hidden representations.
# Changed to support single hidden layer MLP.
# Changed `comparsion` to `comparison_mode`
# Added support for logging train metrics.
# Changed forward to __call__.

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
from metric import Metric
from model.module import Identity, InputVariationalDropout, MeanPooling, Transformer

# Below: added imports
from dataset import collate
import wandb
# end imports

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.optimizer = None
        self.scheduler = None
        self._metric: Optional[Metric] = None
        # Changed below to account for train.
        self.metrics: Dict[str, Dict[str, Metric]] = defaultdict(dict)
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
        # self.hparams: Namespace = hparams
        self.save_hyperparameters(hparams)
        pl.seed_everything(hparams.seed)
        
        # Added the following
        self.run_phases = [Split.train, 'val', Split.test]
        
        one_other_target_language = (len(self.hparams.val_langs) == 2 and constant.SUPERVISED_LANGUAGE in self.hparams.val_langs)
        valid_val_langs = len(self.hparams.val_langs) == 1 or one_other_target_language
        assert valid_val_langs, "target_language/checkpoint was designed with at most 1 non-source language."
        # Phase 2 and 3: If it's a pure optimization, then use that language
        if len(self.hparams.trn_langs) == 1:
            self.target_language = self.hparams.trn_langs[0]
        # Otherwise, if mixed training, favor the non-English language
        elif one_other_target_language:
            not_supervised_languages = list(filter(lambda lang : lang != constant.SUPERVISED_LANGUAGE, self.hparams.val_langs))
            assert len(not_supervised_languages) == 1
            self.target_language = not_supervised_languages[0]
        else:
            assert False, "This case for checkpoint language was not considered."
        # end additions

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrain)
        # Changed below to correspond to classmethod
        self.model = self.build_model(self.hparams.pretrain)
        self.freeze_layers()

        self.mapping = None
        if hparams.mapping:
            assert os.path.isfile(hparams.mapping)
            self.mapping = torch.load(hparams.mapping)
            util.freeze(self.mapping)

        # Changed below line
        self.projector = None
        self.dropout = InputVariationalDropout(hparams.input_dropout)

    # Changed below to accept pretrain as argument so classmethod works.
    @classmethod
    def build_model(self, pretrain):
        config = AutoConfig.from_pretrained(
            pretrain, output_hidden_states=True
        )
        model = AutoModel.from_pretrained(pretrain, config=config)
        return model
        
    def build_mlp(self, input_size, output_size, hidden_size, hidden_layers, nonlinear_first):
        layers = []
        if nonlinear_first:
            layers.append(torch.nn.GELU())
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
        if isinstance(self.model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            util.freeze(self.model.embeddings)
        elif isinstance(self.model, transformers.XLMModel):
            util.freeze(self.model.position_embeddings)
            if self.model.n_langs > 1 and self.model.use_lang_emb:
                util.freeze(self.model.lang_embeddings)
            util.freeze(self.model.embeddings)
        else:
            raise ValueError("Unsupported model")
            
    def freeze_bert(self, encoder):
        # Adapted from model/base.py by taking the logic to freeze up to and including a certain layer
        # Doesn't freeze the pooler, but encode_sent excludes pooler correctly.
        encoder.freeze_embeddings()
        for index in range(encoder.model.config.num_hidden_layers + 1):
            encoder.freeze_layer(index)
        # end adapted
        
    def freeze_layer(self, layer):
        if isinstance(self.model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            util.freeze(self.model.encoder.layer[layer - 1])
        elif isinstance(self.model, transformers.XLMModel):
            util.freeze(self.model.attentions[layer - 1])
            util.freeze(self.model.layer_norm1[layer - 1])
            util.freeze(self.model.ffns[layer - 1])
            util.freeze(self.model.layer_norm2[layer - 1])
        else:
            raise ValueError("Unsupported model")

    @property
    def hidden_size(self):
        # hidden_size = the input to the classifier
        if isinstance(self.model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            # Added logic for concatenated embeddings
            single_layer_size = self.model.config.hidden_size
            if not self.hparams.concat_all_hidden_states:
                return single_layer_size
            else:
                return single_layer_size * (self.model.config.num_hidden_layers + 1)
            # end added
        elif isinstance(self.model, transformers.XLMModel):
            return self.model.dim
        else:
            raise ValueError("Unsupported model")

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

    # Below: changes due to 2d metric dict
    def setup_metrics(self):
        assert self._metric is not None
        langs = self.hparams.trn_langs + self.hparams.val_langs + self.hparams.tst_langs
        langs = sorted(list(set(langs)))
        for phase in self.run_phases:
            for lang in langs:
                self.metrics[phase][lang] = deepcopy(self._metric)

    # Below: changed to permit train logging
    def reset_metrics(self, phase):
        for metric in self.metrics[phase].values():
            metric.reset()

    def get_mask(self, sent: Tensor):
        mask = (sent != self.tokenizer.pad_token_id).long()
        return mask

    def encode_sent(
        self,
        sent: Tensor,
        # added below
        start_indices : Tensor,
        end_indices : Tensor,
        # end changes
        langs: Optional[List[str]] = None,
        segment: Optional[Tensor] = None,
        model: Optional[transformers.PreTrainedModel] = None,
        return_raw_hidden_states: bool = False,
    ):
        if model is None:
            model = self.model
        mask = self.get_mask(sent)
        if isinstance(model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            output = model(input_ids=sent, attention_mask=mask, token_type_ids=segment)
        elif isinstance(model, transformers.XLMModel):
            lang_ids: Optional[torch.Tensor]
            if langs is not None:
                try:
                    batch_size, seq_len = sent.shape
                    lang_ids = torch.tensor(
                        [self.tokenizer.lang2id[lang] for lang in langs],
                        dtype=torch.long,
                        device=sent.device,
                    )
                    lang_ids = lang_ids.unsqueeze(1).expand(batch_size, seq_len)
                except KeyError as e:
                    print(f"KeyError with missing language {e}")
                    lang_ids = None
            output = model(
                input_ids=sent,
                attention_mask=mask,
                langs=lang_ids,
                token_type_ids=segment,
            )
        else:
            raise ValueError("Unsupported model")

        if return_raw_hidden_states:
            return output["hidden_states"]

        hs = self.map_feature(output["hidden_states"], langs)
        hs = self.process_feature(hs)
        hs = self.dropout(hs)
        # Removed projector.
        
        # Below: added averaging.
        averaged_hs = collate.average_embeddings(hs, start_indices, end_indices)
        return averaged_hs
        # end changes

    def map_feature(self, hidden_states: List[Tensor], langs):
        if self.mapping is None:
            return hidden_states

        assert len(set(langs)) == 1, "a batch should contain only one language"
        lang = langs[0]
        lang = constant.LANGUAGE_TO_ISO639.get(lang, lang)
        if lang not in self.mapping:
            return hidden_states

        hs = []
        for h, m in zip(hidden_states, self.mapping[lang]):
            hs.append(m(h))
        return hs

    def process_feature(self, hidden_states: List[Tensor]):
        if not isinstance(hidden_states, tuple):
            assert len(hidden_states[0].shape) == 2, hidden_states.shape[0]
        if self.hparams.concat_all_hidden_states:
            hs: Tensor = torch.cat(hidden_states, dim = -1)
        else:
            hs = hidden_states[self.hparams.feature_layer]
        return hs

    # Renamed variables, function, direct return of loss_dict, no self.log for loss
    # Updated assert message and metrics indexing
    def step_helper(self, batch, prefix):
        loss_dict, encoder_outputs = self.__call__(batch)
        assert (
            len(set(batch["lang"])) == 1
        ), "batch should contain only one language"
        lang = batch["lang"][0]
        self.metrics[prefix][lang].add(batch["labels"], encoder_outputs)
        return loss_dict
    
    def log_wandb(self, phase, lang, loss_dict):
        assert len(lang) > 0, lang
        if self.hparams.log_wandb and not self.trainer.sanity_checking:
            loss_dict = { f"{phase}_{lang[0]}_{metric}" : value for metric, value in loss_dict.items() }
            wandb.log(loss_dict)
        
    # Moved from model/tagger.py.    
    # Changed below to be compatible with later models' loss_dict
    # and to do accuracy updates and wandb logging
    def training_step(self, batch, batch_idx): 
        loss_dict = self.step_helper(batch, 'train')
        loss_dict['loss'] = loss_dict[self.optimization_loss]
        # Detach per the warning. Double detach is safe.
        loss_dict = {
            key : value.detach() if key not in ['lang', 'loss'] else value
            for key, value in loss_dict.items()
        }
        self.log("loss", loss_dict['loss'])
        if batch_idx % self.hparams.log_frequency == 0:
            self.log_wandb('train', batch['lang'], loss_dict)
        return loss_dict
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self.step_helper(batch, "val")
        self.log("val_loss", loss_dict[self.optimization_loss])
        if batch_idx % self.hparams.log_frequency == 0:
            self.log_wandb('val', batch['lang'], loss_dict)
        return loss_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step_helper(batch, "tst")
        
    # Added this function
    def add_language_to_batch_output(self, loss_dict, batch):
        assert all(batch['lang'][0] == elem for elem in batch['lang']), set(batch['lang'])
        loss_dict.update({'lang' : batch['lang'][0]})

    # Changed training_epoch_end

    # Changed all of below to account for language sorting and training logging,
    # as well as updates to arguments to reflect meaning of parameters.
    # Changes to logging for phase and language separation,
    # and renaming of language-aggregated metrics to '_all' modifier,
    # and ignoring added language differentiator from forward call
    def aggregate_outputs(
        self, outputs: List[List[Dict[str, Tensor]]], langs: List[str], phase: str
    ):
        aver_result = defaultdict(list)
        lang_metrics = defaultdict(list)
        for lang, output in zip(langs, outputs):
            for key in output[0]:
                if 'lang' in key or not isinstance(output[0][key], torch.Tensor): continue
                try:
                    mean_val = torch.stack([x[key] for x in output]).mean()
                except: import pdb; pdb.set_trace()
                logging_key = f'{phase}_{lang}_{key}'
                self.log(logging_key, mean_val)
                
                raw_key = logging_key.replace(f"{lang}_", "")
                aver_result[raw_key].append(mean_val)

        for key, vals in aver_result.items():
            self.log(f'{key}_all', torch.stack(vals).mean())

    # Changed prefix to phase to mark meaning
    def aggregate_metrics(self, langs: List[str], phase: str):
        aver_metric = defaultdict(list)
        for lang in langs:
            metric = self.metrics[phase][lang]
            for key, val in metric.get_metric().items():
                self.log(f"{phase}_{lang}_{key}", val)
                aver_metric[key].append(val)
                logging_key = f"{phase}_{lang}_{key}"
                if self.hparams.log_wandb and not self.trainer.sanity_checking:
                    wandb.log({logging_key : val})

        for key, vals in aver_metric.items():
            self.log(f"{phase}_{key}", torch.stack(vals).mean())
            
    # Added training_epoch_end, adapted from the epoch_end functions below,
    # which has output sorting by language
    def training_epoch_end(self, outputs):
        if len(self.hparams.trn_langs) == 1:
            filtered_outputs = [outputs]
        else:
            filtered_outputs = [[] for _ in self.hparams.trn_langs]
            # Need to filter the outputs, such that they belong to a single language
            lang_to_index = { lang : index for index, lang in enumerate(self.hparams.trn_langs) } 
            for batch_outputs in outputs:
                lang = batch_outputs['lang']
                filtered_outputs[lang_to_index[lang]].append(batch_outputs)
        # Enforce correctness of filtered outputs
        for lang_index, lang_outputs in enumerate(filtered_outputs):
            for batch_index, batch_outputs in enumerate(lang_outputs):
                if self.hparams.trn_langs[lang_index] != batch_outputs['lang']:
                    import pdb; pdb.set_trace()
        self.aggregate_outputs(filtered_outputs, self.hparams.trn_langs, Split.train)
        self.aggregate_metrics(self.hparams.trn_langs, Split.train)
        return

    # Below functions: fixed strings to phase names
    def validation_epoch_end(self, outputs):
        if len(self.hparams.val_langs) == 1:
            outputs = [outputs]
        self.aggregate_outputs(outputs, self.hparams.val_langs, 'val')
        self.aggregate_metrics(self.hparams.val_langs, 'val')
        return

    def test_epoch_end(self, outputs):
        if len(self.hparams.tst_langs) == 1:
            outputs = [outputs]
        self.aggregate_outputs(outputs, self.hparams.tst_langs, Split.test)
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

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            }
        )
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            }
        )

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
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
            # Changed below to match new keys.
            scheduler_dict["monitor"] = f'val_{self.target_language}_{self.optimization_loss}'
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
            num_workers=1,
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
                num_workers=1,
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
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        # shared
        parser.add_argument("--task", required=True, choices=Task().choices(), type=str)
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
        parser.add_argument("--pretrain", required=True, type=str)
        parser.add_argument("--freeze_layer", default=-1, type=int)
        parser.add_argument("--feature_layer", default=-1, type=int)
        # Below additions
        parser.add_argument("--use_hidden_layer", default=False, type=util.str2bool)
        parser.add_argument("--hidden_layer_size", default=-1, type=int)
        # end additions
        # Below line: changed from providing weighted features to a concatenated all hidden states
        parser.add_argument("--concat_all_hidden_states", default=False, type=util.str2bool)
        parser.add_argument("--projector", default="id", choices=["id", "meanpool", "transformer"], type=str)
        parser.add_argument("--projector_trm_hidden_size", default=3072, type=int)
        parser.add_argument("--projector_trm_num_heads", default=12, type=int)
        parser.add_argument("--projector_trm_num_layers", default=4, type=int)
        # Changed to remove all types of dropout that are specified separate from BERT (i.e. set them to zero)
        parser.add_argument("--projector_dropout", default=0, type=float)
        parser.add_argument("--input_dropout", default=0, type=float)
        parser.add_argument("--mapping", default="", type=str)
        # misc
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
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
        return parser
