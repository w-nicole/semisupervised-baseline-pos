# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed forward to __call__.
# Added one layer MLP.
# Removed irrelevant code, such as outdated classes, imports, etc.
# and simplified to remove unused parameter choices.
# Added support for mBERT initialization for upper baselines.
# Changed to consider epoch metrics instead for equivalent checkpointing.
# Changed checkpointing metric for compatibility with wandb logging.
# Removed _metric.
# Split logic into `base_tagger` and `tagger` files.

from copy import deepcopy
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial # added this from model/base.py

import util
from metric import LABEL_PAD_ID # changed this from dataset import
from dataset import Dataset, UdPOS, SupervisedUdPOS, UnsupervisedUdPOS
from enumeration import Split, Task
from model.base import Model

# added below
import constant
from dataset import tagging
from torch.utils.data import DataLoader
# end added

class BaseTagger(Model):
    def __init__(self, hparams):
        super(BaseTagger, self).__init__(hparams)
        self._nb_labels: Optional[int] = None
        self._nb_labels = UdPOS.nb_labels()
        
        # Added/edited
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "lang": 0,
            "pos_labels": LABEL_PAD_ID,
            "token_labels" : LABEL_PAD_ID,
            "is_supervised" : LABEL_PAD_ID,
            # end changes
        }

    @property
    def nb_labels(self):
        assert self._nb_labels is not None
        return self._nb_labels

    # Moved training_step to base.py.
    
    # Removed loss logging in original evaluation_step_helper
    # because handled in event and not checkpointing on it.

    def prepare_datasets(self, split: str) -> List[Dataset]:
        hparams = self.hparams
        data_class_dict = { 'supervised' : SupervisedUdPOS, 'unsupervised' : UnsupervisedUdPOS }
        if split == Split.train:
            return self.prepare_datasets_helper(
                data_class_dict, hparams.trn_langs, Split.train, hparams.max_trn_len
            )
        elif split == Split.dev:
            return self.prepare_datasets_helper(
                data_class_dict, hparams.val_langs, Split.dev, hparams.max_tst_len
            )
        elif split == Split.test:
            return self.prepare_datasets_helper(
                data_class_dict, hparams.tst_langs, Split.test, hparams.max_tst_len
            )
        else:
            raise ValueError(f"Unsupported split: {hparams.split}")
        
    # Below: added
    def get_labels(self, lang, split):
        dataset = self.get_dataset(lang, split)
        train_data = dataset.read_file(dataset.filepath, dataset.lang, dataset.split)
        labels = []
        for data in train_data:
            labels.extend(data['labels'])
        numerical_labels = torch.Tensor(list(map(lambda label : dataset.label2id[label], labels))).int()
        return numerical_labels

    def get_label_counts(self, lang, split):
        numerical_labels = self.get_labels(lang, split)    
        counts = torch.bincount(numerical_labels, minlength = self.nb_labels)
        return counts
        
    def get_dataset(self, lang, split):
        # From model/base.py, adapted to simplify and get English dataset
        params = {}
        params["tokenizer"] = self.tokenizer
        params["filepath"] = tagging.UdPOS.get_file(self.hparams.data_dir, lang, split)
        params["lang"] = lang
        params["split"] = split
        params["max_len"] = self.hparams.max_trn_len
        params["subset_ratio"] = self.hparams.subset_ratio
        params["subset_count"] = self.hparams.subset_count
        params["subset_seed"] = self.hparams.subset_seed
        return tagging.UdPOS(**params)
        # end taken
        
    def get_dataloader(self, lang, split):
        # Adapted from model/base.py
        collate_fn = partial(util.default_collate, padding=self.padding)
        return DataLoader(
                self.get_dataset(lang, split),
                batch_size=self.hparams.eval_batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=1,
        )
        # end adapted
        
    def calculate_encoder_loss(self, pos_labels, log_probs):
        encoder_loss = F.nll_loss(
            log_probs.view(-1, self.nb_labels),
            pos_labels.view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        return encoder_loss
        
    # end added