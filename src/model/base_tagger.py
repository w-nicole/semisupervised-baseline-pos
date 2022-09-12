# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

from copy import deepcopy
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
import numpy as np

import util
from dataset.base import Dataset
from dataset.tagging import UdPOS
from constant import LABEL_PAD_ID
from enumeration import Split, Task
from model.base import Model
import constant

class BaseTagger(Model):
    def __init__(self, hparams):
        super(BaseTagger, self).__init__(hparams)
        self._nb_labels: Optional[int] = None
        self._nb_labels = UdPOS.nb_labels()
        
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "lang": 0,
            "pos_labels": LABEL_PAD_ID
        }

    @property
    def nb_labels(self):
        assert self._nb_labels is not None
        return self._nb_labels

    def prepare_datasets(self, split: str) -> List[Dataset]:
        hparams = self.hparams
        if split == Split.train:
            return self.prepare_datasets_helper(
                hparams.trn_langs, Split.train, hparams.max_trn_len
            )
        elif split == Split.dev:
            return self.prepare_datasets_helper(
                hparams.val_langs, Split.dev, hparams.max_tst_len
            )
        elif split == Split.test:
            return self.prepare_datasets_helper(
                hparams.tst_langs, Split.test, hparams.max_tst_len
            )
        else:
            raise ValueError(f"Unsupported split: {hparams.split}")
        
    # Below: added
    def get_flat_labels(self, lang, split):
        dataset = self.get_dataset_by_lang_split(lang, split)
        labels = torch.Tensor(np.concatenate([example['pos_labels'] for example in dataset], axis = 0)).int()
        return labels

    def get_label_counts(self, lang, split):
        numerical_labels = self.get_flat_labels(lang, split)
        clean_numerical_labels = numerical_labels[numerical_labels != LABEL_PAD_ID]
        counts = torch.bincount(clean_numerical_labels, minlength = self.nb_labels)
        return counts
        
    def get_unshuffled_dataloader(self, lang, split):
        # Adapted from model/base.py
        collate_fn = partial(util.default_collate, padding=self.padding)
        return DataLoader(
            self.get_dataset_by_lang_split(lang, split if split != 'val' else Split.dev),
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
            ignore_index=self.padding['pos_labels'],
        )
        return encoder_loss
        
    # end added