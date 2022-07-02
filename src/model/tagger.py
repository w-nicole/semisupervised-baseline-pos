
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Added averaging behavior,
#   and changed "forward" accordingly.
# Changed forward to __call__.
# Added one layer MLP.
# Removed irrelevant code, such as outdated classes, imports, etc.
# and simplified to remove unused parameter choices.
# Added support for mBERT initialization for upper baselines.
# Changed to consider epoch metrics instead for equivalent checkpointing.
# Changed checkpointing metric for compatibility with wandb logging.

from copy import deepcopy
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial # added this from model/base.py

import util
from metric import LABEL_PAD_ID # changed this from dataset import
from dataset import Dataset, UdPOS
from enumeration import Split, Task
from metric import POSMetric
from model.base import Model

# added below
import constant
from dataset import tagging
from torch.utils.data import DataLoader
# end added

# Below: added imports
from dataset import collate
# end imports


class Tagger(Model):
    def __init__(self, hparams):
        super(Tagger, self).__init__(hparams)
        self._comparison_mode = 'max'
        self._selection_criterion = f'val_{self.target_language}_acc_epoch_ckpts'
        self._nb_labels: Optional[int] = None
        self._nb_labels = UdPOS.nb_labels()
        self._metric = POSMetric()
        
        self.is_frozen_mbert = self.hparams.mbert_checkpoint or self.hparams.freeze_mbert
        assert not ((not self.hparams.freeze_mbert) and self.hparams.mbert_checkpoint),\
            "Mutually exclusive. mbert_checkpoint always automatically frozen."
            
        # Reinitialize mBERT alone if given a checkpoint.
        if self.hparams.mbert_checkpoint:
            print('got to here!')
            encoder = Tagger.load_from_checkpoint(self.hparams.mbert_checkpoint)
            self.freeze_bert(encoder)
            self.model = encoder.model
            self.concat_all_hidden_states = encoder.concat_all_hidden_states
        
        if self.hparams.freeze_mbert:
            self.freeze_bert(self)
        # end additions

        self.id2label = UdPOS.get_labels()
        self.classifier = self.build_classifier()
        # Added/edited
        
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "lang": 0,
            # Added below. MUST match START_END_INDEX_PADDING for logic to hold.
            "start_indices": constant.START_END_INDEX_PADDING,
            "end_indices": constant.START_END_INDEX_PADDING,
            # end changes
            "labels": LABEL_PAD_ID,
        }
        # optimization loss added
        self.optimization_loss = 'pos_nll_loss'
        self.setup_metrics()
        
    # Below: added
    def build_classifier(self):
        if self.hparams.encoder_hidden_layers >= 0:
            return self.build_mlp(
                self.hidden_size, self.nb_labels,
                self.hparams.encoder_hidden_size, self.hparams.encoder_hidden_layers,
                self.hparams.encoder_nonlinear_first
            )
        else:
            return nn.Linear(self.hidden_size, self.nb_labels)

    @property
    def nb_labels(self):
        assert self._nb_labels is not None
        return self._nb_labels

    def preprocess_batch(self, batch):
        return batch

    def __call__(self, batch):
        if self.is_frozen_mbert:
            self.model.eval()
            
        batch = self.preprocess_batch(batch)
        # Updated call arguments
        hs = self.encode_sent(batch["sent"], batch["start_indices"], batch["end_indices"], batch["lang"])
        # end updates
        # removed use_crf
        logits = self.classifier(hs)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            log_probs.view(-1, self.nb_labels),
            batch["labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        # Changed below to be compatible with later models' loss_dict and added assert.
        loss_dict = {self.optimization_loss : loss}
        self.add_language_to_batch_output(loss_dict, batch)
        return loss_dict, log_probs

    # Moved training_step to base.py.
    
    # Removed loss logging in original evaluation_step_helper
    # because handled in event and not checkpointing on it.

    def prepare_datasets(self, split: str) -> List[Dataset]:
        hparams = self.hparams
        data_class = UdPOS
        if split == Split.train:
            return self.prepare_datasets_helper(
                data_class, hparams.trn_langs, Split.train, hparams.max_trn_len
            )
        elif split == Split.dev:
            return self.prepare_datasets_helper(
                data_class, hparams.val_langs, Split.dev, hparams.max_tst_len
            )
        elif split == Split.test:
            return self.prepare_datasets_helper(
                data_class, hparams.tst_langs, Split.test, hparams.max_tst_len
            )
        else:
            raise ValueError(f"Unsupported split: {hparams.split}")

    @classmethod
    def add_model_specific_args(cls, parser):
        # Added these arguments, removed crf argument
        # -1 indicates a linear layer alone (no input layer).
        parser = Model.add_model_specific_args(parser)
        parser.add_argument("--encoder_hidden_layers", default=-1, type=int)
        parser.add_argument("--encoder_hidden_size", default=0, type=int)
        parser.add_argument("--encoder_nonlinear_first", default=False, type=util.str2bool)
        parser.add_argument("--freeze_mbert", default=False, type=util.str2bool)
        # No path indicates a fresh initialization from huggingface
        parser.add_argument("--mbert_checkpoint", default="", type=str)
        return parser
        
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
        
    # end added
