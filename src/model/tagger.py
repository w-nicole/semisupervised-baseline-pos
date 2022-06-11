
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Added averaging behavior,
#   and changed "forward" accordingly.
# Added one layer MLP.
# Removed irrelevant code, such as outdated classes, imports, etc.

from copy import deepcopy
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial # added this from model/base.py

import util
from dataset import LABEL_PAD_ID, ConllNER, Dataset, UdPOS, WikiAnnNER
from enumeration import Split, Task
from metric import NERMetric, POSMetric, convert_bio_to_spans
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

        self._comparison = {
            Task.conllner: "max",
            Task.wikiner: "max",
            Task.udpos: "max",
        }[self.hparams.task]
        self._selection_criterion = {
            Task.conllner: "val_f1",
            Task.wikiner: "val_f1",
            Task.udpos: "val_acc",
        }[self.hparams.task]
        self._nb_labels: Optional[int] = None
        self._nb_labels = {
            Task.conllner: ConllNER.nb_labels(),
            Task.wikiner: WikiAnnNER.nb_labels(),
            Task.udpos: UdPOS.nb_labels(),
        }[self.hparams.task]
        self._metric = {
            Task.conllner: NERMetric(ConllNER.get_labels()),
            Task.wikiner: NERMetric(WikiAnnNER.get_labels()),
            Task.udpos: POSMetric(),
        }[self.hparams.task]

        self.id2label = {
            Task.conllner: ConllNER.get_labels(),
            Task.wikiner: WikiAnnNER.get_labels(),
            Task.udpos: UdPOS.get_labels(),
        }[self.hparams.task]

        if self.hparams.tagger_use_crf:
            self.crf = ChainCRF(self.hidden_size, self.nb_labels, bigram=True)
        # Added/edited
        elif self.hparams.use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hparams.hidden_layer_size),
                nn.Linear(self.hparams.hidden_layer_size, self.nb_labels)
            )
        # end additions
        else:
            self.classifier = nn.Linear(self.hidden_size, self.nb_labels)
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "lang": 0,
            # Added below. MUST match START_END_INDEX_PADDING for logic to hold.
            "start_indices": constant.START_END_INDEX_PADDING,
            "end_indices": constant.START_END_INDEX_PADDING,
            # end changes
            "labels": LABEL_PAD_ID,
        }

        self.setup_metrics()

    @property
    def nb_labels(self):
        assert self._nb_labels is not None
        return self._nb_labels

    def preprocess_batch(self, batch):
        return batch

    def forward(self, batch):
        batch = self.preprocess_batch(batch)
        # Updated call arguments
        hs = self.encode_sent(batch["sent"], batch["start_indices"], batch["end_indices"], batch["lang"])
        # end updates
        if self.hparams.tagger_use_crf:
            mask = (batch["labels"] != LABEL_PAD_ID).float()
            energy = self.crf(hs, mask=mask)
            target = batch["labels"].masked_fill(
                batch["labels"] == LABEL_PAD_ID, self.nb_labels
            )
            loss = self.crf.loss(energy, target, mask=mask)
            log_probs = energy
        else:
            logits = self.classifier(hs)
            log_probs = F.log_softmax(logits, dim=-1)

            loss = F.nll_loss(
                log_probs.view(-1, self.nb_labels),
                batch["labels"].view(-1),
                ignore_index=LABEL_PAD_ID,
            )
        return loss, log_probs

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log("loss", loss)
        return loss

    def evaluation_step_helper(self, batch, prefix):
        loss, log_probs = self.forward(batch)

        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]
        if self.hparams.tagger_use_crf:
            energy = log_probs
            prediction = self.crf.decode(
                energy, mask=(batch["labels"] != LABEL_PAD_ID).float()
            )
            self.metrics[lang].add(batch["labels"], prediction)
        else:
            self.metrics[lang].add(batch["labels"], log_probs)

        result = dict()
        result[f"{prefix}_{lang}_loss"] = loss
        return result

    def prepare_datasets(self, split: str) -> List[Dataset]:
        hparams = self.hparams
        data_class: Type[Dataset]
        if hparams.task == Task.conllner:
            data_class = ConllNER
        elif hparams.task == Task.wikiner:
            data_class = WikiAnnNER
        elif hparams.task == Task.udpos:
            data_class = UdPOS
        else:
            raise ValueError(f"Unsupported task: {hparams.task}")

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
        parser.add_argument("--tagger_use_crf", default=False, type=util.str2bool)
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
        counts = torch.bincount(numerical_labels, minlength = int(torch.max(numerical_labels)))
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
