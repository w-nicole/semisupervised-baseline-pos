
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

import glob
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import constant
from dataset.base import DUMMY_LABEL, Dataset
from enumeration import Split
from metric import LABEL_PAD_ID
from collections import defaultdict

import torch

class TaggingDataset(Dataset):
    def before_load(self):
        self.max_len = min(self.max_len, self.tokenizer.max_len_single_sentence)
        self.shift = self.max_len // 2
        self.labels = self.get_labels()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.label2id[DUMMY_LABEL] = LABEL_PAD_ID

    @classmethod
    def nb_labels(cls):
        return len(cls.get_labels())

    @classmethod
    def get_labels(cls) -> List[str]:
        raise NotImplementedError

    # Changed this function to not consider labels.
    def add_special_tokens(self, sent):
        sent = self.tokenizer.build_inputs_with_special_tokens(sent)
        return np.array(sent)
    # end changes
    
    def process_labels_for_return(self, raw_labels, mask):
        labels = np.array(self.tokenizer.build_inputs_with_special_tokens(raw_labels))
        masked_labels = labels * (1 - mask) + LABEL_PAD_ID * mask
        return masked_labels
    
    def process_example_for_return(self, sent, all_label_ids):
        sent = self.tokenizer.build_inputs_with_special_tokens(sent)
        mask = np.array(self.tokenizer.get_special_tokens_mask(
            sent, already_has_special_tokens=True
        ))
        sent = np.array(sent)
        process_with_mask = lambda labels : self.process_labels_for_return(labels, mask)
        masked_labels_dict = {
            label_type : process_with_mask(labels)
            for label_type, labels in all_label_ids.items()
        } 
        masked_labels = tuple(masked_labels_dict[k] for k in sorted(masked_labels_dict.keys()))
        return (sent,) + masked_labels

    def _process_example_helper(
        self, sent: List, labels: List
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        
        token_ids: List[int] = []
        all_label_ids = {'pos' : [], 'token' : [] }

        for idx, (token, label) in enumerate(zip(sent, labels)):
            sub_tokens = self.tokenize(token)
            if not sub_tokens:
                continue
            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            if len(token_ids) + len(sub_tokens) >= self.max_len:
                # don't add more token
                yield self.process_example_for_return(token_ids, all_label_ids)
                token_ids = token_ids[-self.shift :]
                all_label_ids = { k : [LABEL_PAD_ID] * len(token_ids) for k, v in all_label_ids.items() }

            for i, sub_token in enumerate(sub_tokens):
                token_ids.append(sub_token)
                raw_single_token = sub_tokens[0] if len(sub_tokens) == 1 else LABEL_PAD_ID
                mask_not_first = lambda label : label if i == 0 else LABEL_PAD_ID
                all_label_ids['pos'].append(mask_not_first(self.label2id[label]))
                all_label_ids['token'].append(mask_not_first(raw_single_token))
  
        yield self.process_example_for_return(token_ids, all_label_ids)
        
    def process_example(self, example: Dict) -> List[Dict]:
        sent: List = example["sent"]
        labels: List = example["labels"]

        data: List[Dict] = []
        if not sent:
            return data
        for src, tgt, token_labels in self._process_example_helper(sent, labels):
            data.append({
                "sent": src, "pos_labels": tgt, "lang": self.lang,
                "token_labels" : token_labels
            })
        # end changes
        return data


class UdPOS(TaggingDataset):
    @classmethod
    def get_labels(cls):
        return constant.UD_POS_LABELS

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        words: List[str] = []
        labels: List[str] = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                tok = line.strip().split("\t")
                if len(tok) < 2 or line[0] == "#":
                    assert len(words) == len(labels)
                    if words:
                        yield {"sent": words, "labels": labels}
                        words, labels = [], []
                if tok[0].isdigit():
                    word, label = tok[1], tok[3]
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield {"sent": words, "labels": labels}

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/UD_{lang}/*-ud-train.conllu"
        elif split == Split.dev:
            fp = f"{path}/UD_{lang}/*-ud-dev.conllu"
        elif split == Split.test:
            fp = f"{path}/UD_{lang}/*-ud-test.conllu"
        else:
            raise ValueError(f"Unsupported split: {split}")
        _fp = glob.glob(fp)
        
        if len(_fp) == 1:
            return _fp[0]
        elif len(_fp) == 0:
            return None
        else:
            raise ValueError(f"Unsupported split: {split}")
