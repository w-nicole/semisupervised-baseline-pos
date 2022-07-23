
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed truncation to be simply off the end of the example,
# Removed sliding window logic.
# Changed labels to not use first subtoken marking via padding, but just to be labels.
# Changed to return start/end indices.
# Updated imports
# Added lengths as dataloader output
# Removed irrelevant code

import glob
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import constant
from dataset.base import DUMMY_LABEL, Dataset
from enumeration import Split
from metric import LABEL_PAD_ID

import torch

class TaggingDataset(Dataset):
    def before_load(self):
        self.max_len = min(self.max_len, self.tokenizer.max_len_single_sentence)
        # Removed self.shift
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

    # Changed this entire section:
    # to truncate at the max non-CLS/SEP tokens dictated by the tokenizer,
    # to not use any sliding window,
    # to add labels per token directly, rather than using subtokens,
    # to create start/end indices.
    # added lengths.
    # added token_labels
    def _process_example_helper(
        self, sent: List, labels: List
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        
        # start_index, end_index include CLS/SEP (i.e. the first subtoken is index 1)
        token_ids: List[int] = []
        pos_label_ids: List[int] = []
        start_indices: List[int] = []
        end_indices: List[int] = []
        token_labels = []
            
        current_index = 1

        for token, label in zip(sent, labels):
            sub_tokens = self.tokenize(token)
            if not sub_tokens:
                continue
            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            if len(token_ids) + len(sub_tokens) > self.max_len:
                # don't add more token
                break

            pos_label_ids.append(self.label2id[label])
            token_labels.append(sub_tokens[0] if len(sub_tokens) == 1 else LABEL_PAD_ID)
            
            token_ids.extend(sub_tokens)
            start_indices.append(current_index)
            end_indices.append(current_index + len(sub_tokens))
            
            current_index += len(sub_tokens)

        token_ids = self.add_special_tokens(token_ids)
        pos_label_ids = np.array(pos_label_ids)

        # averaging will average all unwanted representations (padding, CLS, SEP) to index constant.START_END_INDEX_PADDING.
        
        pad_indices = lambda indices : np.array(
            [constant.START_END_INDEX_PADDING]
            + indices
            + [constant.START_END_INDEX_PADDING]
        )
        
        start_indices = pad_indices(start_indices)
        end_indices = pad_indices(end_indices)
        
        assert len(pos_label_ids.shape) == 1, pos_label_ids.shape
        yield (token_ids, pos_label_ids, start_indices, end_indices, pos_label_ids.shape[0], token_labels)
        
        # end changes
        
    def process_example(self, example: Dict) -> List[Dict]:
        sent: List = example["sent"]
        labels: List = example["labels"]

        data: List[Dict] = []
        if not sent:
            return data
        # Changed below to accomodate averaging_indices, lengths, token_labels
        for src, tgt, start_indices, end_indices, length, token_labels in self._process_example_helper(sent, labels):
            data.append({
                "sent": src, "pos_labels": tgt, "lang": self.lang,
                "start_indices" : start_indices, "end_indices" : end_indices,
                "length" : length, "token_labels" : token_labels
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
