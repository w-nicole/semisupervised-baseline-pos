
# Adapted from Shijie Wu's crosslingual-nlp repository.
# For LICENSE, see LICENSE.txt.

import glob
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import random

import constant
from dataset.base import DUMMY_LABEL, Dataset
from constant import LABEL_PAD_ID
from enumeration import Split

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

    def add_special_tokens(self, sent, labels):
        sent = self.tokenizer.build_inputs_with_special_tokens(sent)
        labels = self.tokenizer.build_inputs_with_special_tokens(labels)
        mask = self.tokenizer.get_special_tokens_mask(
            sent, already_has_special_tokens=True
        )
        sent, labels, mask = np.array(sent), np.array(labels), np.array(mask)
        label = labels * (1 - mask) + LABEL_PAD_ID * mask
        return sent, label
        
    def get_possibly_masked_sentence(self, token_ids, index):
        prediction_position_token = self.tokenizer.mask_token_id\
                if random.random() < self.mask_probability else token_ids[index]
        # concat will copy, so this is safe
        current_masked_token_ids = np.concatenate([
            token_ids[:index].copy(),
            np.array([prediction_position_token]),
            token_ids[index+1:].copy()]
        )
        return current_masked_token_ids
    
    def yield_postprocessed_outputs(self, raw_token_ids, raw_label_ids):
        token_ids, label_ids = self.add_special_tokens(raw_token_ids, raw_label_ids)
        if not self.unraveled_predictions:
            return [(token_ids, label_ids)]
        assert token_ids.shape[0] == label_ids.shape[0], f"{token_ids.shape}, {label_ids.shape}"
        masked_indices = defaultdict(list)
        masked_labels = []
        for index, label in enumerate(label_ids):
            if label == LABEL_PAD_ID: continue
            # below is true because of guarantee of [SEP], which has -1 as its label id
            assert index != label_ids.shape[0] - 1, label_ids.shape[0]
            
            masked_token_ids_1 = self.get_possibly_masked_sentence(token_ids, index)
            
            current_labels = np.full(label_ids.shape, LABEL_PAD_ID).astype('int')
            current_labels[index] = label_ids[index]
            masked_indices['1'].append(masked_token_ids_1)
            if self.double_pass:
                masked_token_ids_2 = self.get_possibly_masked_sentence(token_ids, index)
                masked_indices['2'].append(masked_token_ids_2)
            masked_labels.append(current_labels)
        if not self.double_pass:
            yield_format = [(sent, labels) for sent, labels in zip(masked_indices['1'], masked_labels)]
        else:
            yield_format = [
                (sent_1, sent_2, labels)
                for sent_1, sent_2, labels in zip(
                        masked_indices['1'], masked_indices['2'], masked_labels
                    )
                ]
        return yield_format

    def _process_example_helper(
        self, sent: List, labels: List
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

        token_ids: List[int] = []
        label_ids: List[int] = []

        for token, label in zip(sent, labels):
            sub_tokens = self.tokenize(token)
            if not sub_tokens:
                continue
            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            if len(token_ids) + len(sub_tokens) >= self.max_len:
                # don't add more token
                for output in self.yield_postprocessed_outputs(token_ids, label_ids):
                    yield output

                token_ids = token_ids[-self.shift :]
                label_ids = [LABEL_PAD_ID] * len(token_ids)

            for i, sub_token in enumerate(sub_tokens):
                token_ids.append(sub_token)
                label_id = self.label2id[label] if i == 0 else LABEL_PAD_ID
                label_ids.append(label_id)

        for output in self.yield_postprocessed_outputs(token_ids, label_ids):
            yield output

    def process_example(self, example: Dict) -> List[Dict]:
        sent: List = example["sent"]
        labels: List = example["labels"]
        data: List[Dict] = []
        
        if not sent:
            return data
        if not self.double_pass:
            output_keys = ['sent', 'pos_labels']
        else:
            output_keys = ['sent_1', 'sent_2', 'pos_labels']
        for raw_example in self._process_example_helper(sent, labels):
            example = { key : output for key, output in zip(output_keys, raw_example)}
            example['lang'] = self.lang
            data.append(example)
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
