from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from predict import match_filtering, predict_utils, softmaxes
import numpy as np
import transformers
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

import util

tqdm.monitor_interval = 0
tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")

from constant import LABEL_PAD_ID
DUMMY_LABEL = "DUMMY_LABEL"


class Tokenizer(transformers.PreTrainedTokenizer):
    pass


class Dataset(TorchDataset):
    def __init__(
        self, use_subset_complement, tokenizer,
        filepath, lang, self_training_args, prediction_format_args,
        split = None,
        max_len = None, subset_ratio = 1, subset_count = -1, subset_seed = 42,
    ):
        self.use_subset_complement = use_subset_complement
        self.self_training_args = self_training_args
        self.unraveled_predictions = prediction_format_args['unraveled_predictions']
        self.mask_probability = prediction_format_args['mask_probability']
        self.double_pass = prediction_format_args['double_pass']
        self.tokenizer = tokenizer
        self.filepath = filepath
        self.lang = self.unpack_language(lang)
        self.split = split
        if max_len is not None:
            assert 0 < max_len <= self.tokenizer.max_len_single_sentence
        self.max_len = (
            max_len if max_len is not None else self.tokenizer.max_len_single_sentence
        )
        self.data: List[Dict[str, np.ndarray]] = []

        assert 0 < subset_ratio <= 1
        assert not (
            subset_ratio < 1 and subset_count > 0
        ), "subset_ratio and subset_count is mutally exclusive"
        self.subset_ratio = subset_ratio
        self.subset_count = subset_count
        self.subset_seed = subset_seed

        self.before_load()
        self.load()

    def unpack_language(self, lang):
        return lang

    def tokenize(self, token):
        if isinstance(self.tokenizer, transformers.XLMTokenizer):
            sub_words = self.tokenizer.tokenize(token, lang=self.lang)
        else:
            sub_words = self.tokenizer.tokenize(token)
        if isinstance(self.tokenizer, transformers.XLMRobertaTokenizer):
            if not sub_words:
                return []
            if sub_words[0] == "▁":
                sub_words = sub_words[1:]
        return sub_words

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def before_load(self):
        pass

    def load_all_predictions(self):
        examples = []
        for index, ex in enumerate(tqdm(
            self.read_file(self.filepath, self.lang, self.split), desc="read data"
        )):
            ex['ud_entry_index'] = index
            examples.append(ex)
                
        return examples
    
    def get_subset_indices(self, raw_full_examples):
        get_indices = lambda examples : [ example['ud_entry_index'] for example in examples ]
        if self.subset_count > 0 or self.subset_ratio < 1:
            if self.subset_count > 0:
                subset_size = self.subset_count
            elif self.subset_ratio < 1:
                subset_size = int(len(raw_full_examples) * self.subset_ratio)
            else:
                raise ValueError("subset_ratio and subset_count is mutally exclusive")

            print(
                f"calculating {subset_size} subset (total {len(raw_full_examples)}) from {self.filepath}"
            )

            seed = np.random.RandomState(self.subset_seed)
            shuffled_examples = seed.permutation(raw_full_examples)
            if not self.use_subset_complement:
                subset_examples = shuffled_examples[:subset_size]
            else:
                subset_examples = shuffled_examples[subset_size:]
            return get_indices(subset_examples)
        else:
            return get_indices(raw_full_examples)
        
    def process_examples(self, examples):
        data = []
        for example in tqdm(examples, desc="parse data"):
            data.extend(self.process_example(example))
        return data
        
    def load(self):
        raise NotImplementedError

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        raise NotImplementedError

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        raise NotImplementedError

    def process_example(self, example: Dict) -> List[Dict]:
        raise NotImplementedError

    @classmethod
    def write_example(cls, example: Dict, file_handler):
        raise NotImplementedError

    @classmethod
    def project_label(
        cls, example: Dict, translation: List[str], mapping: List[Tuple]
    ) -> Dict:
        raise NotImplementedError
