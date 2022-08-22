from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from predict import match_filtering, predict_utils, softmaxes
import numpy as np
import transformers
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

import util
from model import Tagger

tqdm.monitor_interval = 0
tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")

from constant import LABEL_PAD_ID
DUMMY_LABEL = "DUMMY_LABEL"


class Tokenizer(transformers.PreTrainedTokenizer):
    pass


class Dataset(TorchDataset):
    def __init__(
        self, *, tokenizer,
        filepath, lang, masked, split,
        max_len, subset_ratio, subset_count, subset_seed,
        self_training_args = {}
    ):
        self.self_training_args = self_training_args
        self.loading_models = None
        if self.self_training_args:
            self.loading_models = {
                is_masked : util.get_subset_model(Tagger, is_masked)
                for is_masked in [True, False]
            }
        self.masked = masked
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
        for ex in tqdm(
            self.read_file(self.filepath, self.lang, self.split), desc="read data"
        ):
            examples.append(ex)
        return examples
    
    def get_subset_indices(self, full_examples):
        example_indices = list(range(len(examples)))
        if self.subset_count > 0 or self.subset_ratio < 1:
            if self.subset_count > 0:
                subset_size = self.subset_count
            elif self.subset_ratio < 1:
                subset_size = int(len(examples) * self.subset_ratio)
            else:
                raise ValueError("subset_ratio and subset_count is mutally exclusive")

            print(
                f"taking {subset_size} subset (total {len(examples)}) from {self.filepath}"
            )

            seed = np.random.RandomState(self.subset_seed)
            shuffled_example_indices = seed.permutation(example_indices)
            if not self.hparams.use_subset_complement:
                example_indices = shuffled_example_indices[:subset_size]
            else:
                example_indices = shuffled_example_indices[subset_size:]
        return example_indices
        
    def process_examples(self, examples):
        data = []
        for example in tqdm(examples, desc="parse data"):
            data.extend(self.process_example(example))
        return data
        
    def prep_view_components(self):
        view_components = {}
        for view_name in ['1', '2']:
            components = {
                'masked' : self.self_training_args[f'is_masked_view_{view_name}'],
                'checkpoint_path' : self.self_training_args[f'view_checkpoint_{view_name}']
            }
            components['softmax_folder'] = predict_utils.get_predictions_path(components['checkpoint_path'], self.phase)
            softmax_path = os.path.join(components['softmax_folder'], f'{phase}_predictions.pt')
            if os.path.exists(softmax_path):
                softmax = torch.load(softmax_path)[self.lang]
            else:
                model = Tagger.load_from_checkpoint(components['checkpoint_path'])
                dataloader = util.get_subset_dataloader(self.loading_model[components['masked']], self.lang, self.split)
                softmax = softmaxes.get_softmaxes(model, dataloader, softmax_path, self.phase)
            components['predictions'] = match_filtering.get_clean_matching_predictions(
                    softmax, self.loading_model[components['masked']]
                )
            components['labels'] = predict_utils.get_batch_padded_flat_labels(
                    self.loading_model[components['masked']], self.lang, self.phase
                )
            view_components[view_name] = components
        return view_components
        
    def replace_labels_with_ensemble_match(
            self, processed_full_examples, subset_indices, clean_mask, predictions
        ):
            
        # Initial sanity checks
        # The number of predictable positions total matches the number of dense predictions
        get_non_pad_count = lambda examples : sum([
                np.sum(example['labels'] != LABEL_PAD_ID)
                for example in examples
            ])
        assert len(predictions.shape) == 1, predictions.shape
        valid_positions_count = get_non_pad_count(processed_full_examples)
        if not valid_positions_count == predictions.shape[0]:
            import pdb; pdb.set_trace()

        matches_skipped = 0
        updated_examples = []
        for sentence_index, example in enumerate(processed_full_examples):
            clean_ensemble_index = 0
            # Consider if not part of subset
            if sentence_index not in subset_indices:
                # Need to account for skipping all legitimate positions
                valid_prediction_positions = np.sum(example['labels'] != LABEL_PAD_ID)
                current_matches_skipped = clean_mask[ensemble_index:ensemble_index + valid_prediction_positions]
                matches_skipped += current_matches_skipped
                clean_ensemble_index += valid_prediction_positions
                continue
            # Process single example
            new_labels = []
            raw_labels = example['labels']
            for raw_label in raw_labels:
                # Padding in the original
                if raw_label == LABEL_PAD_ID:
                    new_labels.append(LABEL_PAD_ID)
                    continue
                # The present position is legitimate but not a match
                if not clean_mask[clean_ensemble_index]:
                    new_labels.append(LABEL_PAD_ID)
                    clean_ensemble_index += 1
                    continue
                # Otherwise, it's a legitimate prediction position
                # and a match was found.
                assert raw_label != LABEL_PAD_ID and clean_mask[clean_ensemble_index],\
                    f"raw_label: {raw_label}, clean_mask: {clean_mask[clean_ensemble_index]}"
                new_labels.append(predictions[clean_ensemble_index])
                clean_ensemble_index += 1
            updated_labels = np.array(new_labels)
            
            # Individual example sanity checks
            assert updated_labels.shape == raw_labels.shape,\
                f"{updated_labels.shape}, {raw_labels.shape}"
            # All padded positions in the original correspond to padding in the present labels
            assert len(updated_labels.shape) == 1, updated_labels.shape
            get_all_invalid_positions = lambda labels : set(np.where(labels == LABEL_PAD_ID)[0])
            if not get_all_invalid_positions(updated_labels).issubset(get_all_invalid_positions(raw_labels)):
                import pdb; pdb.set_trace()
            updated_examples.append( { 'sent' : example['sent'], 'labels' : updated_labels } )
        
        # Overall sanity checks
        # The number of subsetted examples is the same as the number in the index
        assert len(updated_examples) == len(subset_indices)
        # The number of final predictable positions is the same as the number of matched positions
        subset_matched_positions = clean_mask.sum() - matches_skipped
        final_valid_positions = get_non_pad_count(updated_examples)
        if not final_valid_positions == subset_matched_positions:
            import pdb; pdb.set_trace()
        return updated_examples
        
    def load(self):
        assert self.data == []

        raw_full_examples = self.load_all_predictions()
        subset_indices = self.get_subset_indices(raw_full_examples)
        
        if not self.is_self_training:
            subset_examples = [
                    example
                    for index, example in raw_full_examples
                    if index in set(subset_indices)
                ]
            data = self.process_examples(subset_examples)
        else:
            # Need to align the view labels and then overwrite the original labels
            # Such that Wu/Dredze structure is returned instead.
            full_data = self.process_examples(raw_full_examples)
            view_components = self.prep_view_components()
            predictions_1 = view_components['1']['predictions']
            predictions_2 = view_components['2']['predictions']
            clean_mask = (predictions_1 == predictions_2)
            
            data = self.replace_labels_with_ensemble_match(full_data, subset_indices, clean_mask, predictions_1)
        self.data = data

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
