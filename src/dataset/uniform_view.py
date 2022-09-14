
import torch
import os
import util
import numpy as np

from dataset.tagging import UdPOS
from dataset.base import Dataset
from predict import predict_utils, softmaxes, match_filtering
from constant import LABEL_PAD_ID
from model.single import Single

class UniformViewDataset(UdPOS):
    
    def __init__(
        self, use_subset_complement, tokenizer,
        filepath, lang, self_training_args, prediction_format_args,
        split = None,
        max_len = None, subset_ratio = 1, subset_count = -1, subset_seed = 42,
    ):
        self.loading_models = {
                is_masked : util.get_subset_model(Single, is_masked)
                for is_masked in [True, False]
            }
        super(UniformViewDataset, self).__init__(
                use_subset_complement, tokenizer,
                filepath, lang, masked, split,
                max_len, subset_ratio, subset_count, subset_seed,
                self_training_args
            )
    
    def prep_view_components(self):
        view_components = {}
        for view_name in ['1', '2']:
            components = {
                    'masked' : self.self_training_args[f'is_masked_view_{view_name}'],
                    'checkpoint_path' : self.self_training_args[f'view_checkpoint_{view_name}']
                }
            components['softmax_folder'] = predict_utils.get_phase_predictions_path(
                    components['checkpoint_path'], os.path.join(
                            f"masked={components['masked']}", self.lang
                        )
                )
            softmax_path = os.path.join(components['softmax_folder'], f'{self.split}_predictions.pt')
            if os.path.exists(softmax_path):
                softmax = torch.load(softmax_path)
            else:
                model = Single.load_from_checkpoint(components['checkpoint_path'])
                dataloader = util.get_subset_dataloader(self.loading_models[components['masked']], self.lang, self.split)
                softmax = softmaxes.get_softmaxes(model, dataloader, components['softmax_folder'], self.split)
            loading_model = self.loading_models[components['masked']]
            components['labels'] = predict_utils.get_batch_padded_flat_labels(
                    self.loading_models[components['masked']], self.lang, self.split
                )
            components['predictions'] = match_filtering.get_clean_matching_predictions(
                    softmax, components['labels']
                )
            view_components[view_name] = components
        return view_components
        
    def replace_labels_with_ensemble_match(
            self, processed_full_examples, subset_indices, clean_mask, predictions
        ):
            
        # Initial sanity checks
        # The number of predictable positions total matches the number of dense predictions
        get_non_pad_count = lambda examples : sum([
                np.sum(example['pos_labels'] != LABEL_PAD_ID)
                for example in examples
            ])
        assert len(predictions.shape) == 1, predictions.shape
        valid_positions_count = get_non_pad_count(processed_full_examples)
        if not valid_positions_count == predictions.shape[0]:
            import pdb; pdb.set_trace()

        matches_skipped = 0
        updated_examples = []
        clean_ensemble_index = 0
        for sentence_index, example in enumerate(processed_full_examples):
            # Consider if not part of subset
            if sentence_index not in subset_indices:
                # Need to account for skipping all legitimate positions
                valid_prediction_positions = np.sum(example['pos_labels'] != LABEL_PAD_ID)
                current_matches_skipped = clean_mask[clean_ensemble_index:clean_ensemble_index + valid_prediction_positions].sum()
                matches_skipped += current_matches_skipped.item()
                clean_ensemble_index += valid_prediction_positions
                continue
            # Process single example
            new_labels = []
            raw_labels = example['pos_labels']
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
            # i.e. the updated_labels may only have >= positions invalid than the raw.
            if not get_all_invalid_positions(raw_labels).issubset(get_all_invalid_positions(updated_labels)):
                import pdb; pdb.set_trace()
            updated_examples.append( { 'sent' : example['sent'], 'pos_labels' : updated_labels, 'lang' : example['lang'] } )
        
        # Overall sanity checks
        # The number of subsetted examples is the same as the number in the index
        assert len(updated_examples) == len(subset_indices)
        # The number of final predictable positions is the same as the number of matched positions
        subset_matched_positions = (clean_mask.sum() - matches_skipped).item()
        final_valid_positions = get_non_pad_count(updated_examples)
        if not final_valid_positions == subset_matched_positions:
            import pdb; pdb.set_trace()
        return updated_examples
    
    def load(self):
        assert self.data == []
        raw_full_examples = self.load_all_predictions()
        subset_indices = self.get_subset_indices(raw_full_examples)
        # Need to align the view labels and then overwrite the original labels
        # Such that Wu/Dredze structure is returned instead.
        full_data = self.process_examples(raw_full_examples)
        view_components = self.prep_view_components()
        predictions_1 = view_components['1']['predictions']
        predictions_2 = view_components['2']['predictions']
        clean_mask = (predictions_1 == predictions_2)
            
        data = self.replace_labels_with_ensemble_match(full_data, subset_indices, clean_mask, predictions_1)
        self.data = data
        
    