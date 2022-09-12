
from predict import match_filtering
from constant import LABEL_PAD_ID
from enumeration import Split
from model import Tagger
import predict.predict_utils as predict_utils
import util

from collections import defaultdict
import torch
import glob
import pandas as pd
import os

def get_ensembled_subset_metrics(args_1, args_2):
    
    clean_predictions_1 = match_filtering.get_clean_matching_predictions(args_1['softmax'], args_1['flat_labels'])
    clean_predictions_2 = match_filtering.get_clean_matching_predictions(args_2['softmax'], args_2['flat_labels'])
    assert clean_predictions_1.shape == clean_predictions_2.shape, f'{predictions_1.shape} {predictions_2.shape}'
    clean_mask = (clean_predictions_1 == clean_predictions_2)
    
    raw_flat_labels = args_1['flat_labels']
    clean_flat_labels = raw_flat_labels[raw_flat_labels != LABEL_PAD_ID]
    number_of_clean_labels = clean_flat_labels.shape[0]
    number_of_matched_labels = clean_mask.sum().item()
    if not number_of_clean_labels == (args_2['flat_labels'] != LABEL_PAD_ID).sum().item():
        import pdb; pdb.set_trace()
        
    percent_ensemble_match = (clean_mask.sum().item() / number_of_clean_labels) * 100.0
    get_masked_predictions = lambda mask, predictions : torch.where(mask, predictions, torch.full(mask.shape, LABEL_PAD_ID))
    match_masked_predictions = get_masked_predictions(clean_mask, clean_predictions_1)
    assert torch.all(match_masked_predictions == get_masked_predictions(clean_mask, clean_predictions_2))
    
    # Because clean_flat_labels has no -1, so non-matched positions will never interfere
    matched_accuracy = (match_masked_predictions == clean_flat_labels).sum().item() / number_of_matched_labels * 100.0
    return {
        'matched_labels_count' : number_of_matched_labels,
        'percent_ensemble_match' : percent_ensemble_match,
        'ensemble_accuracy' : matched_accuracy
    }
    
def get_single_to_ensembled_metrics(unmasked_args, matched_labels_count):
    
    raw_flat_labels = unmasked_args['flat_labels']
    clean_softmax = match_filtering.clean_for_ensemble_softmax(unmasked_args['softmax'], unmasked_args['flat_labels'])
    clean_flat_labels = raw_flat_labels[raw_flat_labels != LABEL_PAD_ID]
    assert clean_flat_labels.shape[0] == clean_softmax.shape[0],\
        f'labels: {clean_flat_labels.shape}, softmax: {clean_softmax.shape}'
        
    best_unmasked_indices = torch.argsort(clean_softmax.max(dim=-1)[0], descending = True)[:matched_labels_count]
    predictions = clean_softmax.argmax(dim=-1)
    assert len(predictions.shape) == 1, predictions.shape
    assert predictions.shape == clean_flat_labels.shape,\
        f"{predictions.shape}, {clean_flat_labels.shape}"
            
    best_predictions = torch.take(predictions, best_unmasked_indices)
    best_labels = torch.take(clean_flat_labels, best_unmasked_indices)
    assert best_labels.shape[0] == best_predictions.shape[0] == matched_labels_count,\
        f"{best_labels.shape}, {best_predictions.shape}, {matched_labels_count}"
        
    accuracy = (best_predictions == best_labels).sum().item() / best_labels.shape[0] * 100.0
    return accuracy

def get_path_match(sweep_name, sweep_folder, template):
    full_template = os.path.join(sweep_folder, sweep_name, template)
    matches = glob.glob(full_template)
    if len(matches) != 1: import pdb; pdb.set_trace()
    return matches[0]
        

if __name__ == "__main__":
    lang = 'Dutch'
    subset_counts = [1, 2, 5, 10, 50, 100, 500, 1000, 1500]
    
    sweep_folder = '../../alt/semisupervised-baseline-pos/experiments/subset'
    sweep_names = ['masked', 'unmasked', 'unmasked_alt_seed']
    loading_models = { is_masked : util.get_full_set_model(Tagger, is_masked) for is_masked in [True, False] }
    
    non_ensemble_accuracies = {}
    for count in subset_counts:
        # Below may switch to dev_predictions etc. on future runs
        template = f'subset_count={count}/version_*/ckpts_epoch=*/val_predictions/{lang}/val_predictions.pt'
        # Remove Below "lang" key indexing for future prediction generations 
        args = defaultdict(dict)
        for sweep_name, is_masked in zip(sweep_names, [True, False, False]):
            softmaxes_path = get_path_match(sweep_name, sweep_folder, template)
            args[sweep_name]['softmax'] = torch.load(softmaxes_path)[lang]
            args[sweep_name]['flat_labels'] = predict_utils.get_batch_padded_flat_labels(loading_models[is_masked], lang, Split.dev)
        
        mixed_metrics = get_ensembled_subset_metrics(args['masked'], args['unmasked'])
        pure_metrics = get_ensembled_subset_metrics(args['unmasked'], args['unmasked_alt_seed'])
        
        ensemble_metrics = pd.DataFrame.from_records({
            sweep_name : sweep_metrics
            for sweep_name, sweep_metrics in zip(['mixed', 'pure'], [mixed_metrics, pure_metrics])
        })
        analysis_folder = os.path.join(sweep_folder, 'ensemble', lang, f'subset_count={count}')
        if not os.path.exists(analysis_folder): os.makedirs(analysis_folder)
        ensemble_df_path = os.path.join(analysis_folder, 'ensemble_subset_accuracies.csv')
        ensemble_metrics.to_csv(ensemble_df_path)
        
        non_ensemble_accuracies[count] = get_single_to_ensembled_metrics(args['unmasked'], mixed_metrics['matched_labels_count'])
    
    non_ensemble_df = pd.DataFrame.from_records([non_ensemble_accuracies])
    non_ensemble_folder = os.path.join(sweep_folder, lang)
    if not os.path.exists(non_ensemble_folder): os.makedirs(non_ensemble_folder)
    non_ensemble_df.to_csv(os.path.join(non_ensemble_folder, 'non_ensemble_accuracies.csv'))
        