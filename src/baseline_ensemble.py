
import torch
import os
import glob
import numpy as np
import pandas as pd

import predict.predict as predict
import predict.predict_utils as predict_utils
from model import Tagger
import util
from enumeration import Split
from analysis import sweeps
from dataset import LABEL_PAD_ID

def get_lang_path(base_path, sweep_name, subset, lang):
    get_subset_template = lambda base_path, sweep_name, subset : os.path.join(
        base_path, 'experiments', 'heatmaps', 'subset', sweep_name, f'subset_count={subset}',
        'version_*/ckpts_epoch*/val_predictions'
    )
    get_lang_template = lambda base_path, sweep_name, subset, lang : os.path.join(
        get_subset_template(base_path, sweep_name, subset), 
        lang, 'val_predictions.pt'
    )
    template = get_lang_template(base_path, sweep_name, subset, lang)
    path = glob.glob(template)
    assert len(path) == 1, path
    return path[0]
    
def clean_for_ensemble_softmax(is_masked, softmax, flat_padded_labels):
    flat_softmax = np.concatenate([
            sent_output.reshape(-1, sent_output.shape[-1]).cpu().numpy()
            for sent_output in softmax
        ], axis = 0)
    assert len(flat_softmax.shape) == 2, flat_softmax.shape
    assert len(flat_padded_labels.shape) == 1, flat_padded_labels.shape
    not_padding_indices = np.where(flat_padded_labels != LABEL_PAD_ID)[0]
    clean_softmax = np.take(flat_softmax, not_padding_indices, axis = 0)
    return torch.from_numpy(clean_softmax)
    
def compute_ensemble_df(lang, base_path, weights, subsets, args_1, args_2):
    sweep_name_1, sweep_name_2 = args_1['sweep_name'], args_2['sweep_name']
    padded_labels_1, padded_labels_2 = args_1['padded_labels_dict'][lang], args_2['padded_labels_dict'][lang]
    all_results = []
    for subset in subsets:
        path_1 = get_lang_path(base_path, args_1['sweep_name'], subset, lang)
        path_2 = get_lang_path(base_path, args_2['sweep_name'], subset, lang)
        # Below is probably temporary indexing into torch.load dict
        # -- remove when you regen all predictions
        softmaxes_1 = clean_for_ensemble_softmax(args_1['is_masked'], torch.load(path_1)[lang], padded_labels_1)
        softmaxes_2 = clean_for_ensemble_softmax(args_2['is_masked'], torch.load(path_2)[lang], padded_labels_2)
        if not softmaxes_1.shape == softmaxes_2.shape:
            import pdb; pdb.set_trace()
        for weight in weights:
            try:
                ensemble_softmax = weight * softmaxes_1  + (1 - weight) * softmaxes_2
            except: import pdb; pdb.set_trace()
            get_clean_labels = lambda padded_labels : padded_labels[padded_labels != LABEL_PAD_ID]
            clean_labels = get_clean_labels(padded_labels_1)
            if not torch.all(clean_labels == get_clean_labels(padded_labels_2)):
                import pdb; pdb.set_trace()
            # the fact that it internally cleans labels doesn't matter
            accuracy = predict.softmax_to_accuracy(ensemble_softmax, clean_labels)
            current_results = { 'subset' : subset , 'first_weight' : weight, 'accuracy' : accuracy * 100.0 }
            all_results.append(current_results)
    
    df = pd.DataFrame.from_records(all_results)
    return df

def compute_all_ensemble_grids(languages, base_path, weights, subsets, args_1, args_2, modifier):
    print(f'processing {modifier}')
    phase = Split.dev
    get_padded_labels_dict = lambda loading_model, languages, phase : {
        lang : predict_utils.get_batch_padded_flat_labels(loading_model, lang, phase)
        for lang in languages
    }
    labels_args = (languages, phase)
    args_1['padded_labels_dict'] = get_padded_labels_dict(args_1['loading_model'], *labels_args)
    args_2['padded_labels_dict'] = get_padded_labels_dict(args_2['loading_model'], *labels_args)
    for lang in languages:
        print(lang)
        analysis_path = os.path.join(base_path, 'heatmaps', modifier, phase, lang)
        df = compute_ensemble_df(lang, base_path, weights, subsets, args_1, args_2)
        sweeps.get_heatmap(df, 'first_weight', 'subset', 'accuracy', analysis_path)
        
if __name__ == '__main__':
    
    # Assumes that single_predict has been run for the relevant baselines
    
    base_path = '../../alt/semisupervised-baseline-pos'
    
    languages = ['English', 'Dutch', 'Turkish', 'Irish']
    subsets = [1, 2, 5, 10, 50, 100, 500, 1000, 1500]
    weights = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    
    args = (languages, base_path, weights, subsets)
    
    masked_loading_model = util.get_full_set_model(Tagger, True)
    unmasked_loading_model = util.get_full_set_model(Tagger, False)
    masked_args = {
        'sweep_name' : 'masked',
        'is_masked' : True,
        'loading_model' : masked_loading_model
    } 
    unmasked_args = {
        'sweep_name' : 'unmasked',
        'is_masked' : False,
        'loading_model' : unmasked_loading_model
    }
    unmasked_alt_seed_args = {
        'sweep_name' : 'unmasked_alt_seed',
        'is_masked' : False,
        'loading_model' : unmasked_loading_model
    }

    mixed_df = compute_all_ensemble_grids(*(args + (masked_args, unmasked_args, 'mask_unmasked')))
    pure_unmasked_df = compute_all_ensemble_grids(*(args + (unmasked_args, unmasked_alt_seed_args, 'pure_unmasked')))
    