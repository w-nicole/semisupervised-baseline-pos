
import pandas as pd
import os
import torch
import numpy as np

from model import Tagger
from dataset import LABEL_PAD_ID

import baseline_ensemble
import constant
import util
import predict.predict_utils as predict_utils

# TODO: Make this not dependent on checkpoint path later if needed to sweep this over all.
def get_by_class_accuracies(base_path, sweep_name, subset, lang, loading_model, checkpoint_path):
    phase = 'dev'
    flat_labels_by_dataset = util.get_full_set_labels(loading_model, lang, phase).cpu()
    flat_labels_by_dataloader = predict_utils.get_batch_padded_flat_labels(loading_model, lang, phase).cpu()
    
    predictions_path = baseline_ensemble.get_lang_path(base_path, sweep_name, subset, lang)
    softmaxes = torch.load(predictions_path)[lang]
    flat_raw_predictions = torch.cat([
            torch.argmax(softmax.reshape(-1, softmax.shape[-1]), dim = -1)
            for softmax in softmaxes
        ], dim = 0).cpu()
        
    clean_by_dataloader_mask = (flat_labels_by_dataloader != LABEL_PAD_ID)
    clean_by_dataset_mask = (flat_labels_by_dataset != LABEL_PAD_ID)
    clean_predictions = flat_raw_predictions[clean_by_dataloader_mask]
    clean_labels = flat_labels_by_dataset[clean_by_dataset_mask]
    
    clean_matches = (clean_predictions == clean_labels)    
    get_accuracy = lambda matches, mask :  (( matches.sum() / mask.sum() ) * 100.0).item() if mask.sum() > 0 else -1
    class_accuracies = {}
    for class_index in range(len(constant.UD_POS_LABELS)):
        class_label_mask = clean_labels == class_index
        class_matches = clean_matches & (class_label_mask)
        class_accuracies[constant.UD_POS_LABELS[class_index]] = get_accuracy(class_matches, class_label_mask)
    class_accuracies['all'] = ((clean_predictions == clean_labels).sum().item() / clean_labels.shape[0]) * 100.0
    df = pd.DataFrame.from_records([class_accuracies])
    df_path = os.path.join(predict_utils.get_analysis_path(checkpoint_path), 'class_accuracies.csv')
    df.to_csv(df_path)
    print(f'Written to: {df_path}')
    return df
    
if __name__ == '__main__':
    masked_loading_model = util.get_full_set_model(Tagger, True)
    unmasked_loading_model = util.get_full_set_model(Tagger, False)
    
    subset = 1000
    lang = 'English'
    
    base_path = '../../alt/semisupervised-baseline-pos/'
    sweep_path = os.path.join(base_path, 'experiments/subset')
    masked_checkpoint_path = os.path.join(sweep_path, 'masked/subset_count=1000/version_l6q0hkqs/ckpts/ckpts_epoch=6-val_English_pos_acc_epoch=76.730.ckpt')
    unmasked_checkpoint_path = os.path.join(sweep_path, 'unmasked/subset_count=1000/version_ftgxwvip/ckpts/ckpts_epoch=7-val_English_pos_acc_epoch=95.367.ckpt')
    
    masked_df = get_by_class_accuracies(base_path, 'masked', subset, lang, masked_loading_model, masked_checkpoint_path)
    unmasked_df = get_by_class_accuracies(base_path, 'unmasked', subset, lang, unmasked_loading_model, unmasked_checkpoint_path)
    
    comparison_path = os.path.join(sweep_path, 'masked_unmasked/subset_count=1000')
    (unmasked_df - masked_df).to_csv(os.path.join(comparison_path, 'class_accuracies_difference.csv'))
    
    import pdb; pdb.set_trace()
    
    