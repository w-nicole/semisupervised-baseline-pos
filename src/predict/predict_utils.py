
import torch
import os

from enumeration import Split
import util
import metric
from predict import predict_utils

def get_over_languages_args(model_class, phase, languages, subset_count = -1):
    loading_models = {
        is_masked : util.get_subset_model(model_class, is_masked, subset_count)
        for is_masked in [True, False]
    }
    dataloaders_dict, padded_labels_dict = dict(), dict()
    for is_masked, loading_model in loading_models.items():
        dataloaders_dict[is_masked] = {
            lang : util.get_subset_dataloader(loading_models[is_masked], lang, phase)
            for lang in languages
        } 
        padded_labels_dict[is_masked] = {
            lang : predict_utils.get_batch_padded_flat_labels(loading_models[is_masked], lang, phase)
            for lang in languages
        }
    return dataloaders_dict, padded_labels_dict

def get_phase_predictions_path(checkpoint_path, phase):
    return os.path.join(get_analysis_path(checkpoint_path), f'{phase}_predictions')

def clean_padded_labels_and_predictions(padded_labels, padded_predictions):
    padded_labels = util.remove_from_gpu(padded_labels)
    mask_for_non_pad = (padded_labels != metric.LABEL_PAD_ID)
    labels = padded_labels[mask_for_non_pad]
    outputs = padded_predictions[mask_for_non_pad]
    return outputs.cpu(), labels.cpu()
    
def get_batch_padded_flat_labels(loading_model, lang, split):
    dataloader = loading_model.get_unshuffled_dataloader(lang, split)
    all_labels_dataloader = list(map(lambda example : example['pos_labels'], dataloader))
    return torch.cat(list(map(lambda tensor : tensor.flatten(), all_labels_dataloader)))

def get_analysis_path(checkpoint_path):
    path_components = checkpoint_path.split('/')
    decoder_folder = util.get_folder_from_checkpoint_path(checkpoint_path)
    checkpoint_name = path_components[-1]
    analysis_path = os.path.join(decoder_folder, os.path.splitext(checkpoint_name)[0])
    print(f'Analysis path: {analysis_path}')
    return analysis_path