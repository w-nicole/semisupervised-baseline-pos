
from explore import tflogs2pandas
from model import VAE, Tagger
from enumeration import Split
import constant

import pytorch_lightning as pl
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import glob
import torch
import torch.nn.functional as F

import util
import metric

from pprint import pprint

def get_all_predictions(model, langs):
    predictions = {}
    for lang in langs:
        if lang == "English" and isinstance(model, VAE):
            print('Skipping English because it is a labeled case.')
            continue
        trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0)
        model.reset_metrics()
        dataloader = model.get_dataloader(lang, Split.dev)
        predictions[lang] = trainer.predict(model, dataloaders = [dataloader], return_predictions = True)
    return predictions
    
def get_analysis_path(checkpoint_path):
    path_components = checkpoint_path.split('/')
    decoder_folder = util.get_folder_from_checkpoint_path(checkpoint_path)
    checkpoint_name = path_components[-1]
    analysis_path = os.path.join(decoder_folder, os.path.splitext(checkpoint_name)[0])
    print(f'Analysis path: {analysis_path}')
    return analysis_path
    
def get_padded_labels(model, lang):
    dataloader = model.get_dataloader(lang, Split.dev)
    raw_labels = []
    for batch in dataloader:
        raw_labels.append(batch['labels'].flatten())
    labels = torch.cat([batch['labels'].flatten() for batch in dataloader])
    return labels

def clean_padded_labels_pair(model, lang, padded_predictions, number_of_dims):
    padded_labels = util.remove_from_gpu(get_padded_labels(model, lang))
    mask_for_non_pad = (padded_labels != metric.LABEL_PAD_ID)
    labels = padded_labels[mask_for_non_pad]
    if number_of_dims == 1:
        outputs = padded_predictions[mask_for_non_pad]
    elif number_of_dims == 2:
        outputs = padded_predictions[mask_for_non_pad, :]
    elif number_of_dims == 3:
        outputs = padded_predictions[mask_for_non_pad, :, :]
    else:
        assert False, "This number of dimensions is not supported."
    return outputs, labels
    
    
# def compare_validation_predictions(model, checkpoint_path):
    
#     analysis_path = get_analysis_path(checkpoint_path)
#     predictions = torch.load(os.path.join(analysis_path, 'predictions.pt'))
    
#     try:
#         for lang, raw_logits in predictions.items():
#             to_flat_label = lambda probs : torch.argmax(probs.reshape(-1, probs.shape[-1]), dim = -1)
#             raw_outputs = torch.argmax(raw_logits, dim = -1)
#             flat_logits = to_flat_label(raw_outputs)
#             outputs, labels = clean_padded_labels_pair(model, lang, flat_logits)
    
#             plt.hist(outputs.numpy().flat, alpha = 0.5, color = 'r', label = 'predicted')
#             plt.hist(labels.numpy().flat, alpha = 0.5, color = 'g', label = 'true')
            
#             accuracy = (torch.sum(outputs == labels) / outputs.shape[0]).item()
#             plt.legend()
#             plt.title(f'Frequency of class predictions vs labels. Accuracy: {round(accuracy * 100, 4)}%')
#             plt.xlabel('Numerical label')
#             plt.ylabel('Counts')
#             plt.xticks(range(len(constant.UD_POS_LABELS)), constant.UD_POS_LABELS, rotation = 45)
#             figure_path = os.path.join(analysis_path, f'prediction_comparison_val_{lang}.png')
#             plt.savefig(figure_path)
#             print(f'Figure written to: {figure_path}')
#             plt.figure()
#     except: import pdb; pdb.set_trace()

def get_log_q_given_input(predictions_dict, lang):
    log_q_given_input = torch.cat([
        outputs[1].sum(dim=1)
        for outputs in predictions_dict[lang]
    ], axis = 0)
    return log_q_given_input
    
def compare_english_prior(model, checkpoint_path):
    
    analysis_folder = get_analysis_path(checkpoint_path)
    prediction_dict = torch.load(os.path.join(analysis_folder, 'predictions.pt'))
    try:
        lang = 'Dutch'
        english_prior = model.get_smoothed_english_prior()
        raw_outputs = get_log_q_given_input(prediction_dict, lang)
        predictions, _ = clean_padded_labels_pair(model, lang, raw_outputs)
        repeated_prior = english_prior.unsqueeze(0).repeat(predictions.shape[0], 1)
        
        mean = torch.abs(predictions - repeated_prior).mean(dim = 0)
        stdev = (predictions - repeated_prior).std(dim = 0)
        
        results = {
            'mean' : mean,
            'stdev' : stdev,
            'smoothed_english_prior' : english_prior
        }
        
        results_path = os.path.join(analysis_folder, 'prior_compare_stats.pt')
        torch.save(results, results_path)
    except: import pdb; pdb.set_trace()
    
    return results
    
    
def get_event_df(event_path):
    return tflogs2pandas.tflog2pandas(event_path)
    
if __name__ == '__main__':
    
    # Decoder
    model_folder = './experiments/decoder_baseline/one_kl_weight_with_ref_val/'
    checkpoint_name = 'ckpts_epoch=1-val_acc=81.349.ckpt'
    model_type = VAE
    
    ## Encoders
    #model_type = Tagger
    
    # Encoder: finetuned
    # model_folder = './experiments/encoder_for_baseline/english_trained_dutch_val'
    # checkpoint_name = 'ckpts_epoch=0-val_acc=81.318.ckpt'
    
    checkpoint_path = os.path.join(model_folder, 'ckpts', checkpoint_name)
    model = model_type.load_from_checkpoint(checkpoint_path)
    
    analysis_path = get_analysis_path(checkpoint_path)
    predictions_path = os.path.join(analysis_path, 'predictions.pt')
    if not os.path.exists(predictions_path):
        raw_predictions = get_all_predictions(model, ['Dutch'])
        analysis_path = get_analysis_path(checkpoint_path)
        torch.save(raw_predictions, predictions_path)
    else:
        raw_predictions = torch.load(predictions_path)
    
    #compare_validation_predictions(model, checkpoint_path)
    results = compare_english_prior(model, checkpoint_path)
    
    