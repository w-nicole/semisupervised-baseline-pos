
from model import VAE, Tagger, BaseVAE
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

def clean_padded_labels_and_predictions(model, lang, padded_predictions, padded_labels):
    mask_for_non_pad = (padded_labels != metric.LABEL_PAD_ID)
    labels = padded_labels[mask_for_non_pad]
    outputs = padded_predictions[mask_for_non_pad]
    return outputs, labels
    
def clean_padded_predictions_without_label(model, lang, padded_predictions)
    padded_labels = util.remove_from_gpu(get_padded_labels(model, lang))
    return clean_padded_labels_and_predictions(model, lang, padded_predictions, padded_labels)
    
def get_base_vae_prediction(model, batch):
    inputs = { key : util.apply_gpu(item) if isinstance(item, torch.Tensor) else item for key, item in batch.items() }
    logits = model.classifier(model.calculate_hidden_states(inputs))
    return logits
  
def get_all_base_vae_predictions(model, lang):
    try:
        model = util.apply_gpu(model)
        dataloader = model.get_dataloader(lang, Split.dev)
        raw_logits = []
        for batch in dataloader:
            raw_logits.append(util.remove_from_gpu(get_base_vae_prediction(model, batch).reshape(-1, model.nb_labels)))
        raw_logits = torch.cat(raw_logits, dim = 0)
    except: import pdb; pdb.set_trace()
    return raw_logits
    
def compare_validation_predictions(model, raw_logits, checkpoint_path):
    """raw_logits: (batch, position, class)"""
    try:
        lang = 'English'
        analysis_path = get_analysis_path(checkpoint_path)
        
        flat_predicted_labels = torch.argmax(raw_logits.softmax(dim=-1), dim=-1)
        outputs, labels = clean_padded_predictions_without_label(model, lang, flat_predicted_labels)
    
        plt.hist(outputs.numpy().flat, alpha = 0.5, color = 'r', label = 'predicted')
        plt.hist(labels.numpy().flat, alpha = 0.5, color = 'g', label = 'true')
        
        accuracy = (torch.sum(outputs == labels) / outputs.shape[0]).item()
        plt.legend()
        plt.title(f'Frequency of class predictions vs labels. Accuracy: {round(accuracy * 100, 4)}%')
        plt.xlabel('Numerical label')
        plt.ylabel('Counts')
        plt.xticks(range(len(constant.UD_POS_LABELS)), constant.UD_POS_LABELS, rotation = 45)
        figure_path = os.path.join(analysis_path, f'prediction_comparison_val_{lang}.png')
        plt.savefig(figure_path)
        print(f'Figure written to: {figure_path}')
        plt.figure()
    except: import pdb; pdb.set_trace()

if __name__ == '__main__':
    
    # Decoder
    model_folder = './experiments/decoder_for_baseline/long_train'
    checkpoint_name = 'ckpts_epoch=50-decoder_loss=0.060.ckpt'
    model_type = BaseVAE
    
    checkpoint_path = os.path.join(model_folder, 'ckpts', checkpoint_name)
    model = model_type.load_from_checkpoint(checkpoint_path)
    model = model.eval()
    
    # analysis_path = get_analysis_path(checkpoint_path)
    # predictions_path = os.path.join(analysis_path, 'predictions.pt')
    # if not os.path.exists(predictions_path):
    #     raw_predictions = get_all_predictions(model, ['Dutch'])
    #     analysis_path = get_analysis_path(checkpoint_path)
    #     torch.save(raw_predictions, predictions_path)
    # else:
    #     raw_predictions = torch.load(predictions_path)
    
    lang = 'English'
    raw_logits = get_all_base_vae_predictions(model, lang)
    compare_validation_predictions(model, {lang : raw_logits}, checkpoint_path)
    
    