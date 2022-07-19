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
from model import Tagger

from pprint import pprint

def get_all_predictions(model, langs):
    predictions = {}
    for lang in langs:
        trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0)
        model.reset_metrics('val')
        dataloader = model.get_dataloader(lang, Split.dev)
        predictions[lang] = trainer.predict(model, dataloaders = [dataloader], return_predictions = True)
        predictions[lang] = [output[1].exp() for output in predictions[lang]]
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

def clean_padded_labels_and_predictions(model, lang, padded_predictions):
    padded_labels = util.remove_from_gpu(get_padded_labels(model, lang))
    mask_for_non_pad = (padded_labels != metric.LABEL_PAD_ID)
    labels = padded_labels[mask_for_non_pad]
    outputs = padded_predictions[mask_for_non_pad]
    return outputs.cpu(), labels.cpu()
    
def compare_validation_predictions(model, predictions, checkpoint_path):
    """predictions: {lang : (batch, position, class)}"""
    
    analysis_path = get_analysis_path(checkpoint_path)
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    for lang, raw_logits in predictions.items():
        logits = torch.cat(list(map(lambda tensor : tensor.reshape(-1, tensor.shape[-1]), raw_logits)), dim = 0)
        flat_predicted_labels = torch.argmax(logits.softmax(dim=-1), dim=-1)
        outputs, labels = clean_padded_labels_and_predictions(model, lang, flat_predicted_labels)

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


if __name__ == '__main__':
    
    # Decoder
    model_folder = './pre-sweep-experiments/components/mbert_pretrained/linear/version_1ylqtlld'
    checkpoint_name = 'ckpts_epoch=2-val_English_acc_epoch_monitor=96.807.ckpt'
    model_type = Tagger
    
    checkpoint_path = os.path.join(model_folder, 'ckpts', checkpoint_name)
    model = model_type.load_from_checkpoint(checkpoint_path)
    
    analysis_path = get_analysis_path(checkpoint_path)
    if not os.path.exists(analysis_path): os.makedirs(analysis_path)
    predictions_path = os.path.join(analysis_path, 'predictions.pt')
    if True:#not os.path.exists(predictions_path):
        raw_predictions = get_all_predictions(model, ['Dutch'])
        analysis_path = get_analysis_path(checkpoint_path)
        torch.save(raw_predictions, predictions_path)
    else:
        raw_predictions = torch.load(predictions_path)
    
    compare_validation_predictions(model, raw_predictions, checkpoint_path)