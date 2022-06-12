

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
    
def to_flat_label(flat_probs):
    flat_predicted_labels = torch.argmax(flat_probs, dim = -1)
    return flat_predicted_labels

def predict_validation(model, langs):
    
    predictions = {}
    for lang in langs:
        if lang == "English" and isinstance(model, VAE):
            print('Skipping English because it is a labeled case.')
            continue
        trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0)
        model.reset_metrics()
        dataloader = model.get_dataloader(lang, Split.dev)
        current_predictions = trainer.predict(model, dataloaders = [dataloader], return_predictions = True)

    try:
        predictions[lang] = util.remove_from_gpu(
            torch.cat([
                        F.softmax(outputs[1].reshape(-1, model.nb_labels), dim = -1)
                        for outputs in current_predictions
                    ], dim = 0
            )
        )
    except:
        import pdb; pdb.set_trace()
    return predictions
    
    
def get_analysis_path(checkpoint_path):
    path_components = checkpoint_path.split('/')
    decoder_folder = util.get_folder_from_checkpoint_path(checkpoint_path)
    checkpoint_name = path_components[-1]
    analysis_path = os.path.join(decoder_folder, os.path.splitext(checkpoint_name)[0])
    print(f'Analysis path: {analysis_path}')
    return analysis_path
        
        
def get_validation_predictions(model, checkpoint_path):
    
    analysis_path = get_analysis_path(checkpoint_path)
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    predictions = predict_validation(model, ['English', 'Dutch'])
    torch.save(predictions, os.path.join(analysis_path, 'predictions.pt'))
    
    return predictions
    
def get_padded_labels(model, lang):
    
    dataloader = model.get_dataloader(lang, Split.dev)
    raw_labels = []
    for batch in dataloader:
        raw_labels.append(batch['labels'].flatten())
    labels = torch.cat([batch['labels'].flatten() for batch in dataloader])
    return labels
    
def compare_validation_predictions(model, checkpoint_path):
    
    analysis_path = get_analysis_path(checkpoint_path)
    
    predictions = torch.load(os.path.join(analysis_path, 'predictions.pt'))
    
    try:
        for lang, raw_logits in predictions.items():
            raw_outputs = torch.cat([to_flat_label(raw_logits[i]) for i in range(raw_logits.shape[0])], dim = 0)
            
            padded_labels = util.remove_from_gpu(get_padded_labels(model, lang))
            mask_for_non_pad = (padded_labels != metric.LABEL_PAD_ID)
            labels = padded_labels[mask_for_non_pad]
            outputs = raw_outputs[mask_for_non_pad]
    
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

def compare_english_prior(model):
    
    try:
        english_prior = model.get_smoothed_english_prior()
        predictions = torch.load(os.path.join(analysis_folder, 'predictions.pt'))['Dutch']
        repeated_prior = english_prior.unsqueeze(0).repeat(predictions.shape[0], 1)
        
        mean = (predictions - repeated_prior).mean(dim = 0)
        stdev = (predictions - repeated_prior).std(dim = 0)
        
        results = {
            'mean' : mean,
            'stdev' : stdev,
            'smoothed_english_prior' : english_prior
        }
        
        results_path = os.path.join(get_analysis_path(checkpoint_path), 'prior_compare_stats.pt')
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
    #get_validation_predictions(model, checkpoint_path)
    #compare_validation_predictions(model, checkpoint_path)
    results = compare_english_prior(model)
    
    pprint(results)
    
    
    