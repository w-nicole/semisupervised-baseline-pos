

from explore import tflogs2pandas
from model import EncoderDecoder, Tagger
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

def get_all_checkpoint_paths(checkpoint):
    checkpoints = glob.glob(os.path.join(decoder_folder, 'ckpts'))
    return checkpoints
    
def to_flat_label(flat_logits):
    flat_probs = F.softmax(flat_logits, dim = -1)
    flat_predicted_labels = torch.argmax(flat_probs, dim = -1)
    return flat_predicted_labels

def predict_validation(model, langs):
    
    predictions = {}
    for lang in langs:
        trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0)
        model.reset_metrics()
        dataloader = model.get_dataloader(lang, Split.dev)
        current_predictions = trainer.predict(model, dataloaders = [dataloader], return_predictions = True)

        try:
            predictions[lang] = util.remove_from_gpu(
                to_flat_label(
                    torch.cat([
                            outputs[1].reshape(-1, model.nb_labels)
                            for outputs in current_predictions
                        ], dim = 0
                    )
                )
            )
        except:
            import pdb; pdb.set_trace()
    return predictions
    
    
def get_analysis_path(checkpoint_path):
    
    path_components = checkpoint_path.split('/')
    checkpoint_name = path_components[-1]
    decoder_folder = '/'.join(path_components[:-2])
    analysis_path = os.path.join(decoder_folder, os.path.splitext(checkpoint_name)[0])
    print(f'Analysis path: {analysis_path}')
    
    return analysis_path
        
        
def get_validation_predictions(checkpoint_path, model_type):
    
    analysis_path = get_analysis_path(checkpoint_path)
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    model = model_type.load_from_checkpoint(checkpoint_path)
    
    predictions = predict_validation(model, ['Dutch', 'German'])
    torch.save(predictions, os.path.join(analysis_path, 'predictions.pt'))
    
    return predictions
    
def get_padded_labels(model, lang):
    
    dataloader = model.get_dataloader(lang, Split.dev)
    raw_labels = []
    for batch in dataloader:
        raw_labels.append(batch['labels'].flatten())
    labels = torch.cat([batch['labels'].flatten() for batch in dataloader])
    return labels
    
def compare_validation_predictions(checkpoint_path, model_type):
    
    model = model_type.load_from_checkpoint(checkpoint_path)
    analysis_path = get_analysis_path(checkpoint_path)
    
    predictions = torch.load(os.path.join(analysis_path, 'predictions.pt'))
    
    for lang, raw_outputs in predictions.items():
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

def get_event_df(event_path):
    return tflogs2pandas.tflog2pandas(event_path)
    
if __name__ == '__main__':
    
    # Decoder
    #model_folder = './experiments/decoder_baseline/version_0/'
    #checkpoint_name = 'ckpts_epoch=1-val_acc=7.616.ckpt'
    #model_type = EncoderDecoder
    
    ## Encoders
    model_type = Tagger
    
    # Encoder: finetuned
    # model_folder = './experiments/encoder_for_baseline/version_0'
    # checkpoint_name = 'ckpts_epoch=2-val_acc=97.057.ckpt'
    
    # Encoder: frozen
    model_folder = './experiments/was_ok_freeze/layer_search_frozen/udpos/0-shot-finetune-layerconcat/bert-base-multilingual-cased/bs16-lr5e-5-ep4/version_0'
    checkpoint_name = 'ckpts_epoch=3-val_English_acc=95.181.ckpt'
    
    checkpoint_path = os.path.join(model_folder, 'ckpts', checkpoint_name)
    
    get_validation_predictions(checkpoint_path, model_type)
    compare_validation_predictions(checkpoint_path, model_type)
    
    
# For running in interactive mode
# Run everything from the /src directory.

# import explore_main
# import os

# ------------------

# Getting the event df

# event_folder = '../experiments/decoder_baseline/version_0'
# event_name = 'events.out.tfevents.1654803013.node0023.1339734.0'
# event_path = os.path.join(event_folder, event_name)
# df = explore_main.get_event_df(event_path)
# df.to_csv(os.path.join(event_folder, event_name + '.csv'))
