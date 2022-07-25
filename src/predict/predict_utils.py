
import torch
import os

from enumeration import Split
import util
import metric

def get_padded_labels(model, lang, phase):
    dataloader = model.get_dataloader(lang, Split.dev if phase == 'val' else phase)
    raw_labels = []
    for batch in dataloader:
        raw_labels.append(batch['pos_labels'].flatten())
    labels = torch.cat([batch['pos_labels'].flatten() for batch in dataloader])
    return labels

def clean_padded_labels_and_predictions(model, lang, padded_predictions, phase):
    padded_labels = util.remove_from_gpu(get_padded_labels(model, lang, phase))
    mask_for_non_pad = (padded_labels != metric.LABEL_PAD_ID)
    labels = padded_labels[mask_for_non_pad]
    outputs = padded_predictions[mask_for_non_pad]
    return outputs.cpu(), labels.cpu()
    
def get_analysis_path(checkpoint_path):
    path_components = checkpoint_path.split('/')
    decoder_folder = util.get_folder_from_checkpoint_path(checkpoint_path)
    checkpoint_name = path_components[-1]
    analysis_path = os.path.join(decoder_folder, os.path.splitext(checkpoint_name)[0])
    print(f'Analysis path: {analysis_path}')
    return analysis_path