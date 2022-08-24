
import util
from constant import LABEL_PAD_ID
import predict.predict_utils as predict_utils

import numpy as np
import torch

def clean_for_ensemble_softmax(softmax, flat_labels_by_dataloader):
    flat_softmax = np.concatenate([
            sent_output.reshape(-1, sent_output.shape[-1]).cpu().numpy()
            for sent_output in softmax
        ], axis = 0)
    assert len(flat_softmax.shape) == 2, flat_softmax.shape
    assert len(flat_labels_by_dataloader.shape) == 1, flat_labels_by_dataloader.shape
    not_padding_indices = np.where(flat_labels_by_dataloader != LABEL_PAD_ID)[0]
    try:
        clean_softmax = np.take(flat_softmax, not_padding_indices, axis = 0)
    except: import pdb; pdb.set_trace()
    
    return torch.from_numpy(clean_softmax)

def get_clean_matching_predictions(raw_softmaxes, flat_labels_by_dataloader):
    clean_softmaxes = clean_for_ensemble_softmax(raw_softmaxes, flat_labels_by_dataloader)
    predictions = torch.argmax(clean_softmaxes, -1)
    return predictions
    