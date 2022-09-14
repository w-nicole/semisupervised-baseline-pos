
from enumeration import Split

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import torch

import util
import constant
from predict import predict_utils

def get_all_softmaxes(model, dataloader):
    model.eval()
    trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0)
    with torch.no_grad():
        raw_predictions = trainer.predict(model, dataloaders = [dataloader], return_predictions = True)
        softmaxes = [output[1]['pos'][0].exp() for output in raw_predictions]
    if not model.hparams.double_pass:
        return softmaxes
    # Format such that you have the two views.
    assert len(softmaxes) % 2 == 0, len(softmaxes)
    first_half = softmaxes[:softmaxes // 2]
    second_half = softmaxes[softmaxes // 2:]
    return (first_half, second_half)

    
def softmax_to_accuracy(softmax, padded_labels):
    """softmax: {lang : (batch, position, class)}"""
    flat_softmax = torch.cat(list(map(lambda tensor : tensor.reshape(-1, tensor.shape[-1]), softmax)), dim = 0)
    flat_predicted_labels = torch.argmax(flat_softmax, dim=-1)
    outputs, labels = predict_utils.clean_padded_labels_and_predictions(padded_labels, flat_predicted_labels)
    accuracy = (torch.sum(outputs == labels) / outputs.shape[0]).item()
    return accuracy
    
    
def get_softmaxes(model, dataloader, analysis_path, phase):
    if not os.path.exists(analysis_path): os.makedirs(analysis_path)
    predictions_path = os.path.join(analysis_path, f'{phase}_predictions.pt')
    if not os.path.exists(predictions_path):
        raw_predictions = get_all_softmaxes(model, dataloader)
        torch.save(raw_predictions, predictions_path)
    else:
        raw_predictions = torch.load(predictions_path)
    return raw_predictions