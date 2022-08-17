
from enumeration import Split

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import torch

import util
import constant
from predict import predict_utils

def get_all_predictions(model, langs, phase):
    predictions = {}
    model.eval()
    for lang in langs:
        trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0)
        model.reset_metrics('val')
        dataloader = model.get_unshuffled_dataloader(lang, phase if phase != 'val' else Split.dev)
        predictions[lang] = trainer.predict(model, dataloaders = [dataloader], return_predictions = True)
        try:
            predictions[lang] = [output[1]['pos'][0].exp() for output in predictions[lang]]
        except: import pdb; pdb.set_trace()
    return predictions

    
def softmax_to_accuracy(model, softmaxes, phase):
    for lang, softmax in softmaxes.items():
        logits = torch.cat(list(map(lambda tensor : tensor.reshape(-1, tensor.shape[-1]), softmax)), dim = 0)
        flat_predicted_labels = torch.argmax(softmax, dim=-1)
        outputs, labels = predict_utils.clean_padded_labels_and_predictions(model, lang, flat_predicted_labels, phase)
        accuracy = (torch.sum(outputs == labels) / outputs.shape[0]).item()
    return accuracy
    
def prediction_to_accuracy(model, predictions, phase):
    """predictions: {lang : (batch, position, class)}"""
    softmax_dict = { lang : raw_logits.softmax(dim=-1)  for lang, raw_logits in predictions.items() }
    return softmax_to_accuracy(model, predictions, phase):
    
def get_predictions(model, language, analysis_path, phase):
    if not os.path.exists(analysis_path): os.makedirs(analysis_path)
    predictions_path = os.path.join(analysis_path, f'{phase}_predictions.pt')
    if not os.path.exists(predictions_path):
        raw_predictions = get_all_predictions(model, [language], phase)
        torch.save(raw_predictions, predictions_path)
    else:
        raw_predictions = torch.load(predictions_path)
    return raw_predictions
    

