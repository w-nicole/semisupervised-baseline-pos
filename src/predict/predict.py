
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
        dataloader = model.get_dataloader(lang, phase if phase != 'val' else Split.dev)
        predictions[lang] = trainer.predict(model, dataloaders = [dataloader], return_predictions = True)
        try:
            predictions[lang] = [output[1]['pos'].exp() for output in predictions[lang]]
        except: import pdb; pdb.set_trace()
    return predictions

    
def prediction_to_accuracy(model, predictions, checkpoint_path, phase):
    """predictions: {lang : (batch, position, class)}"""
    
    analysis_path = predict_utils.get_analysis_path(checkpoint_path)
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    for lang, raw_logits in predictions.items():
        logits = torch.cat(list(map(lambda tensor : tensor.reshape(-1, tensor.shape[-1]), raw_logits)), dim = 0)
        flat_predicted_labels = torch.argmax(logits.softmax(dim=-1), dim=-1)
        outputs, labels = predict_utils.clean_padded_labels_and_predictions(model, lang, flat_predicted_labels, phase)
        accuracy = (torch.sum(outputs == labels) / outputs.shape[0]).item()
    
    return accuracy
    
def get_predictions(model, language, analysis_path, phase):
    if not os.path.exists(analysis_path): os.makedirs(analysis_path)
    predictions_path = os.path.join(analysis_path, f'{phase}_predictions.pt')
    if not os.path.exists(predictions_path):
        raw_predictions = get_all_predictions(model, [language], phase)
        torch.save(raw_predictions, predictions_path)
    else:
        raw_predictions = torch.load(predictions_path)
    return raw_predictions
    

