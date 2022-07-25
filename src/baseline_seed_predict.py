import os
import pandas as pd
import glob
import json
import torch

from model import LatentBase, Tagger
from predict import predict_utils, predict

def predict_over_languages(checkpoint_path, model_class, phase):
    
    model = model_class.load_from_checkpoint(checkpoint_path)
    languages = "Bulgarian Danish German English Spanish Persian Hungarian Italian Dutch Polish Portuguese Romanian Slovak Slovenian Swedish".split()
    
    analysis_parent_path = os.path.join(predict_utils.get_analysis_path(checkpoint_path), f'{phase}_predictions')
    accuracies = {}
    for language in languages:
        print(language)
        analysis_path = os.path.join(analysis_parent_path, language)
        raw_predictions = predict.get_predictions(model, language, analysis_path, phase)
        accuracy = predict.prediction_to_accuracy(model, raw_predictions, checkpoint_path, phase)
        accuracies[language] = accuracy
    
    df = pd.DataFrame.from_records([accuracies])
    df.to_csv(os.path.join(analysis_parent_path, f'{phase}_accuracies.csv'))
    return df
    
if __name__ == '__main__':
    seeds = [42, 0, 1, 2, 3]

    model_names = ['baseline', 'latent_base_64']
    name_to_languages = {
        # 'baseline' : [
        #     'Bulgarian', 'Danish', 'German', 'English', 'Spanish', 'Persian',
        #     'Hungarian', 'Italian', 'Dutch', 'Polish', 'Portuguese', 'Romanian',
        #     'Slovak', 'Slovenian', 'Swedish'
        # ],
        'baseline' : ['English', 'Dutch'],
        'latent_base_64': ['English', 'Dutch']
    }
    
    phase = 'train'
    
    for model_type in model_names:
        languages = name_to_languages[model_type]
        model_path = os.path.join('./pre-opt-experiments/sweeps/seed', model_type)
        for seed in seeds:
            template = os.path.join(model_path, f'seed={seed}', 'version_*', 'ckpts', '*.ckpt')
            matches = glob.glob(template)
            assert len(matches) == 1, f'{template}, {matches}'
            checkpoint_path = matches[0]
            df = predict_over_languages(checkpoint_path, Tagger if model_type == 'baseline' else LatentBase, phase)
        
   