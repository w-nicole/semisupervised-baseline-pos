
import os
import pandas as pd
import glob

import predict.predict as predict
import predict.predict_utils as predict_utils

def predict_over_languages(checkpoint_path, model_class, phase, languages):
    
    model = model_class.load_from_checkpoint(checkpoint_path)
    
    analysis_parent_path = os.path.join(predict_utils.get_analysis_path(checkpoint_path), f'{phase}_predictions')
    accuracies = {}
    for language in languages:
        print(language)
        analysis_path = os.path.join(analysis_parent_path, language)
        raw_predictions = predict.get_predictions(model, language, analysis_path, phase)
        accuracy = predict.prediction_to_accuracy(model, raw_predictions, phase)
        accuracies[language] = accuracy
    
    df = pd.DataFrame.from_records([accuracies])
    df.to_csv(os.path.join(analysis_parent_path, f'{phase}_accuracies.csv'))
    return df
    
def get_sweep_matches(sweep_folder, hparams_template):
    hparams_path_matches = glob.glob(hparams_template)
    assert len(set(hparams_path_matches)) == len(hparams_path_matches), hparams_path_matches
    
    paths = []
    for hparams_path in hparams_path_matches:
        template = os.path.join(hparams_path, 'version_*', 'ckpts', '*.ckpt')
        matches = glob.glob(template)
        if len(matches) == 0:
            print(f'No matches for: {template}, skipping')
            continue
        assert len(matches) == 1, f'{template}, {matches}'
        paths.append(matches[0])
    return paths