
import os
import pandas as pd
import glob

from predict import softmaxes, predict_utils

def predict_over_languages(checkpoint_path, model_class, phase, languages, dataloaders_dict, padded_labels_dict, modifier):
    
    model = model_class.load_from_checkpoint(checkpoint_path)
    
    analysis_parent_path = os.path.join(predict_utils.get_analysis_path(checkpoint_path), f'{modifier}_{phase}_predictions')
    accuracies = {}
    for language in languages:
        print(language)
        padded_labels = padded_labels_dict[language]
        dataloader = dataloaders_dict[language]
        analysis_path = os.path.join(analysis_parent_path, language)
        raw_predictions = softmaxes.get_softmaxes(model, dataloader, analysis_path, phase)
        accuracy = softmaxes.softmax_to_accuracy(raw_predictions, padded_labels)
        accuracies[language] = accuracy
    
    df = pd.DataFrame.from_records([accuracies])
    df.to_csv(os.path.join(analysis_parent_path, f'{modifier}_{phase}_accuracies.csv'))
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