import os
import glob
import pandas as pd

from predict import predict_utils

if __name__ == '__main__':
    seeds = [42, 0, 1, 2, 3]
    results = []

    model_names = ['latent_base_64', 'baseline']
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
        results = []
        languages = name_to_languages[model_type]
        model_path = os.path.join('./pre-opt-experiments/sweeps/seed', model_type)
        for seed in seeds:
            template = os.path.join(model_path, f'seed={seed}', 'version_*', 'ckpts', '*.ckpt')
            matches = glob.glob(template)
            assert len(matches) == 1, f'{template}, {matches}'
            checkpoint_path = matches[0]
            analysis_path = predict_utils.get_analysis_path(checkpoint_path)
            accuracy_df = pd.read_csv(os.path.join(analysis_path, f'{phase}_predictions', f'{phase}_accuracies.csv'))
            results.append(accuracy_df[languages])
        
        raw_accuracy_df = pd.concat(results)
        all_accuracy_df = raw_accuracy_df.append(raw_accuracy_df[languages].mean(axis=0), ignore_index = True)
        all_accuracy_df = all_accuracy_df.append(raw_accuracy_df[languages].std(axis=0), ignore_index = True)
        all_accuracy_df['type'] = list(map(lambda seed : f'seed={seed}', seeds)) + ['mean', 'stdev']
        
        all_accuracy_df.to_csv(f'./pre-opt-experiments/sweeps/seed/{model_type}/{phase}_accuracy_across_seeds.csv')
    