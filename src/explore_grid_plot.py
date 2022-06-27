
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    requested_baseline_path = 'requested_runs'
    figure_folder = os.path.join('./experiments', requested_baseline_path)
    if not os.path.exists(figure_folder): os.makedirs(figure_folder)
    
    model_path_sections = {
        'normal' : 'phase_3_variants/normal/linear_lr_1e-2_bs16/version_0',
        'small_kl' : 'phase_3_variants/small_kl/linear_lr_1e-3_bs16/version_0',
        'random' : 'phase_3_variants/random/linear_lr_1e-1_bs16/version_0',
    }
    
    colors = {
        'normal' : 'black',
        'small_kl' : 'blue', 
        'random' : 'red'
    }
    
    patterns = {
        'train' : '+',
        'val' : 'o'
    }
    
    phase_3_metric_types = [
        'MSE',
        'KL_against_train_English',
        'KL_against_val_Dutch',
        'acc',
        'loss_KL'
    ]
    
    for metric in phase_3_metric_types:
        for phase in ['train', 'val']:
            for language in ['English', 'Dutch']:
                for model, path_section in model_path_sections.items():
                    df_paths = glob.glob(os.path.join('./experiments', path_section, '*.csv'))
                    assert len(df_paths) == 1, f'{metric}, {model}, {df_paths}'
                    df = pd.read_csv(df_paths[0])
                    metric_data = df[df.metric == f'{phase}_{language}_{metric}']
                    plt.plot(
                        range(metric_data.shape[0]), metric_data.value,
                        color = colors[model], marker = patterns[phase],
                        label = model, linewidth = 2
                    )
                current_folder = os.path.join(figure_folder, metric, phase)
                if not os.path.exists(current_folder): os.makedirs(current_folder)
                plt.title(f'{phase}_{language}_{metric}')
                plt.xlabel('Epochs'); plt.ylabel('Metric value'); plt.legend()
                plt.savefig(os.path.join(current_folder, f'{language}.png'))
                plt.clf()
    
    print(f'Written to {figure_folder}')
            