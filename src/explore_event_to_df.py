
import argparse
import os
import glob
import matplotlib.pyplot as plt

from explore import tflogs2pandas
import util

def get_event_df(event_path):
    return tflogs2pandas.tflog2pandas(event_path)

def plot_subset(metric_key, color, marker):
    subset = df[df.metric == metric_key]
    if subset.shape[0] == 0:
        return
    print(subset['metric'])
    plt.plot(
        range(subset.shape[0]), subset['value'], label = metric_key,
        linewidth = 2, marker = marker, color = color,
        alpha = 0.2 if metric_key.startswith('train') else 1
    )
    
def complete_plot(analysis_path, metric):
    figure_path = os.path.join(analysis_path, f'{metric}.png')
    plt.title(metric)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(figure_path)
    plt.figure()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path_section', type=str)
    parser.add_argument('--is_phase_3_decoder', default = False, type=util.str2bool)
    
    args = parser.parse_args()
    filenames = glob.glob(os.path.join('./experiments', args.model_path_section, 'events.out.tfevents*'))
    
    event_path = sorted(filenames)[0]
    df_path = event_path + '.csv'
    df = get_event_df(event_path)
    
    event_file_if_valid = sorted(filenames)[0]
    assert len(filenames) == 1 or set(filenames) == {event_file_if_valid, df_path}, filenames
    
    df.to_csv(df_path)
    
    analysis_path = os.path.join('./experiments', args.model_path_section)
    phase_3_metric_types = [
        'MSE',
        'decoder_loss',
        'encoder_loss',
        'KL_against_train_English',
        'KL_against_val_Dutch',
        'acc',
        'loss_KL'
    ]
    
    english_pair = ('English', 'g')
    languages = [english_pair, ('Dutch', 'r')]
    phases = [('train', '+'), ('val', 'o')]
    
    # Metrics that are the same for both BaseVAE and VAE.
    if not args.is_phase_3_decoder:
        for language, color in languages:
            for phase, marker in phases:
                plot_subset(f'{phase}_{language}_MSE', color, marker)
        complete_plot(analysis_path, 'MSE')
    else:
        for metric in phase_3_metric_types:
            for language, color in languages:
                for phase, marker in phases:
                    plot_subset(f'{phase}_{language}_{metric}', color, marker)
            complete_plot(analysis_path, metric)
        
    # print(f'Written events file to {df_path}')