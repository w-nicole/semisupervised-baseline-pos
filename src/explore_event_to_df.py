
import argparse
import os
import glob
import matplotlib.pyplot as plt

from explore import tflogs2pandas

def get_event_df(event_path):
    return tflogs2pandas.tflog2pandas(event_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path_section', type=str)
    
    args = parser.parse_args()
    filenames = glob.glob(os.path.join('./experiments', args.model_path_section, 'events.out.tfevents*'))
    
    event_path = sorted(filenames)[0]
    df_path = event_path + '.csv'
    df = get_event_df(event_path)
    
    event_file_if_valid = sorted(filenames)[0]
    assert len(filenames) == 1 or set(filenames) == {event_file_if_valid, df_path}, filenames
    
    df.to_csv(df_path)
    
    analysis_path = os.path.join('./experiments', args.model_path_section)
    for current_metric in ['KL', 'target_KL', 'MSE', 'decoder_loss', 'val_Dutch_acc']:
        figure_path = os.path.join(analysis_path, f'{current_metric}.png')
        subset = df[df.metric == current_metric]
        print(subset['metric'])
        plt.title(current_metric)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(range(subset.shape[0]), subset['value'])
        plt.savefig(figure_path)
        plt.figure()
    
    print(f'Written events file to {df_path}')