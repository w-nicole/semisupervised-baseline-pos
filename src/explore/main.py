
import os
import glob

from explore import tflogs2pandas

def get_all_checkpoint_paths(checkpoint):

    checkpoints = glob.glob(os.path.join(decoder_folder, 'ckpts'))
    return checkpoints
    
def get_event_df(event_path):
    return tflogs2pandas.tflog2pandas(event_path)
    
# For running in interactive mode
# Run everything from the /src directory.

# from explore import main
# import os

# event_name = 'events.out.tfevents.1654710396.node0029.1205277.0'
# decoder_folder = '../experiments/decoder_baseline/version_0/'

# event_path = os.path.join(decoder_folder, event_name)
# df = main.get_event_df(event_path)
# paths = main.get_all_checkpoint_paths(decoder_folder)