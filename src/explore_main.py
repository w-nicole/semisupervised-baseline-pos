

from explore import tflogs2pandas

def get_event_df(event_path):
    return tflogs2pandas.tflog2pandas(event_path)

# For running in interactive mode
# Run everything from the /src directory.

# import explore_main
# import os

# ------------------

# Getting the event df

# event_folder = '../experiments/decoder_baseline/version_0'
# event_name = 'events.out.tfevents.1654803013.node0023.1339734.0'
# event_path = os.path.join(event_folder, event_name)
# df = explore_main.get_event_df(event_path)
# df.to_csv(os.path.join(event_folder, event_name + '.csv'))
