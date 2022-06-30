
# Small pieces taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Updated arguments for encode_sent call.

import argparse
import glob

import torch
import os
import json

from model import Tagger
from enumeration import Split
import util

from pprint import pprint

def get_non_label_mask(labels, tensor, label):
    repeated_labels = util.apply_gpu(labels.unsqueeze(1).repeat(1, tensor.shape[-1]))
    mask = (repeated_labels != label)
    return mask
    
def set_non_label_to_zero(labels, tensor, label):
    repeated_label_mask = get_non_label_mask(labels, tensor, label)
    zeros = torch.zeros(tensor.shape)
    clean_tensor = torch.where(repeated_label_mask.cpu(), tensor.cpu(), zeros.cpu())
    return clean_tensor
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_part', type=str)
    
    args = parser.parse_args()
    filenames = glob.glob(os.path.join('./experiments', args.path_part, 'ckpts/*.ckpt'))
    assert len(filenames) == 1, filenames
    checkpoint_path = filenames[0]

    model = Tagger.load_from_checkpoint(checkpoint_path)
    model = util.apply_gpu(model.eval())
    analysis_path = './experiments/data_analysis'
    
    for language in ['English', 'Dutch']:
        for phase in ['train', 'val']:
            
            print(f'Processing: {language}, {phase}')
            label_counts = model.get_label_counts(language, Split.dev if phase == 'val' else 'train').cpu()
            
            dataloader = model.val_dataloader()[0]
            assert dataloader.dataset.lang == 'English'
            
            all_states = []
            all_padded_labels = []
            
            with torch.no_grad():
                for batch in dataloader:
                    # Adapted from `tagger.py`, rearranged/updated the arguments
                    
                    if isinstance(model, Tagger):
                        hs = model.encode_sent(
                            util.apply_gpu(batch["sent"]),
                            util.apply_gpu(batch["start_indices"]),
                            util.apply_gpu(batch["end_indices"]),
                            batch["lang"]
                        )
                    else:
                        assert isinstance(model, BaseVAE)
                        hs = model.
                        
                    # end taken
                    flat_states = hs.reshape(-1, hs.shape[-1])
                    all_states.append(flat_states.cpu())
                    all_padded_labels.append(batch['labels'].flatten().cpu())
                
                states = torch.cat(all_states, dim = 0)
                padded_labels = torch.cat(all_padded_labels, dim = 0)
                
                # Compute centroids
                
                centroids = {}
                for label in range(model.nb_labels):
                    centroids[label] = 
                
                # Compute MSE per class
                average_distances = {}
                all_distances = 0
                for label in range(model.nb_labels):
                    key = str(label)
                    if label_counts[label] == 0:
                        average_distances[key] = 0
                        continue
                    label_hs = set_non_label_to_zero(padded_labels, states, label).cpu()
                    assert len(label_hs.shape) == 2
                    centroid = (label_hs.sum(dim=0) / label_counts[label]).cpu()
                    mask = get_non_label_mask(padded_labels, states, label).cpu()
                    current_distance = ((label_hs - centroid) * mask).sum()
                    average_distances[key] = (current_distance / label_counts[label]).item()
                    all_distances += current_distance
                    
                average_distances['all'] = (all_distances / torch.sum(label_counts)).item()
        
            standard_average_distances = { key : value / flat_states.shape[-1] for key, value in average_distances.items() }
            
            stats_folder = os.path.join(analysis_path, 'centroid_baseline', args.path_part)
            if not os.path.exists(stats_folder): os.makedirs(stats_folder)
            stats_path = os.path.join(stats_folder, f"{phase}_{language}.json")
            standardized_stats_path = os.path.join(stats_folder, f"standard_{phase}_{language}.json")
            
            for path, data_dict in zip(
                [stats_path, standardized_stats_path],
                [average_distances, standard_average_distances]
            ):
                if os.path.exists(path):
                    os.remove(path)
                
                with open(path, 'w') as f:
                    json.dump(data_dict, f)
    
    