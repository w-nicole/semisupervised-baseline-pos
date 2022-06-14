
# Small pieces taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Updated arguments for encode_sent call.

import torch
import os
import json

from model import BaseVAE
from enumeration import Split
import util

from pprint import pprint

def get_non_label_mask(labels, tensor, label):
    repeated_labels = util.apply_gpu(labels.unsqueeze(1).repeat(1, tensor.shape[-1]))
    mask = (repeated_labels != label)
    return mask
    
def set_non_label_to_zero(labels, tensor, label):
    repeated_label_mask = get_non_label_mask(labels, tensor, label)
    clean_tensor = torch.where(
        repeated_label_mask,
        tensor, util.apply_gpu(torch.zeros(tensor.shape))
    )
    return clean_tensor
        
if __name__ == '__main__':
    checkpoint_path = "./experiments/decoder_for_baseline/no_auxiliary_short/ckpts/ckpts_epoch=19-decoder_loss=0.066.ckpt"
    model = BaseVAE.load_from_checkpoint(checkpoint_path)
    model = util.apply_gpu(model.eval())
    analysis_path = './experiments/data_analysis'
    
    try:
        label_counts = model.get_label_counts('English', Split.dev)
        
        dataloader = model.val_dataloader()[0]
        assert dataloader.dataset.lang == 'English'
        
        all_states = []
        all_padded_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Adapted from `tagger.py`, rearranged/updated the arguments
                hs = model.encode_sent(
                    util.apply_gpu(batch["sent"]),
                    util.apply_gpu(batch["start_indices"]),
                    util.apply_gpu(batch["end_indices"]),
                    batch["lang"]
                )
                # end taken
                flat_states = hs.reshape(-1, hs.shape[-1])
                all_states.append(flat_states)
                all_padded_labels.append(batch['labels'].flatten())
                
            # Compute MSE per class
            states = torch.cat(all_states, dim = 0)
            padded_labels = torch.cat(all_padded_labels, dim = 0)
            
            average_distances = {}
            all_distances = 0
            for label in range(model.nb_labels):
                key = str(label)
                if label_counts[label] == 0:
                    average_distances[key] = 0
                    continue
                label_hs = set_non_label_to_zero(padded_labels, states, label)
                assert len(label_hs.shape) == 2
                centroid = label_hs.sum(dim=0) / label_counts[label]
                mask = get_non_label_mask(padded_labels, states, label)
                current_distance = ((label_hs - centroid) * mask).sum() / label_counts[label]
                average_distances[key] = current_distance.item()
                all_distances += current_distance
                
            average_distances['all'] = (all_distances / torch.sum(label_counts)).item()
    
        updated_average_distances = { key : value / flat_states.shape[-1] for key, value in average_distances.items() }
        pprint(updated_average_distances)
        
        stats_path = os.path.join(analysis_path, 'centroid_baseline.json')
        with open(stats_path, 'w') as f:
            json.dump(updated_average_distances, f)
            
    except: import pdb; pdb.set_trace()
    
    
    