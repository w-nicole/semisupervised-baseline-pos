
# Entire file added to the codebase.

import torch
import torch.nn.utils
import constant
import util

def clean_indices(indices):
    # Below is valid because real averaging indices are given from 1
    assert constant.START_END_INDEX_PADDING < 1
    return indices[indices != constant.START_END_INDEX_PADDING]
    
def average_embeddings(padded_embeddings, padded_start_indices, padded_end_indices):
    """
    Returns a stack of averaged representations with padding/sentence structure.
    Expects padded_embeddings to NOT have CLS/SEP removed.
    """
    
    all_clean_averaged_embeddings = []
    for index, sentence in enumerate(padded_embeddings):
        try:
            start_indices = clean_indices(padded_start_indices[index])
        except:
            import pdb
            pdb.set_trace()
        end_indices = clean_indices(padded_end_indices[index])
        assert start_indices.shape == end_indices.shape
        
        clean_averaged_embeddings = []
        
        for start_index, end_index in zip(start_indices, end_indices):
            raw_embeddings = sentence[start_index:end_index]
            assert len(raw_embeddings.shape) == 2
            average = torch.mean(raw_embeddings, dim = 0)
            clean_averaged_embeddings.append(average)
        
        assert start_indices.shape[0] == len(clean_averaged_embeddings)
       
        # Handle empty sentence examples
        to_use_embeddings = clean_averaged_embeddings if clean_averaged_embeddings else [util.apply_gpu(torch.zeros((padded_embeddings.shape[-1],)))]
        sentence_embeddings = torch.stack(to_use_embeddings, axis = 0)

        all_clean_averaged_embeddings.append(sentence_embeddings)
    
    padded_averages = torch.nn.utils.rnn.pad_sequence(
        all_clean_averaged_embeddings,
        batch_first = True,
        padding_value = constant.PACK_PADDING
    )
    
    assert padded_averages.shape[0] == padded_embeddings.shape[0]
    return padded_averages
    