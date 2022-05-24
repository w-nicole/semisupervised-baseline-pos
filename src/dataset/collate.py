
# Entire file added to the codebase.

import torch
import constant

def average_embeddings(cut_padded_embeddings, padded_start_indices, padded_end_indices):
    """
    Returns a stack of representations without padding used for input into encoders,
        without sentence structure.
    
    Expects cut_padded_embeddings to NOT have CLS/SEP removed.
    """

    # Below is valid because real averaging indices are given from 1
    assert constant.START_END_INDEX_PADDING < 1
    clean_indices = lambda indices : indices[indices != constant.START_END_INDEX_PADDING]
    clean_averaged_embeddings = []
    
    for index, embedding in enumerate(cut_padded_embeddings):
        start_indices = clean_indices(padded_start_indices[index])
        end_indices = clean_indices(padded_end_indices[index])
        assert start_indices.shape == end_indices.shape
        for start_index, end_index in zip(start_indices, end_indices):
            raw_embeddings = cut_padded_embeddings[start_index:end_index]
            print(raw_embeddings.shape)
            assert len(raw_embeddings.shape) == 2
            average = torch.mean(raw_embeddings, dim = 0)
            clean_averaged_embeddings.append(average)
            
    return torch.cat(clean_averaged_embeddings, axis = 0)
