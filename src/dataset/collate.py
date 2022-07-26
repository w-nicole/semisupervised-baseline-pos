
# Entire file added to the codebase.

import torch_scatter

def average_embeddings(padded_embeddings, padded_averaging_indices):
    """
    Returns a stack of averaged representations with padding/sentence structure.
    Expects padded_embeddings to NOT have CLS/SEP removed.
    """
    # Will pad with 0s, which is also desired for the LSTM.
    raw_averages = torch_scatter.scatter(
        padded_embeddings.float(), padded_averaging_indices.long(),
        dim = 1, reduce = 'mean'
    )
    # All of the padding is averaged to position 0.
    clean_averages = raw_averages[:, 1:, :]
    return clean_averages
    