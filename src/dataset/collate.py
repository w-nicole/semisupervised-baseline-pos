
# Entire file added to the codebase.

import torch

# Affects code logic, hardcoded here.
AVERAGE_PADDING = 0

def decollate_embeddings_encoder(cut_padded_embeddings, padded_averaging_indices):
    """
    Returns a stack of representations without padding used for input into encoders,
        without sentence structure.
    
    Expects cut_padded_embeddings to NOT have CLS/SEP removed.
    """

    # Average the embeddings
    assert AVERAGE_PADDING == 0, "AVERAGE_PADDING must be 0 to correctly cut the 'averaged out' CLS, SEP, and padding representations."

    raw_averages = average_to_padded_tokens(cut_padded_embeddings, padded_averaging_indices)

    # Cut out of the index used to "average out" the padding representations
    # Note that the maximum length sequence averages to indices also shifted by 1,
    #   so the first index is purely non-real data
    #   despite maximum length sequence having no padding.

    clean_averages = raw_averages[:, 1:, :]
    cut_padded_embeddings

    # Real data already indexed from 1, so these now correspond to lengths
    subtoken_lengths, _ = torch.max(padded_averaging_indices, dim = 1)

    all_cut_embeddings = []

    # CLS/SEP has already been cut out of the embeddings via 1: above.
    for embeddings, subtoken_length in zip(clean_averages, subtoken_lengths):
        cut_embeddings = embeddings[:subtoken_length]
        all_cut_embeddings.append(cut_embeddings)

    return torch.cat(all_cut_embeddings, axis = 0)

# Designed to replace torch_scatter's scatter_mean due to installation issues.
def average_to_padded_tokens(padded_embeddings, averaging_indices):

    averaged_embeddings = []

    for averaging_index in torch.unique(averaging_indices):
        averaging_mask = (averaging_indices == averaging_index).unsqueeze(2).repeat((1, 1, padded_embeddings.shape[-1]))
        selected_embeddings = padded_embeddings * averaging_mask
        number_of_averaged_elements = torch.sum(averaging_mask, axis = 1)

        current_token_average = torch.sum(selected_embeddings, axis = 1) / number_of_averaged_elements
        averaged_embeddings.append(current_token_average)

    averaged_embeddings = torch.stack(averaged_embeddings, axis = 1)
    return averaged_embeddings
