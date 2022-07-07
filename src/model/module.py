
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes: removed irrelevant code.

import torch
import torch.nn as nn
import torch.nn.functional as F

class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian
    Approximation: Representing Model Uncertainty in Deep Learning"
    (https://arxiv.org/abs/1506.02142) to a 3D tensor.

    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps,
    embedding_dim)`` and samples a single dropout mask of shape ``(batch_size,
    embedding_dim)`` and applies it to every time step.
    """

    def forward(self, input_tensor):
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = F.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


