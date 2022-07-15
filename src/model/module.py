
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes: removed irrelevant code.
# Added all classes and related code except InputVariationalDropout

import torch
import torch.nn as nn
import torch.nn.functional as F
import constant
import util

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

class LSTMLinear(torch.nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, hidden_layers):
        super(LSTMLinear, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, hidden_layers, batch_first = True, bidirectional = True)
        self.linear = torch.nn.Linear(2 * hidden_size, output_size)
        
    def pack_padded_input(self, batch, padded_input):
        return util.apply_gpu(torch.nn.utils.rnn.pack_padded_sequence(padded_input, batch['length'].cpu(), batch_first=True, enforce_sorted=False))

    def forward(self, batch, lstm_input_raw):
        lstm_input = self.pack_padded_input(batch, lstm_input_raw)
        lstm_output_raw = self.lstm(lstm_input)
        lstm_output = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output_raw[0],
            batch_first = True, padding_value = constant.PACK_PADDING
        )
        model_output = self.linear(lstm_output[0])
        return model_output
        
        
