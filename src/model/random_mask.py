
import torch
import numpy as np

from model.single import Single
from constant import LABEL_PAD_ID
import util

class RandomMask(Single):
    def __init__(self, hparams):
        super(RandomMask, self).__init__(hparams)

    def mask_batch(self, batch):
        new_sent = batch['sent'].clone()
        for batch_index, (sub_sent, sub_labels, current_non_pad_length) in enumerate(zip(
                batch['sent'],
                batch['pos_labels'],
                batch['non_pad_length']
            )):
            assert len(sub_sent.shape) == 1 and len(sub_labels.shape) == 1
            raw_start_indices = torch.nonzero(sub_labels != LABEL_PAD_ID)
            assert raw_start_indices.shape[-1] == 1 and len(raw_start_indices.shape) == 2,\
                raw_start_indices.shape
            # the torch.nonzero will create an extra dimension of shape 1 at the end, remove it
            start_indices = raw_start_indices.reshape(-1)
            # Below: omit the [SEP] from being part of the mask for the last word.
            # the labels are padded for considering [CLS], [SEP].
            end_indices = util.apply_gpu(torch.Tensor([current_non_pad_length - 1]))
            if start_indices.shape[0] > 1:
                end_indices = torch.cat([start_indices[1:], end_indices])
            end_indices = end_indices.int()
            assert sub_labels.shape[0] == current_non_pad_length \
                or torch.all(sub_labels[current_non_pad_length:] == LABEL_PAD_ID) 
            # Below: conservative check
            assert torch.all(torch.take(sub_sent, start_indices) != self.tokenizer.pad_token_id)
            for start_index, end_index in zip(start_indices, end_indices):
                is_mask = np.random.uniform() < self.hparams.mask_probability
                if is_mask:
                    new_sent[batch_index][start_index:end_index] = self.tokenizer.mask_token_id
        new_batch = { k : v for k, v in batch.items() }
        new_batch['sent'] = new_sent
        return new_batch
            
        
    def __call__(self, batch):
        modified_batch = self.mask_batch(batch)
        return modified_batch
        #return super(RandomMask, self).__call__(modified_batch)


