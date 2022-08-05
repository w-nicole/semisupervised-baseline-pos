
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.
# Added NMI metric and imports, removed irrelevant ones and code.

from typing import Dict, List

import torch
import constant
import util

LABEL_PAD_ID = -1

def to_tensor(wrapped_func):
    def func(*args, **kwargs):
        result = wrapped_func(*args, **kwargs)
        return {k: torch.tensor(v, dtype=torch.float) for k, v in result.items()}
    return func


class Metric(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def unpack(*tensors: torch.Tensor):
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors) 

# Begin added metrics

class AverageMetric(Metric):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.number_of_tokens = 0
        self.total_metric = 0
        
    def add(self, averaged_value, batch_number_of_tokens):
        self.total_metric += averaged_value * batch_number_of_tokens
        self.number_of_tokens += batch_number_of_tokens
    
    @to_tensor
    def get_metric(self):
        if self.number_of_tokens == 0: return { self.metric_name : 0 }
        return { self.metric_name : self.total_metric / self.number_of_tokens }
        
    def reset(self):
        self.number_of_tokens = 0
        self.total_metric = 0

# End

# Added modifier
class POSMetric(Metric):
    def __init__(self, modifier):
        self.num_correct = 0
        self.num_tokens = 0
        self.modifier = modifier

    # Renamed prediction -> logits, removed comment
    def add(self, gold, logits):
        gold, logits = self.unpack(gold, logits)
        _, prediction = torch.max(logits, dim=-1)
        
        non_pad_mask = (gold != LABEL_PAD_ID)
        true_correct_mask = non_pad_mask & (prediction == gold)
        
        self.num_correct += true_correct_mask.sum().item()
        self.num_tokens += non_pad_mask.sum().item()
 

    @to_tensor
    def get_metric(self):
        try:
            acc = self.num_correct / self.num_tokens
        except ZeroDivisionError:
            acc = 0
        return {f"{self.modifier}_acc": acc * 100}

    def reset(self):
        self.num_correct = 0
        self.num_tokens = 0
