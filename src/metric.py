
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.
# Added NMI metric and imports, removed irrelevant ones and code.

from typing import Dict, List

import torch
import constant

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
        self.number_of_tokens = batch_number_of_tokens
    
    @to_tensor
    def get_metric(self):
        if self.number_of_tokens == 0: return { self.metric_name : 0 }
        return { self.metric_name : self.total_metric / self.number_of_tokens }
        
    def reset(self):
        self.number_of_tokens = 0
        self.total_metric = 0

class NMIMetric(Metric):
    def __init__(self):
        self.number_of_labels = len(constant.UD_POS_LABELS)
        self.predicted_by_label_counts = torch.zeros((self.number_of_labels, self.number_of_labels))

    def add(self, padded_labels, encoder_log_probs):
        # Convert to predictions
        to_token_dims = lambda tensor : tensor.reshape(-1, tensor.shape[-1])
        padded_labels, padded_log_probs = self.unpack(to_token_dims(padded_labels), to_token_dims(encoder_log_probs))
        padded_predictions = torch.argmax(padded_log_probs.log_softmax(dim=-1), dim=-1)

        # Cut out padding
        mask_for_non_pad = (padded_labels != LABEL_PAD_ID)
        labels = padded_labels[mask_for_non_pad]
        predictions = padded_predictions[mask_for_non_pad]
        
        assert labels.shape == predictions.shape and len(labels.shape) == 1,\
        f"labels: {labels.shape}, predictions: {predictions.shape}"
        for label in labels:
            for prediction in predictions:
                self.predicted_by_label_counts[label][prediction] += 1
        
    @to_tensor
    def get_metric(self):
        # Basic distributions
        number_of_tokens = self.predicted_by_label_counts.sum()
        joint_distribution = self.predicted_by_label_counts / number_of_tokens
        label_distribution = torch.sum(self.predicted_by_label_counts, axis = 1) / number_of_tokens
        predicted_distribution = torch.sum(self.predicted_by_label_counts, axis = 0) / number_of_tokens
        
        # log term
        repeated_label = label_distribution.unsqueeze(1).repeat(1, self.number_of_labels)
        repeated_predicted = predicted_distribution.unsqueeze(0).repeat(self.number_of_labels, 1)
        pre_sum = joint_distribution * torch.log(joint_distribution / (repeated_label * repeated_predicted))
        
        # Calculate NMI
        mi = pre_sum.sum()
        entropy = lambda distribution : -(distribution * torch.log(distribution)).sum()
        h_label = entropy(label_distribution)
        h_predicted = entropy(predicted_distribution)
        eps = 1e-10
        return {
            'mi' : mi,
            'nmi_added': 2 * mi / ( (h_label + h_predicted) + eps ),
            'nmi_min' : mi / ( min(h_label, h_predicted) + eps )
        }
        
    def reset(self):
        self.predicted_by_label_counts.fill_(0)
        
# End added metrics

class POSMetric(Metric):
    def __init__(self):
        self.num_correct = 0
        self.num_tokens = 0

    def add(self, gold, prediction):
        """
        gold is label
        prediction is logits
        """
        gold, prediction = self.unpack(gold, prediction)
        _, prediction = torch.max(prediction, dim=-1)
        bs, seq_len = prediction.shape
        for ii in range(bs):
            for jj in range(seq_len):
                gold_label, pred_label = gold[ii, jj], prediction[ii, jj]
                if gold_label == LABEL_PAD_ID:
                    continue
                if gold_label == pred_label:
                    self.num_correct += 1
                self.num_tokens += 1

    @to_tensor
    def get_metric(self):
        try:
            acc = self.num_correct / self.num_tokens
        except ZeroDivisionError:
            acc = 0
        return {"acc": acc * 100}

    def reset(self):
        self.num_correct = 0
        self.num_tokens = 0
