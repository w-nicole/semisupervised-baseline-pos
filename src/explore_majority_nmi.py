
import torch
import torch.nn.functional as F

from model import Tagger
from metric import NMIMetric, LABEL_PAD_ID
from collections import defaultdict

import json
import os

if __name__ == '__main__':
    
    model = Tagger.load_from_checkpoint("./experiments/components/mbert_pretrained/encoder/linear/version_17h7usu3/ckpts/ckpts_epoch=2-val_English_acc_epoch_monitor=96.807.ckpt")
    
    try:
        results = defaultdict(dict)
        for lang in ['Dutch', 'English']:
            for split in ['dev', 'train']:
                labels = model.get_labels(lang, split)
                counts = torch.bincount(labels, minlength = model.nb_labels)
                majority_class = torch.argmax(counts)
                
                predict_majority_class = F.one_hot((torch.ones(labels.shape) * majority_class).long(), num_classes = model.nb_labels)
                # calculate the nmi for this majority class?
                
                metric = NMIMetric()
                metric.add(labels, predict_majority_class)
                nmi = metric.get_metric()
                
                results[lang][split] = {k : v.item() for k, v in nmi.items()}
                results[lang][split].update({'acc' : (counts[majority_class.item()] / (labels != LABEL_PAD_ID).sum()).item() })
        
        analysis_path = './experiments/dataset'
        if not os.path.exists(analysis_path): os.makedirs(analysis_path)
        with open(os.path.join(analysis_path, 'majority_baseline_nmi.json'), 'w') as f:
            json.dump(results, f)
    except: import pdb; pdb.set_trace()
        
                