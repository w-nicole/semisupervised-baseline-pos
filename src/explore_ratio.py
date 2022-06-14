
import os
import torch

from model import VAE
from enumeration import Split

if __name__ == '__main__':
    
    try:
        model_path = './experiments/decoder_baseline/corrected_one_kl'
        checkpoint_name = 'ckpts_epoch=7-val_acc=81.240.ckpt'
        checkpoint_path = os.path.join(model_path, 'ckpts', checkpoint_name)
        
        model = VAE.load_from_checkpoint(checkpoint_path)
        
        dutch_counts = model.get_label_counts('Dutch', Split.dev)
        english_counts = model.get_label_counts('English', Split.train)
    
        english_train = model.trn_datasets[0]    
        dutch_train = model.trn_datasets[1]
    
        assert dutch_train.lang == 'Dutch', dutch_train.lang
        assert english_train.lang == 'English', english_train.lang
        
        english_majority = torch.argmax(english_counts).item()
        majority_baseline_dutch = dutch_counts[english_majority] / torch.sum(dutch_counts)
        
        print('Ratio (dutch/english) train, in sentences: {len(dutch_train) / len(english_train)}')
        print(f'Majority baseline, dutch validation: {majority_baseline_dutch * 100}%')
    except: import pdb; pdb.set_trace()