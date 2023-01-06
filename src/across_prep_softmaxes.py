
import os
from model import RandomMask, OnSplitEnsemble
from predict import across, predict_utils
import util

if __name__ == '__main__':
    
    phase = 'dev'
    languages = ['English', 'Dutch']
    
    loading_model = util.get_subset_model(RandomMask, 0)
    dataloaders_dict, padded_labels_dict = {}, {}
    dataloaders_dict = {
        lang : util.get_subset_dataloader(loading_model, lang, phase)
        for lang in languages
    } 
    padded_labels_dict = {
        lang : predict_utils.get_batch_padded_flat_labels(loading_model, lang, phase)
        for lang in languages
    }
    
    base_folder = '../../dev/semisupervised-baseline-pos/experiments/random_mask/ensemble'
    checkpoint_paths = [
        #'random_mask/ensemble/0/version_3g4ioy36/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=94.469.ckpt'
        '0.1/version_2dw7wfhz/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=95.048.ckpt',
        '0.25/version_3i2556bb/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=94.716.ckpt',
        '0.5/version_1ll320uw/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=93.921.ckpt',
        '0.75/version_avq8ul6l/ckpts/ckpts_epoch=2-val_English_pos_acc_epoch=90.525.ckpt',
        '0.9/version_1oxooja9/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=86.896.ckpt'
    ]
    for checkpoint_tail in checkpoint_paths:
        checkpoint_path = os.path.join(base_folder, checkpoint_tail)
        df = across.predict_over_languages(checkpoint_path, OnSplitEnsemble, phase, languages, dataloaders_dict, padded_labels_dict, 'unmasked_validation')
        
            
   