
import os
from model import Single
from predict import across, predict_utils
import util

if __name__ == '__main__':
    
    phase = 'dev'
    languages = ['English', 'Dutch']
    
    loading_models = { is_masked : util.get_subset_model(Single, is_masked) for is_masked in [True, False] }
    dataloaders_dict, padded_labels_dict = {}, {}
    for is_masked, loading_model in loading_models.items():
        dataloaders_dict[is_masked] = {
            lang : util.get_subset_dataloader(loading_models[is_masked], lang, phase)
            for lang in languages
        } 
        padded_labels_dict[is_masked] = {
            lang : predict_utils.get_batch_padded_flat_labels(loading_models[is_masked], lang, phase)
            for lang in languages
        }
    
    base_folder = '../../alt/semisupervised-baseline-pos/experiments/self_train'
    checkpoint_paths = [
        'english/mixed/mixed/version_38g602fp/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=98.615.ckpt',
        'english/pure/pure/version_295owqwl/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=98.047.ckpt'
    ]
    for checkpoint_tail, is_masked in zip(checkpoint_paths, [False, True]):
        checkpoint_path = os.path.join(base_folder, checkpoint_tail)
        df = across.predict_over_languages(checkpoint_path, Single, phase, languages, dataloaders_dict[is_masked], padded_labels_dict[is_masked], 'flipped_true_labels')
            
   