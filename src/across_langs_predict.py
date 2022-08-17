
import os
import glob

from model import Tagger
import predict.predict_across as predict_across

if __name__ == '__main__':
    
    phase = 'val'
    languages = ['English', 'Dutch', 'Turkish', 'Irish']
    folder = '../../alt/semisupervised-baseline-pos/experiments/subset/unmasked_alt_seed'
    is_masked = False

    hparams_template = os.path.join(folder, '*')
    hparams_path_matches = glob.glob(hparams_template)
    assert len(set(hparams_path_matches)) == len(hparams_path_matches), hparams_path_matches
    
    checkpoint_paths = predict_across.get_sweep_matches(folder, hparams_template)
    # Below is acceptable for dev.
    
    loading_model = util.get_full_set_model(Tagger, is_masked)
    dataloaders_dict = { util.get_full_set_dataloader(loading_model, lang, phase) for lang in languages}
    padded_labels_dict = { util.get_full_set_labels(loading_model, lang, phase) for lang in languages }
    
    for checkpoint_path in checkpoint_paths:
        df = predict_across.predict_over_languages(checkpoint_path, Tagger, phase, languages, dataloaders_dict, padded_labels_dict)
        
   