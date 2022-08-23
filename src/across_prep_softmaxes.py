
import os
import glob

from model import Tagger
import predict.predict_across as predict_across
import predict.predict_utils as predict_utils
import util

if __name__ == '__main__':
    
    phase = 'dev'
    languages = ['English', 'Dutch']
    
    checkpoint_path = ""
    model = Tagger.load_from_checkpoint(components['checkpoint_path'])
    dataloader = util.get_subset_dataloader(loading_model[components['masked']], self.lang, self.split)
    
    
    loading_model = util.get_full_set_model(Tagger, is_masked)
    dataloaders_dict = { lang : util.get_full_set_dataloader(loading_model, lang, phase) for lang in languages }
    padded_labels_dict = { lang : predict_utils.get_batch_padded_flat_labels(loading_model, lang, phase) for lang in languages }
        
    for sweep_name, is_masked in zip(['masked', 'unmasked', 'unmasked_alt_seed'], [True, False, False]):
    
        folder = os.path.join(folder_base, sweep_name)
        hparams_template = os.path.join(folder, '*')
        hparams_path_matches = glob.glob(hparams_template)
        assert len(set(hparams_path_matches)) == len(hparams_path_matches), hparams_path_matches
        
        checkpoint_paths = predict_across.get_sweep_matches(folder, hparams_template)
        # Below is acceptable for dev.
        
        for checkpoint_path in checkpoint_paths:
            df = predict_across.predict_over_languages(checkpoint_path, Tagger, phase, languages, dataloaders_dict, padded_labels_dict)
        
   