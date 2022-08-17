
import os
import glob

from model import Tagger
import predict.predict_across as predict_across

if __name__ == '__main__':
    
    phase = 'val'
    languages = ['English', 'Dutch', 'Turkish', 'Irish']
    folder = '../../alt/semisupervised-baseline-pos/experiments/subset/unmasked_alt_seed'

    hparams_template = os.path.join(folder, '*')
    hparams_path_matches = glob.glob(hparams_template)
    assert len(set(hparams_path_matches)) == len(hparams_path_matches), hparams_path_matches
    
    for checkpoint_path in predict_across.get_sweep_matches(folder, hparams_template):
        df = predict_across.predict_over_languages(checkpoint_path, Tagger, phase, languages)
        
   