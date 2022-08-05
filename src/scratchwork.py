
import os
import torch
import numpy as np

if __name__ == '__main__':
    
    folder = '../../scratchwork'
    for phase in ['train', 'dev']:
        lang = 'English'
        theirs = torch.load(os.path.join(folder, f'crosslingual_{lang}_{phase}.pt'))
        mine_to_theirs = lambda entry : { 'sent' : entry['sent'], 'labels' : entry['pos_labels'], 'lang' : entry['lang'] }
        mine_raw = torch.load(os.path.join(folder, f'mine_{lang}_{phase}.pt'))
        mine = list(map(mine_to_theirs, mine_raw))
        for entry1, entry2 in zip(theirs, mine):
            assert sorted(entry1.keys()) == sorted(entry2.keys())
            for k in entry1.keys():
                v1, v2 = entry1[k], entry2[k]
                if isinstance(v1, str): assert v1 == v2
                else: assert np.all(v1 == v2)
        print('done')