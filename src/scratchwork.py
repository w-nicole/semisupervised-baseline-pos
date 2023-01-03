
import os
import torch
import numpy as np

import util
from model import RandomMask

if __name__ == '__main__':
    
    # seed = 42
    # seed1 = np.random.RandomState(seed)
    # seed2 = np.random.RandomState(seed)
    
    # list1 = [1, 2, 3, 4, 5]
    # list2 = [5, 4, 3, 2, 1]
    # for _ in range(50):
    #     new_list1 = seed1.permutation(list1)
    #     new_list2 = seed2.permutation(list2)
    #     new_list3 = seed2.permutation(list2)
    #     print(new_list1)
    #     print(new_list2)
    #     print(new_list3)
    #     import pdb; pdb.set_trace()
    #     print()
    
    # folder = '../../scratchwork'
    # for phase, ratio in zip(['train', 'dev'], [0.01, 1]):
    # #for phase, ratio in zip(['dev'], [1]):
    #     lang = 'English'
    #     theirs = torch.load(os.path.join(folder, f'theirs_{lang}_{phase}_{ratio}.pt'))
    #     mine_to_theirs = lambda entry : { 'sent' : entry['sent'], 'labels' : entry['pos_labels'], 'lang' : entry['lang'] }
    #     mine_raw = torch.load(os.path.join(folder, f'ours_{lang}_{phase}_{ratio}.pt'))
    #     mine = list(map(mine_to_theirs, mine_raw))
    #     for index, (entry1, entry2) in enumerate(zip(theirs, mine)):
    #         assert sorted(entry1.keys()) == sorted(entry2.keys())
    #         for k in entry1.keys():
    #             v1, v2 = entry1[k], entry2[k]
    #             if isinstance(v1, str): assert v1 == v2
    #             else:
    #                 if not np.all(v1 == v2): import pdb; pdb.set_trace()
    #     print('done')
    
    # model = util.get_subset_model(RandomMask, False)
    # #val_dataloader = model.val_dataloader()[0]
    # # import pdb; pdb.set_trace()
    # #batch = next(val_dataloader.__iter__())
    
    # batch = {
    #     'sent' : torch.Tensor([
    #             [101, 1, 2, 3, 4, 5, 6, 102, 0, 0, 0],
    #             [101, 1, 2, 3, 102, 0, 0, 0, 0, 0, 0],
    #         ]),
    #     'pos_labels' : torch.Tensor([
    #             [-1, 8, -1, 3, 4, -1, -1, -1, -1, -1, -1],
    #             [-1, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1]
    #         ]),
    #     'lang' : ['English'] * 11,
    #     'non_pad_length' : [8, 5],
    #     'is_mask' : [
    #             [1, 0, 1],
    #             [1, 0, 1]
    #         ]
    # }
    
    # #try:
    # masked_batch = model.mask_batch(batch)
    # from pprint import pprint
    # import pdb; pdb.set_trace()
    
    
    
    