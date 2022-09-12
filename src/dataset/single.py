
from dataset.tagging import UdPOS

class SingleDataset(UdPOS):
    
    def load(self):
        assert self.data == []
        raw_full_examples = self.load_all_predictions()
        subset_indices = self.get_subset_indices(raw_full_examples)
        assert len(set(map(lambda entry : entry['ud_entry_index'], raw_full_examples))) == len(raw_full_examples)
        double_indexed_examples = {
            example['ud_entry_index'] : example
            for example in raw_full_examples
        }
        # Put in same shuffled order as the original codebase
        subset_examples = [
                double_indexed_examples[index]
                for index in subset_indices
            ]
        data = self.process_examples(subset_examples)
        self.data = data
        
        # import torch; import os
        # if not os.path.exists('../../scratchwork'): os.makedirs('../../scratchwork')
        # torch.save(data, f'../../scratchwork/ours_{self.lang}_{self.split}_{self.subset_ratio}.pt')