
# Some code (initialization of SupervisedUdPOS, parts of load functions)
# taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

from dataset.tagging import UdPOS
from dataset.base import Tokenizer

class SupervisedUdPOS(UdPOS):
    
    def __init__(self, kwargs, use_rest_unsupervised):
        super(SupervisedUdPOS, self).__init__(**kwargs)
        self.use_rest_unsupervised = use_rest_unsupervised
        self.load()
        
    def load(self):
        assert self.data == []
        try:
            prefix, suffix = self.get_applied_if_split_data()
            data = self.process_with_supervised_flag(prefix, is_supervised=True)
            if self.use_rest_unsupervised:
                data += self.process_with_supervised_flag(suffix, is_supervised=False)
            self.data = data 
        except: import pdb; pdb.set_trace()
        
class UnsupervisedUdPOS(UdPOS):
    
    def __init__(self, kwargs):
        super(UnsupervisedUdPOS, self).__init__(**kwargs)
        self.load()
        
    def load(self):
        assert self.data == []
        examples, _ = self.get_applied_if_split_data()
        self.data = self.process_with_supervised_flag(examples, is_supervised=False)
        
        
    
    
    