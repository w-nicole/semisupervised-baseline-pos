
import glob
from collections import defaultdict
import pprint

if __name__ == '__main__':
    repetitions = 7
    ckpt_paths = sorted(glob.glob('./experiments/baseline' + '/*' * repetitions + '/ckpts/*.ckpt'))
    start_modifier_index = 3; end_modifier_index = 8 # exclusive
    
    results = defaultdict(dict)
    for path in ckpt_paths:
        try:
            components = path.split('/')
            exp_name = '/'.join(components[start_modifier_index:end_modifier_index])
            lang = components[start_modifier_index]
            start_localizer = 'monitor='
            start_acc_index = path.index(start_localizer) + len(start_localizer)
            end_acc_index = path.index('.ckpt')
            acc = float(path[start_acc_index:end_acc_index])
            results[lang][exp_name] = acc
        except: import pdb; pdb.set_trace()
    pprint.pprint(results)