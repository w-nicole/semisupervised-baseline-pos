
from compare import significance
from pprint import pprint

if __name__ == '__main__':
    base_folder = "./experiments/debug/t_test/"
    name1 = 'dutch_mse_script'
    name2 = 'dutch_mse'
    lang = 'Dutch'
    
    seeds = [42, 0, 1, 2, 3]
    t_test = significance.compare_seed_sweeps(base_folder, name1, name2, seeds, lang)
    pprint(t_test)
    