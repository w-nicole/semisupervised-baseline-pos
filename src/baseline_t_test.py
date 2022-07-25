
import pandas as pd
import scipy
import scipy.stats
import json
from collections import defaultdict

from pprint import pprint

if __name__ == '__main__':
    
    their_df = pd.read_csv('../crosslingual-nlp-private/experiments/seed_variation/val_accuracy_across_seeds.csv')
    tagger_df = pd.read_csv('./experiments/sweeps/seed/baseline/val_accuracy_across_seeds.csv')
    latent_df = pd.read_csv('./experiments/sweeps/seed/latent_base_64/val_accuracy_across_seeds.csv')
    
    extract_mean = lambda df : df[df.type.str.startswith('seed=')]
    languages = ['English', 'Dutch']
    
    results = defaultdict(dict)
    name_to_df = {
        'theirs' : their_df,
        'tagger' : tagger_df,
        'latent_64' : latent_df,
    }
    sorted_names = sorted(list(name_to_df.keys()))
    for idx1 in range(len(name_to_df)):
        for idx2 in range(idx1, len(name_to_df)):
            name1, name2 = sorted_names[idx1], sorted_names[idx2]
            df1, df2 = name_to_df[name1], name_to_df[name2]
            if name1 == name2: continue
            for language in languages:
                acc1 = extract_mean(df1)[language]
                acc2 = extract_mean(df2)[language]
                _, p_value = scipy.stats.ttest_rel(acc1, acc2)
                results[f'{name1}_to_{name2}'][language] = p_value
    
    with open('./experiments/sweeps/seed/t_test.json', 'w') as f:
        json.dump(results, f)
    pprint(results)
            