import scipy
import scipy.stats
import numpy as np

if __name__ == '__main__':
    
    ours_all = {
        'identity' : [73.48622131347656, 72.7745132446289, 73.09082794189453, 72.78581237792969, 73.11341857910156],
        'latent_64_english_only' : [72.76321411132812, 72.16448211669922, 72.66154479980469, 71.79168701171875, 73.30546569824219],
    }
    ours_key = 'latent_64_english_only'
    ours = np.array(ours_all[ours_key])
    # On the different checkpoint
    theirs = np.array([0.7314730882644653, 0.7315860986709595, 0.737799346446991, 0.7311342358589172, 0.7346362471580505]) * 100.0
    
    print(f'Ours: {ours_key}')
    print(f'\tMean: {ours.mean()}')
    print(f'\tStdev: {ours.std(ddof=1)}')
    print()
    print('Theirs')
    print(f'\tMean: {theirs.mean()}')
    print(f'\tStdev: {theirs.std(ddof=1)}')
    print()
    print(f'Difference (theirs - ours): {theirs.mean() - ours.mean()}')
    print()
    print(f'p-value: {scipy.stats.ttest_rel(ours, theirs)[1]}')
    

    