import scipy
import scipy.stats
import numpy as np

if __name__ == '__main__':
    
    ours_all = {
        'identity' : [73.48622131347656, 72.7745132446289, 73.09082794189453, 72.78581237792969, 73.11341857910156],
        'latent_64_english_only' : [72.76321411132812, 72.16448211669922, 72.66154479980469, 71.79168701171875, 73.30546569824219],
        'latent_768_english_only' : [73.22638702392578, 73.52011108398438, 72.1305923461914, 73.01174926757812, 72.42430877685547],
        'fixed_latent_64_english_only' : [56.04383087158203, 56.24717712402344, 55.38861083984375, 57.07184982299805, 55.07229995727539],
        'fixed_latent_64_english_only_10' : [59.85087966918945, 58.51784896850586, 58.75508499145508, 59.98644256591797, 58.40488052368164],
    }
    ours_key = 'fixed_latent_64_english_only_10'
    ours = np.array(ours_all[ours_key])
    # On the different checkpoint
    theirs_all = {
        'non_best_composite' : [0.7314730882644653, 0.7315860986709595, 0.737799346446991, 0.7311342358589172, 0.7346362471580505],
        'fixed_best_val_acc_epoch' : [0.555467665195465, 0.513556241989136, 0.523271560668945, 0.546656131744385, 0.547107994556427],    
        'best_val_acc_epoch' : [0.734636247158051, 0.73316764831543, 0.737799346446991, 0.732376873493195, 0.738138258457184],
        'fixed_best_val_acc_epoch_10' : [0.583145022392273, 0.563827395439148, 0.575124263763428, 0.576931774616242, 0.582580208778381],
    }
    theirs_key = 'fixed_best_val_acc_epoch_10'
    theirs = np.array(theirs_all[theirs_key]) * 100.0
    
    print(f'Ours: {ours_key}')
    print(f'\tMean: {ours.mean()}')
    print(f'\tStdev: {ours.std(ddof=1)}')
    print()
    print(f'Theirs: {theirs_key}')
    print(f'\tMean: {theirs.mean()}')
    print(f'\tStdev: {theirs.std(ddof=1)}')
    print()
    print(f'Difference (theirs - ours): {theirs.mean() - ours.mean()}')
    print()
    print(f'p-value: {scipy.stats.ttest_rel(ours, theirs)[1]}')
    

    