
from model import LatentToPOS

if __name__ == '__main__':
    path = './experiments/debug/moving_target/lstm/pretrained/nll_kl/default/version_2kmukeqi/ckpts/ckpts_epoch=0-val_Dutch_acc_epoch=18.371.ckpt'
    model = LatentToPOS.load_from_checkpoint(path)
    
    print('Exploring path')
    print(path)
    
    val_dataloader = model.val_dataloader()[0]
    number_of_examples = 10
    for idx, batch in enumerate(val_dataloader):
        if idx == number_of_examples: break
        intermediates = model.calculate_intermediates(batch)
        names = ['encoder_hs', 'latent_sample', 'latent_mean', 'latent_sigma']
        for name, output in zip(names[:1], intermediates[:1]):
            print(f'___________ Example {idx}')
            print(f'\t{name}')
            print(f'\t\t{output}')
            import pdb; pdb.set_trace()
    
    
    