import torch
import os
import numpy as np
import random
import torch.utils.tensorboard as tb
from architectures.ccwavegan_gp import CCWaveGAN_GP
from models import ccwavegan_gen_xs, ccwavegan_dis_xs
from utils.utils import get_n_classes, create_date_folder, create_dataset, write_parameters

# Seed
seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

n_gpu = torch.cuda.device_count()
print("n_gpu: ", n_gpu)

if n_gpu == 0:
    print("Warning: There\'s no GPU available on this machine,"
            "training will be performed on CPU.")
    device = 'cpu'
else:
    print("Visible devices: ", os.environ["CUDA_VISIBLE_DEVICES"])
    device = 'cuda:0'
    torch.cuda.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Default tensor type set to torch.cuda.FloatTensor")

print("Device: ", device)

def train_model(
    sr=16000,
    n_batches=120001,
    batch_size=16,
    audio_path='audio/',
    checkpoints_path='checkpoints/',
    path_to_model='model.pth',
    resume_training=False,
    override_saved_model=False,
    synth_frequency=20000,
    n_synth_samples=10,
    save_frequency=40000,
    metrics_frequency=1000,
    n_synth_samples_metrics=50,
    latent_dim=100,
    upsample_mode='zeros',
    g_lr = 1e-4,
    d_lr = 1e-4,
    g_adam_b1=0.8,
    g_adam_b2=0.99,
    d_adam_b1=0.8,
    d_adam_b2=0.99,
    alpha_lrelu = 0.2,
    d_extra_steps = 5,
    verbose=False,
    device=device
):
    
    '''
    Train the conditional WaveGAN architecture.
    Args:
        sr (int):                   sampling rate (default: 16000)
        n_batches (int):            number of batches to train for.
                                    to save model up to last batch needs to end with 1 (e.g. 20001)
        batch_size (int):           batch size (for the training process).
        audio_path (str):           path to training data (wav files - one folder per class).
        checkpoints_path (str):     path to root folder where to save training session data (model, audio, tensorboard etc.)
        path_to_model (str):        path to save/resume the model
        resume_training (bool)      
        override_saved_model (bool)
        synth_frequency (int):      how often to synthesise samples during training (in batches).
        n_synth_samples (int):      number of samples to synthesise for each class during training
        save_frequency (int):       how often to save model during training (in batches).
        metrics_frequency (int):    how often to compute metrics (fad and mmd) during training (in batches)
        n_synth_samples_metrics(int):   number of samples per class used to compute metrics
        latent_dim (int):           dimension of the latent variable.
        upsample_mode (enum ['zeros', 'nn', 'linear', 'cubic']):    upsample using transposed convolution (zeros) or 
                                                                    interpolation (nn, linear, cubic) and convolution
        g_lr (float):               generator learning rate.
        d_lr (float):               discriminator learning rate.
        g_adam_b1 (float):          beta_1 parameter for adam generator optimizer.                       
        g_adam_b2 (float):          beta_2 parameter for adam generator optimizer.
        d_adam_b1 (float):          beta_1 parameter for adam discriminator optimizer.
        d_adam_b2 (float):          beta_2 parameter for adam discriminator optimizer.
        alpha_lrelu (float):        alpha parameter for discriminator leaky relu.
        d_extra_steps (int):        number of weights updates for discriminator for each generator updates
        verbose (bool):             prints tensor dimensions at each layer
        device (str):               cpu or gpu
    '''
    
    # get number of classes from the audio folder
    n_classes = get_n_classes(audio_path)
     
    # build generator and discriminator
    torch.manual_seed(seed)
    generator = ccwavegan_gen_xs.CCWaveGANGenerator(
                    latent_dim,
                    n_classes,
                    upsample_mode,
                    verbose
                )
    discriminator = ccwavegan_dis_xs.CCWaveGANDiscriminator(
                        alpha_lrelu,
                        n_classes,
                        verbose,
                    )

    # set the optimizers
    g_optimizer =   torch.optim.Adam(
                                generator.parameters(), 
                                lr = g_lr,
                                betas=[g_adam_b1, g_adam_b2]
                            )
    d_optimizer =   torch.optim.Adam(
                                    discriminator.parameters(), 
                                    lr = d_lr,
                                    betas=[d_adam_b1, d_adam_b2]
                                )
    # build the gan
    gan =   CCWaveGAN_GP(
                latent_dim,
                generator,
                discriminator,
                n_classes,
                g_optimizer,
                d_optimizer,
                device,
                d_extra_steps,
                latent_dim,
            )

    # make a folder with the current date to store the current session
    checkpoints_path = create_date_folder(checkpoints_path)

    # tensorboard
    d_writer = tb.SummaryWriter(log_dir=f'{checkpoints_path}/logdis')
    g_writer = tb.SummaryWriter(log_dir=f'{checkpoints_path}/loggen')
    
    # create dataset
    audio, labels = create_dataset(audio_path, sr, checkpoints_path)
    audio = torch.from_numpy(audio)
    labels = torch.from_numpy(labels)
    dataset_size = audio.shape[0]
    print('Dataset size: ', dataset_size)

    # load the desired weights (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_model}')
        checkpoint = torch.load(path_to_model)
        generator.load_state_dict(checkpoint['g_state_dict'])
        discriminator.load_state_dict(checkpoint['d_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

    #save training parameters to checkpoints folder
    params = {
        'sr': sr,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'audio_path': audio_path,
        'dataset_size': dataset_size,
        'path_to_model': path_to_model,
        'resume_training': resume_training,
        'override_saved_model': override_saved_model,
        'synth_frequency': synth_frequency,
        'n_synth_samples': n_synth_samples,
        'save_frequency': save_frequency,
        'metrics_frequency': metrics_frequency,
        'n_synth_samples_metrics': n_synth_samples_metrics,
        'latent_dim': latent_dim,
        'upsample_mode': upsample_mode,
        'alpha_lrelu': alpha_lrelu,
        'd_extra_steps': d_extra_steps,
        'device': device,
        'seed': seed,

        'generator_optimizer': g_optimizer,
        'discriminator_optimizer': d_optimizer,

        'generator': generator,
        'discriminator': discriminator
    }

    write_parameters(
        checkpoints_path,
        **params
    )
    
    gan.train(
        x = audio, 
        y = labels, 
        batch_size = batch_size, 
        n_batches = n_batches, 
        synth_frequency = synth_frequency, 
        n_synth_samples = n_synth_samples,
        save_frequency = save_frequency,
        metrics_frequency = metrics_frequency,
        n_synth_samples_metrics = n_synth_samples_metrics,
        sr = sr, 
        n_classes = n_classes,
        checkpoints_path = checkpoints_path, 
        override_saved_model = override_saved_model,
        writer = [d_writer, g_writer],
        audio_path=audio_path,
        verbose=verbose
    )


if __name__ == '__main__':
    train_model(sr = 16000,
                n_batches = 1,    # to test on CPU
                batch_size = 2,
                # n_batches = 120001,
                # batch_size = 16,
                audio_path = '../_footsteps_data/zapsplat_pack_footsteps_high_heels_1s_aligned/',
                checkpoints_path = 'checkpoints/',
                path_to_model = 'model.pth',
                resume_training = False,
                override_saved_model = False,
                synth_frequency = 20000,
                n_synth_samples = 10,
                save_frequency = 40000,
                metrics_frequency = 1000,
                n_synth_samples_metrics = 50,
                latent_dim = 100,
                upsample_mode = 'zeros',
                g_lr = 1e-4,
                d_lr = 1e-4,
                g_adam_b1 = 0.8,
                g_adam_b2 = 0.99,
                d_adam_b1 = 0.8,
                d_adam_b2 = 0.99,
                alpha_lrelu = 0.2,
                d_extra_steps = 5,
                verbose = False,
                device = device
    )