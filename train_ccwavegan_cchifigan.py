import torch
import utils
import os
import itertools
import numpy as np
import random
import torch.utils.tensorboard as tb
from architectures.ccwavegan_cchifigan import CCWaveGAN_CCHiFiGAN
from models import ccwavegan_gen_xs, hifi_ccmpd, hifi_ccmsd
from models.ccwavegan_gen_xs import CCWaveGANGenerator
from models.hifi_ccmpd import CCMultiPeriodDiscriminator
from models.hifi_ccmsd import CCMultiScaleDiscriminator 
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
    verbose=False,
    device=device
):
    
    """
    Train Class-contidional WaveGAN architecture.
    Args:
        sr (int):                   sampling rate
        n_batches (int):            number of batches to train for.
                                    to save model up to last batch needs to end with 1 (e.g. 20001).
        batch_size (int):           batch size (for the training process).
        audio_path (str):           path to training data (wav files - one folder per class).
        checkpoints_path (str):     path to root folder where to save training session data (model, audio, tensorboard etc.).
        path_to_model (str):        path to save/resume the model.
        resume_training (bool)      
        override_saved_model (bool)
        synth_frequency (int):      how often to synthesise samples during training (in batches).
        n_synth_samples (int):      number of samples to synthesise for each class during training.
        save_frequency (int):       how often to save model during training (in batches).
        metrics_frequency (int):    how often to compute metrics (fad and mmd) during training (in batches).
        n_synth_samples_metrics(int):   number of samples per class used to compute metrics.
        latent_dim (int):           dimension of the latent variable.
        upsample_mode (enum ['zeros', 'nn', 'linear', 'cubic']):    upsample using transposed convolution (zeros) or 
                                                                    interpolation (nn, linear, cubic) and convolution.
        g_lr (float):               generator learning rate.
        d_lr (float):               discriminator learning rate.
        g_adam_b1 (float):          beta_1 parameter for adam generator optimizer.                       
        g_adam_b2 (float):          beta_2 parameter for adam generator optimizer.
        d_adam_b1 (float):          beta_1 parameter for adam discriminator optimizer.
        d_adam_b2 (float):          beta_2 parameter for adam discriminator optimizer.
        verbose (bool):             prints tensor dimensions at each layer.
        device (str):               cpu or gpu.
    """
    
    # get the number of classes from the audio folder
    n_classes = get_n_classes(audio_path)
     
    # build the generator
    torch.manual_seed(seed)
    generator = CCWaveGANGenerator(
                    latent_dim,
                    n_classes,
                    upsample_mode,
                    verbose
                )

    # build the discriminator
    torch.manual_seed(seed)
    multiperiod_disc =  CCMultiPeriodDiscriminator(
                            n_classes,
                            verbose
                        )
    multiscale_disc =   CCMultiScaleDiscriminator(
                            n_classes,
                            verbose
                        )
    
    # set the optimizers
    g_optimizer =   torch.optim.AdamW(
                                generator.parameters(), 
                                lr = g_lr, 
                                betas=[g_adam_b1, g_adam_b2]
                            )
    d_optimizer =   torch.optim.AdamW(
                                    itertools.chain(multiperiod_disc.parameters(), multiscale_disc.parameters()),
                                    lr = d_lr, 
                                    betas=[d_adam_b1, d_adam_b2]
                                )
    
    # build the gan
    gan =   CCWaveGAN_CCHiFiGAN(
                latent_dim,
                generator,
                multiperiod_disc,
                multiscale_disc,
                n_classes, 
                d_optimizer,
                g_optimizer,
                device
            )

    # make a folder with the current date to store the current session
    checkpoints_path = create_date_folder(checkpoints_path)

    # Tensorboard
    d_writer = tb.SummaryWriter(log_dir=f'{checkpoints_path}/logdis')
    g_writer = tb.SummaryWriter(log_dir=f'{checkpoints_path}/loggen')
    
    # create dataset
    audio, labels = create_dataset(audio_path, sr, checkpoints_path)
    audio = torch.from_numpy(audio)
    labels = torch.from_numpy(labels)
    dataset_size = audio.shape[0]
    print('Dataset size: ', dataset_size)

    # load the desired weights in path (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_model}')
        checkpoint = torch.load(path_to_model)
        generator.load_state_dict(checkpoint['g_state_dict'])
        multiperiod_disc.load_state_dict(checkpoint['mpd_state_dict'])
        multiscale_disc.load_state_dict(checkpoint['msd_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

    #save the training parameters used to the checkpoints folder,
    #it makes it easier to retrieve the parameters/hyperparameters afterwards
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
        'device': device,
        'seed': seed,

        'g_optimizer': g_optimizer,
        'd_optimizer': d_optimizer,

        'generator': generator,
        'multiperiod_disc': multiperiod_disc,
        'multiscale_disc': multiscale_disc
    }

    write_parameters(
        checkpoints_path,
        **params
    )
    
    #train the gan for the desired number of batches
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
    train_model(
        sr = 16000,
        n_batches = 1,      # to test on CPU
        batch_size = 2,     # to test on CPU
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
        metrics_frequency=1000,
        n_synth_samples_metrics=50,
        latent_dim = 100,
        upsample_mode='zeros',
        g_lr = 1e-4,
        d_lr = 1e-4,
        g_adam_b1=0.8,
        g_adam_b2=0.99,
        d_adam_b1=0.8,
        d_adam_b2=0.99,
        verbose=False,
        device=device
    )