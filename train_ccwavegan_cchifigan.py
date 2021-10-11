import torch
import utils
import os
import itertools
import numpy as np
import random
import torch.utils.tensorboard as tb
from architectures import ccwavegan_cchifigan
from models import ccwavegan_gen_sm, ccwavegan_gen_xs2, hifi_ccmpd, hifi_ccmsd

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
    sampling_rate=22050,
    n_batches=10000,
    batch_size=128,
    audio_path='audio/',
    checkpoints_path='checkpoints/',
    architecture_size='large',
    resume_training=False,
    path_to_model='checkpoints/model.pth',
    override_saved_model=False,
    synth_frequency=200,
    n_synth_samples=10,
    save_frequency=200,
    loss_weight_frequency=200,
    n_loss_weight_synth_samples=10,
    latent_dim=100,
    use_batch_norm_gen=False,
    use_batch_norm_dis=False,
    upsample='zeros',
    generator_learning_rate=0.00004,
    generator_adam_b1=0.8,
    generator_adam_b2=0.99,
    discriminator_learning_rate=0.00004,
    discriminator_adam_b1=0.8,
    discriminator_adam_b2=0.99,
    verbose=False,
    device=device
):
    
    '''
    Train the conditional WaveGAN architecture.
    Args:
        sampling_rate (int): Sampling rate of the loaded/synthesised audio.
        n_batches (int): Number of batches to train for.
        batch_size (int): batch size (for the training process).
        audio_path (str): Path where your training data (wav files) are store. 
            Each class should be in a folder with the class name
        checkpoints_path (str): Path to save the model / synth the audio during training
        architecture_size (str) = size of the wavegan architecture. Eeach size processes the following number 
            of audio samples: 'small' = 16384, 'medium' = 32768, 'large' = 65536"
        resume_training (bool) = Restore the model weights from a previous session?
        path_to_weights (str) = Where the model weights are (when resuming training)
        override_saved_model (bool) = save the model overwriting 
            the previous saved model (in a past epoch)?. Be aware the saved files could be large!
        synth_frequency (int): How often do you want to synthesise a sample during training (in batches).
        save_frequency (int): How often do you want to save the model during training (in batches).
        latent_dim (int): Dimension of the latent space.
        use_batch_norm (bool): Use batch normalization?
        upsample (enum ['zeros', 'nn', 'linear', 'cubic']): upsample using transposed convolution (zeros) or interpolation (nn, linear, cubic) and convolution
        discriminator_learning_rate (float): Discriminator learning rate.
        generator_learning_rate (float): Generator learning rate.
        discriminator_extra_steps (int): How many steps the discriminator is trained per step of the generator.
        phaseshuffle_samples (int): Discriminator phase shuffle. 0 for no phases shuffle.
    '''
    
    # get the number of classes from the audio folder
    n_classes = utils.get_n_classes(audio_path)
     
    # build the generator
    torch.manual_seed(seed)
    if architecture_size == 'small':
        generator = ccwavegan_gen_sm.CCWaveGANGenerator(
                        z_dim=latent_dim,
                        n_classes=n_classes,
                        verbose=verbose
                    )
    elif architecture_size == 'extrasmall':
        generator = ccwavegan_gen_xs2.CCWaveGANGenerator(
                        z_dim=latent_dim,
                        n_classes=n_classes,
                        verbose=verbose
                    )

    # build the discriminator
    torch.manual_seed(seed)
    multiperiod_disc = hifi_ccmpd.CCMultiPeriodDiscriminator(
        architecture_size=architecture_size,
        n_classes=n_classes,
        verbose=verbose
    )
    multiscale_disc = hifi_ccmsd.CCMultiScaleDiscriminator(
        architecture_size=architecture_size,
        n_classes=n_classes,
        verbose=verbose
    )
    
    # set the optimizers
    # discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = discriminator_learning_rate)
    # generator_optimizer = torch.optim.Adam(generator.parameters(), lr = generator_learning_rate)
    generator_optimizer =   torch.optim.AdamW(
                                generator.parameters(), 
                                lr = generator_learning_rate, 
                                betas=[generator_adam_b1, generator_adam_b2]
                            )
    discriminator_optimizer =   torch.optim.AdamW(
                                    itertools.chain(multiperiod_disc.parameters(), multiscale_disc.parameters()),
                                    lr = discriminator_learning_rate, 
                                    betas=[discriminator_adam_b1, discriminator_adam_b2]
                                )
    
    # build the gan
    gan = ccwavegan_cchifigan.CCWaveGAN_CCHiFiGAN(
            latent_dim=latent_dim, 
            generator=generator,
            multiperiod_disc=multiperiod_disc,
            multiscale_disc=multiscale_disc,
            n_classes = n_classes, 
            d_optimizer = discriminator_optimizer,
            g_optimizer = generator_optimizer,
            device=device
        )

    # make a folder with the current date to store the current session
    checkpoints_path = utils.create_date_folder(checkpoints_path)

    # Tensorboard
    d_writer = tb.SummaryWriter(log_dir=f'{checkpoints_path}/logdis')
    g_writer = tb.SummaryWriter(log_dir=f'{checkpoints_path}/loggen')
    
    # create the dataset from the class folders in '/audio'
    audio, labels = utils.create_dataset(audio_path, sampling_rate, architecture_size, checkpoints_path)
    audio = torch.from_numpy(audio)
    labels = torch.from_numpy(labels)
    print('Dataset size: ', audio.shape[0])

    # load the desired weights in path (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_model}')
        checkpoint = torch.load(path_to_model)
        generator.load_state_dict(checkpoint['g_state_dict'])
        multiperiod_disc.load_state_dict(checkpoint['mpd_state_dict'])
        multiscale_disc.load_state_dict(checkpoint['msd_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

    #save the training parameters used to the checkpoints folder,
    #it makes it easier to retrieve the parameters/hyperparameters afterwards
    params = {
        'sampling_rate': sampling_rate,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'audio_path': audio_path,
        'dataset_size': audio.shape[0],
        'architecture_size': architecture_size,
        'resume_training': resume_training,
        'path_to_model': path_to_model,
        'override_saved_model': override_saved_model,
        'synth_frequency': synth_frequency,
        'n_synth_samples': n_synth_samples,
        'save_frequency': save_frequency,
        'loss_weight_frequency': loss_weight_frequency,
        'n_loss_weight_synth_samples': n_loss_weight_synth_samples,
        'latent_dim': latent_dim, 
        'use_batch_norm_gen': use_batch_norm_gen,
        'use_batch_norm_dis': use_batch_norm_dis,
        'upsample': upsample,
        'verbose': verbose,
        'device': device,
        'seed': seed,

        'generator_optimizer': generator_optimizer,
        'discriminator_optimizer': discriminator_optimizer,

        'generator': generator,
        'multiperiod_disc': multiperiod_disc,
        'multiscale_disc': multiscale_disc
    }

    utils.write_parameters(
        checkpoints_path,
        **params
    )
    
    #train the gan for the desired number of batches
    gan.train(
        x = audio, 
        y = labels, 
        batch_size = batch_size, 
        batches = n_batches, 
        synth_frequency = synth_frequency, 
        n_synth_samples = n_synth_samples,
        save_frequency = save_frequency,
        loss_weight_frequency = loss_weight_frequency,
        n_loss_weight_synth_samples = n_loss_weight_synth_samples,
        checkpoints_path = checkpoints_path, 
        override_saved_model = override_saved_model,
        sampling_rate = sampling_rate, 
        n_classes = n_classes,
        writer=[d_writer, g_writer],
        audio_path=audio_path,
        verbose=verbose
    )


if __name__ == '__main__':
    train_model(sampling_rate = 16000,
                n_batches = 1,
                batch_size = 2,
                # n_batches = 120001,
                # batch_size = 16,
                audio_path = '../_footsteps_data/zapsplat_pack_footsteps_high_heels_1s_aligned/',
                checkpoints_path = 'checkpoints/',
                architecture_size = 'extrasmall',
                path_to_model = 'model.pth',
                resume_training = False,
                override_saved_model = False,
                synth_frequency = 20000,
                n_synth_samples = 10,
                save_frequency = 40000,
                loss_weight_frequency=1000,
                n_loss_weight_synth_samples=50,
                latent_dim = 100,
                use_batch_norm_gen = True,
                use_batch_norm_dis = False,
                upsample='zeros',
                generator_learning_rate = 1e-4,
                discriminator_learning_rate = 1e-4,
                generator_adam_b1=0.8,
                generator_adam_b2=0.99,
                discriminator_adam_b1=0.8,
                discriminator_adam_b2=0.99,
                verbose=False,
                device=device
    )