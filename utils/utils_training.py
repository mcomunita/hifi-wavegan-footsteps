import os
import sys
import torch
import torch.nn.functional as F
import soundfile
import subprocess
import numpy as np
import torchopenl3
import evaluation.utils as utils_eval
import csv
import pandas as pd
import random

from utils import GPU_is_available, mkdir_in_path, list_files_in_subfolders

def synth_samples_at_batch(
    generator,
    batch_number,
    checkpoints_path,
    output_dir,
    n_classes,
    n_samples_per_class,
    latent_dim,
    sr
):
    print(f'\nSynthesising audio at batch {batch_number}. Path: {checkpoints_path}/{output_dir}')
    for c in range(n_classes):
        z = torch.normal(mean=0.0, std=1.0, size=[n_samples_per_class, latent_dim])
        label_synth = torch.full((n_samples_per_class, 1), c)
        synth_audio = generator(z, label_synth)
        dirpath = mkdir_in_path(path=f'{checkpoints_path}', dirname=output_dir)
        dirpath = mkdir_in_path(path=dirpath, dirname=f'class_{c}')
        for s in range(n_samples_per_class):
            soundfile.write(
                file = f'{dirpath}/{batch_number}batch_c{c}_s{s}.wav',
                data = np.squeeze(synth_audio[s].detach().cpu().numpy()), 
                samplerate = sr
                )
    print(f'Done.')

def synth_fad_samples_at_batch(
    generator,
    batch_number,
    checkpoints_path,
    output_dir,
    n_classes,
    n_samples_per_class,
    latent_dim,
    sr
):
    print(f'\nSynthesising audio for FAD at batch {batch_number}. Path: {checkpoints_path}/{output_dir}')
    for c in range(n_classes):
        z = torch.normal(mean=0.0, std=1.0, size=[n_samples_per_class, latent_dim])
        label_synth = torch.full((n_samples_per_class, 1), c)
        synth_audio = generator(z, label_synth)
        if synth_audio.shape[2] < 16000: # VGGish needs at least 16000 samples input audio
            padding = 16000 - synth_audio.shape[2]
            synth_audio = F.pad(synth_audio, pad=(0, padding), mode='constant', value=0)
        dirpath = mkdir_in_path(path=f'{checkpoints_path}', dirname=output_dir)
        dirpath = mkdir_in_path(path=dirpath, dirname=f'class_{c}')
        for s in range(n_samples_per_class):
            soundfile.write(
                file = f'{dirpath}/{batch_number}batch_c{c}_s{s}.wav',
                data = np.squeeze(synth_audio[s].detach().cpu().numpy()), 
                samplerate = sr
                )
    print(f'Done.')

def synth_mmd_samples_at_batch(
    generator,
    batch_number,
    checkpoints_path,
    output_dir,
    n_classes,
    n_samples_per_class,
    latent_dim,
    sr
):
    print(f'\nSynthesising audio for MMD at batch {batch_number}. Path: {checkpoints_path}/{output_dir}')
    for c in range(n_classes):
        z = torch.normal(mean=0.0, std=1.0, size=[n_samples_per_class, latent_dim])
        label_synth = torch.full((n_samples_per_class, 1), c)
        synth_audio = generator(z, label_synth)
        # if synth_audio.shape[2] < 16000: # VGGish needs at least 16000 samples input audio
        #     padding = 16000 - synth_audio.shape[2]
        #     synth_audio = F.pad(synth_audio, pad=(0, padding), mode='constant', value=0)
        dirpath = mkdir_in_path(path=f'{checkpoints_path}', dirname=output_dir)
        dirpath = mkdir_in_path(path=dirpath, dirname=f'class_{c}')
        for s in range(n_samples_per_class):
            soundfile.write(
                file = f'{dirpath}/{batch_number}batch_c{c}_s{s}.wav',
                data = np.squeeze(synth_audio[s].detach().cpu().numpy()), 
                samplerate = sr
                )
    print(f'Done.')

def save_wavegan_at_batch(
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    g_loss,
    d_loss,
    batch_number,
    checkpoints_path,
    override_saved_model=False
):
    print(f'\nSaving the model at batch {batch_number}. Path: {checkpoints_path}')
    if override_saved_model == False:
        # save gan
        torch.save({
            'batch': batch_number,
            'g_state_dict': generator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_state_dict': discriminator.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            }, f'{checkpoints_path}/{batch_number}_batch_model.pth')
    else:
        torch.save({
            'batch': batch_number,
            'g_state_dict': generator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_state_dict': discriminator.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            }, f'{checkpoints_path}/model.pth')
    print(f'Model saved.')

def save_wavegan_hifigan_at_batch(
    generator,
    mpd,
    msd,
    g_optimizer,
    d_optimizer,
    g_loss,
    d_loss,
    batch_number,
    checkpoints_path,
    override_saved_model=False
):
    print(f'\nSaving the model at batch {batch_number}. Path: {checkpoints_path}')
    if override_saved_model == False:
        # save gan
        torch.save({
            'batch': batch_number,
            'g_state_dict': generator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'mpd_state_dict': mpd.state_dict(),
            'msd_state_dict': msd.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            }, f'{checkpoints_path}/{batch_number}_batch_model.pth')
    else:
        torch.save({
            'batch': batch_number,
            'g_state_dict': generator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'mpd_state_dict': mpd.state_dict(),
            'msd_state_dict': msd.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            }, f'{checkpoints_path}/model.pth')
    print(f'Model saved.')

def compute_fad_at_batch(
    real_samples_path,
    synth_samples_path,
    checkpoints_path,
    batch_number,
    verbose=False,
):
    if GPU_is_available:
        device = 'cuda'
    else:
        device = 'cpu'

    real_files = list_files_in_subfolders(real_samples_path, format='.wav', verbose=verbose)
    synth_files = list_files_in_subfolders(synth_samples_path, format='.wav', verbose=verbose)
    output_path = mkdir_in_path(path=checkpoints_path, dirname='fad')
    if verbose:
        print('fad output_path: ', output_path)
    
    # real_paths_csv = f"{output_path}/real_audio.csv"
    real_paths_csv = os.path.join(os.path.abspath(output_path), 'real_audio.csv')
    with open(real_paths_csv, "w") as f:
        for file_path in real_files:
            f.write(file_path + '\n')
    
    # synth_paths_csv = f"{output_path}/synth_audio.csv"
    synth_paths_csv = os.path.join(os.path.abspath(output_path), 'synth_audio.csv')
    with open(synth_paths_csv, "w") as f:
        for file_path in synth_files:
            f.write(file_path + '\n')
    
    output_path = os.path.abspath(output_path)

    # run shell script to compute fad
    # the script will - in turn - run the fad computation from the google_research_fad repo
    fad = float( \
            subprocess.check_output(
                [
                    "sh",
                    "shell_scripts/fad.sh",
                    "--real="+real_paths_csv,
                    "--synth="+synth_paths_csv,
                    "--output="+output_path
                ]).decode()[-10:-1]
            )

    # save result
    with open(f"{output_path}/fad_{len(synth_files)}.txt", "a") as f:
        f.write(f"batch: {str(batch_number)} - fad: {str(fad)}\n")
        f.close()
    
    print(f"FAD = {fad:.4f}")
    
    return fad

def compute_mmd_at_batch(
    real_samples_path,
    synth_samples_path,
    checkpoints_path,
    batch_number,
    verbose=False,
):
    if GPU_is_available:
        device = 'cuda'
    else:
        device = 'cpu'

    input_repr = 'mel128'   # "linear", "mel128" (default), "mel256"
    content_type = 'env'    # “env” (environmental), “music” (default)
    embedding_size = 512    # 512, 6144 (default)

    # get embeddings
    model = torchopenl3.models.load_audio_embedding_model(
                                input_repr=input_repr, 
                                content_type=content_type,
                                embedding_size=embedding_size
                                )
    openl3_embs_real = utils_eval.get_openl3_embs_subfolders(model, path=real_samples_path)
    openl3_embs_synth = utils_eval.get_openl3_embs_subfolders(model, path=synth_samples_path)

    output_path = mkdir_in_path(path=checkpoints_path, dirname='mmd')
    if verbose:
        print('mmd output_path: ', output_path)
    csv_filename_real = 'openl3_embeddings_real.csv'
    csv_filename_synth = 'openl3_embeddings_synth.csv'

    # write to csv
    embs_to_csv = []
    for row in openl3_embs_real:
        emb = [row[0], row[1]]
        for el in row[2]:
            emb.append(el)
        embs_to_csv.append(emb)

    fields = ['class', 'filename']
    for i in range(512):
        fields.append(f'openl3_{i}')

    with open(f'{output_path}/{csv_filename_real}', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(embs_to_csv)
    
    embs_to_csv = []
    for row in openl3_embs_synth:
        emb = [row[0], row[1]]
        for el in row[2]:
            emb.append(el)
        embs_to_csv.append(emb)

    fields = ['class', 'filename']
    for i in range(512):
        fields.append(f'openl3_{i}')

    with open(f'{output_path}/{csv_filename_synth}', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(embs_to_csv)
    
    df_real = pd.read_csv(f'{output_path}/{csv_filename_real}')
    df_synth = pd.read_csv(f'{output_path}/{csv_filename_synth}')
    feat_cols = [ 'openl3_'+str(i) for i in range(0,512) ]

    # MMD real vs synth
    n_mmds = 1000
    n_real_samples = df_real.shape[0]
    n_synth_samples = df_synth.shape[0]

    mmds = []

    for i in range(0, n_mmds):
        idx = sorted(random.sample(range(0, n_synth_samples-1), n_real_samples))
        df_synth_subset = df_synth.loc[idx]
        mmd = utils_eval.mmd(x=df_real[feat_cols].values, y=df_synth_subset[feat_cols].values, distance='euclidean')
        mmds.append(mmd)

    mmd_mean = np.average(mmds)
    mmd_std = np.std(mmds)

    # save result
    with open(f"{output_path}/mmd_{n_synth_samples}.txt", "a") as f:
        f.write(f"batch: {str(batch_number)} - mmd_mean: {str(mmd_mean)}, mmd_std: {str(mmd_std)}\n")
        f.close()
    
    print(f'MMD mean: {mmd_mean:.4f}')
    print(f'MMD std: {mmd_std:.4f}')
    
    return mmd_mean