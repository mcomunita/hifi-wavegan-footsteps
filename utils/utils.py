import torch
import math
import librosa
import numpy as np
import json
import os
import sys
from datetime import datetime


# WGANGP Utils

def update_optimizer_lr(optimizer, lr, decay):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * decay


def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


# HIFI-Gan utils

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


# Other Utils

# get number of classes from number of folders in the audio dir
def get_n_classes(audio_path):
    root, dirs, files = next(os.walk(audio_path))
    n_classes = len(dirs)
    print(f'Found {n_classes} different classes in {audio_path}')
    return n_classes

# load audio 
# pad if file is shorter than maximum architecture capacity
def load_audio(audio_path, sr, audio_size_samples):
    X_audio, _ = librosa.load(audio_path, sr = sr)
    if X_audio.size < audio_size_samples:
        padding = audio_size_samples - X_audio.size
        X_audio = np.pad(X_audio, (0, padding), mode = 'constant')
    elif (X_audio.size >= audio_size_samples):
        X_audio = X_audio[0:audio_size_samples]
    return X_audio

# save label names for inference
def save_label_names(audio_path, save_folder):
    label_names = {}
    for i, folder in enumerate(sorted(next(os.walk(audio_path))[1])):
        label_names[i] = folder
    # save dictionary to use it later with the standalone generator
    with open(os.path.join(save_folder, 'label_names.json'), 'w') as outfile:
        json.dump(label_names, outfile)
        
#create dataset from audio path folder
def create_dataset(audio_path, sample_rate, labels_saving_path):
    
    audio_size_samples = 8192
    
    #save label names in a dict
    save_label_names(audio_path, labels_saving_path)
    audio = []
    labels_names = []
    for folder in sorted(next(os.walk(audio_path))[1]):
        for wavfile in os.listdir(audio_path+folder):
            audio.append(load_audio(audio_path = f'{audio_path}{folder}/{wavfile}', sr = sample_rate, audio_size_samples = audio_size_samples))
            labels_names.append(folder)
    audio_np = np.asarray(audio)
    audio_np = np.expand_dims(audio_np, axis = 1)
    labels = np.unique(labels_names, return_inverse=True)[1]
    labels_np = np.expand_dims(labels, axis = 1)
    
    return audio_np, labels_np

#create folder with current date
def create_date_folder(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    date = datetime.now()
    day = date.strftime('%Y-%m-%d_')
    path = f'{checkpoints_path}{day}{str(date.hour)}h{str(date.minute)}m'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(f'{path}/synth_audio'):
        os.mkdir(f'{path}/synth_audio')     
    return path

#save training arguments
def write_parameters(checkpoints_path, **kwargs):
    print(f'\nSaving the training parameters to disk in {checkpoints_path}/training_parameters.txt')
    with open(f'{checkpoints_path}/training_parameters.txt', "w") as handle:
        for key, value in kwargs.items():
            if key in [
                'generator_optimizer', 'discriminator_optimizer', 
                'generator', 'discriminator'
            ]:
                handle.write("\n{}: {}\n" .format(key, value))
            else:
                handle.write("{}: {}\n" .format(key, value))

def GPU_is_available():
    cuda_available = torch.cuda.is_available()
    if not cuda_available: print("Cuda not available. Running on CPU")
    return cuda_available

def checkexists_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return False
    else:
        return True

def mkdir_in_path(path, dirname):
    dirpath = os.path.join(path, dirname)
    checkexists_mkdir(dirpath)
    return dirpath

def filter_files_in_path(dir_path, format='.wav'):
    return filter(lambda x: x.endswith(format), os.listdir(dir_path))

def list_files_abs_path(dir_path, format='.wav'):
    return [os.path.join(os.path.abspath(dir_path), x) for x in filter_files_in_path(dir_path, format)]

def list_files_in_subfolders(dir_path, format='.wav', verbose=False):
    files = []
    if verbose:
        print('Listing files in folders:')
    for folder in sorted(next(os.walk(dir_path))[1]):
        if verbose:
            print(os.path.join(os.path.abspath(dir_path), folder))
        files = files + \
                list_files_abs_path(
                    os.path.join(os.path.abspath(dir_path), folder), 
                    format
                )
    return files