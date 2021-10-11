import torch
import os
import argparse
import torch.nn.functional as F
# import ipdb
import numpy as np
import logging
import math

from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from os.path import dirname, realpath, join
from datetime import datetime
# from gans.ac_criterion import ACGANCriterion
from sklearn.metrics import confusion_matrix, classification_report

from datetime import datetime

from pprint import pprint

import data.preprocessing as preprocessing
import data.audio_transforms as audio_transforms
import data.loaders as loaders
import utils.utils as utils
import evaluation.inception_network as inception_network

# from drumgan_evaluation.data.preprocessing import AudioProcessor
# from data.preprocessing import AudioProcessor
# from drumgan_evaluation.data.audio_transforms import MelScale
# from drumgan_evaluation.data.loaders import get_data_loader
# from drumgan_evaluation.utils.utils import mkdir_in_path, GPU_is_available
# from drumgan_evaluation.evaluation.inception_network import SpectrogramInception3
# from drumgan_evaluation.utils.utils import read_json


def train_inception_model(name: str, path: str, labels: list, config: str, batch_size: int=50, n_epoch=100):

    output_path = utils.mkdir_in_path(path, 'inception_models')
    output_file = join(output_path, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.pt")
    output_log = join(output_path, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")
    logging.basicConfig(filename=output_log, level=logging.INFO)

    assert os.path.exists(config), f"Path to config {config} does not exist"
    config = utils.read_json(config)

    loader_config = config['loader_config']
    # print("-- TRAIN INCEPTION MODEL: loader_config --")
    # print(loader_config)
    # print()
    
    transform_config = config['transform_config']
    # print("-- TRAIN INCEPTION MODEL: transform_config --")
    # print(transform_config)
    # print()
    
    transform = transform_config['transform']
    
    dbname = loader_config.pop('dbname')
    # print("-- TRAIN INCEPTION MODEL: dbname --")
    # print(dbname)
    # print()
    
    loader_module = loaders.get_data_loader(dbname)
    # print("-- TRAIN INCEPTION MODEL: loader_module --")
    # print(loader_module)
    # print()
    
    processor = preprocessing.AudioProcessor(**transform_config)
    
    loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)
    
    mel = audio_transforms.MelScale(sample_rate=transform_config['sample_rate'],
                   fft_size=transform_config['fft_size'],
                   n_mel=transform_config.get('n_mel', 256),
                   rm_dc=True)

    val_data, val_labels = loader.get_validation_set()
    val_data = val_data[:, 0:1]
    # print("-- TRAIN INCEPTION MODEL: val_labels --")
    # print(val_labels)
    # print()

    # att_dict = loader.header['attributes']
    # att_classes = att_dict.keys()

    # num_classes = sum(len(att_dict[k]['values']) for k in att_classes)
  
    data_loader = DataLoader(loader,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)
    
    num_classes = len(set(loader.metadata))
    tr_data_len = len(loader.data)
    val_data_len = len(loader.val_data)
    print("-- TR DATA --")
    print(len(loader.data))
    print()
    print("-- VAL DATA --")
    print(len(loader.val_data))
    print()
    print("-- NUM CLASSES --")
    print(num_classes)
    print()

    device = "cuda" if utils.GPU_is_available() else "cpu"

    inception_model = nn.DataParallel(
            inception_network.SpectrogramInception3(num_classes, aux_logits=False))
    # inception_model = SpectrogramInception3(num_classes, aux_logits=False)
    inception_model.to(device)
    # print('-- TRAIN INCEPTION MODEL: inception_model')
    # print(inception_model)

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, inception_model.parameters()),
                       betas=[0, 0.99], lr=0.001)

    # criterion = ACGANCriterion(att_dict)
    criterion = torch.nn.CrossEntropyLoss()
    
    epochs = trange(n_epoch, desc='train-loop') 

    vloss_best = math.inf

    for i in epochs:
        data_iter = iter(data_loader)
        iter_bar = trange(len(data_iter), desc='epoch-loop')
        inception_model.train()
        tr_loss = 0
        tr_correct = 0

        for j in iter_bar:
            input, labels = data_iter.next()
            # print('-- TRAIN INCEPTION MODEL')
            # print('input: ', input.shape)
            # print('labels: ', labels.shape)
            input.requires_grad = True
            input.to(device)

            # take magnitude
            input = mel(input.float())
            # print('input = mel(input.float())')
            
            mag_input = F.interpolate(input[:, 0:1], (299, 299))
            optim.zero_grad()
            
            output = inception_model(mag_input.float())
            # loss = criterion.getCriterion(output, target.to(device))
            
            loss = criterion(output, labels.to(device))
            tr_loss += loss
            correct = utils.get_num_correct_labels(output, labels.to(device))
            tr_correct += correct

            loss.backward()
            state_msg = f'Iter: {j}; loss: {loss.item():0.2f} '
            iter_bar.set_description(state_msg)
            optim.step()

        # # SAVE CHECK-POINT
        # if i % 10 == 0:
        #     if isinstance(inception_model, torch.nn.DataParallel):
        #         torch.save(inception_model.module.state_dict(), output_file)
        #     else:
        #         torch.save(inception_model.state_dict(), output_file)

        # EVALUATION
        with torch.no_grad():
            inception_model.eval()

            val_i = int(np.ceil(len(val_data) / batch_size))
            vloss = 0
            val_correct = 0
            # prec = 0
            y_pred = []
            y_true = []
            # prec = {k: 0 for k in att_classes}

            for k in range(val_i):
                vlabels = val_labels[k*batch_size:batch_size * (k+1)]
                vdata = val_data[k*batch_size:batch_size * (k+1)]
                vdata = mel(vdata.float())
                vdata = F.interpolate(vdata, (299, 299))

                vpred = inception_model(vdata.to(device))
                # vloss += criterion.getCriterion(vpred, vlabels.to(device)).item()
                
                loss = criterion(vpred, vlabels.to(device)).item()
                vloss += loss
                correct = utils.get_num_correct_labels(vpred, vlabels.to(device))
                val_correct += correct

                # vlabels_pred, _ = criterion.getPredictionLabels(vpred)
                # y_pred.append(vlabels_pred)
                y_pred.append(vlabels)
                # y_true += list(vlabels)

            y_pred = torch.cat(y_pred)

            # pred_labels = loader.index_to_labels(y_pred)
            # true_labels = loader.index_to_labels(val_labels)
            # for i, c in enumerate(att_classes):
            #     # if class is xentroopy...
            #     if att_dict[c]['loss'] == 'mse': continue
            #     logging.info(c)
            #     pred = [l[i] for l in pred_labels]
            #     true = [l[i] for l in true_labels]
            #     cm = confusion_matrix(
            #         true, pred,
            #         labels=att_dict[c]['values'])
            #     print("")
            #     print("Confusion Matrix")
            #     print(cm)
            #     logging.info(cm)
            #     print("")
            #     target_names = [str(v) for v in att_dict[c]['values']]
            #     crep = classification_report(true, pred, target_names=target_names, labels=target_names)
            #     logging.info(crep)
            #     print(crep)
        
        tr_accuracy = tr_correct / tr_data_len
        val_accuracy = val_correct / val_data_len
        state_msg2 = f'epoch {i}; tr_loss (scaled x1000): {(tr_loss / tr_data_len)*1000: 0.2f}; tr_accuracy: {tr_accuracy:.2f}; val_loss: {(vloss / val_data_len)*1000:.2f}; val_accuracy: {val_accuracy:.2f}'
        logging.info(state_msg2)
        epochs.set_description(state_msg2)

        # SAVE BEST
        if vloss < vloss_best:
            vloss_best = vloss
            if isinstance(inception_model, torch.nn.DataParallel):
                torch.save(inception_model.module.state_dict(), output_file)
            else:
                torch.save(inception_model.state_dict(), output_file)
            logging.info('SAVED BEST')
        
        # EARLY STOP
        if val_accuracy >= 0.9:
            logging.info('EARLY STOP')
            break



if __name__=='__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--name', dest='name', type=str, default="default_inception_model",
                     help="Name of the output inception model")

    argparser.add_argument('-c', '--config', dest='config', type=str, default="default_inception_model",
                     help="Name of the output inception model")

    argparser.add_argument('-p', '--path', dest='path', type=str,
                     default=dirname(realpath(__file__)))
    argparser.add_argument('--batch-size', dest='batch_size', type=int, default=100,
                     help="Name of the output inception model")
    argparser.add_argument('-l', '--labels', dest='labels', nargs='+', help='Labels to train on')
    argparser.add_argument('-e', '--epochs', dest='n_epoch', type=int, default=100,
                           help='Labels to train on')

    args = argparser.parse_args()

    train_inception_model(**vars(args))

