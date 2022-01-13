import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import time
import click

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gluformer.attention import *
from gluformer.encoder import *
from gluformer.decoder import *
from gluformer.embed import *
from gluformer.model import *

from gludata.data_loader import *
from utils.train import *


def load_data(num_samples, batch_size, len_pred, len_label, len_seq):
    # load data
    PATH = os.getcwd() + '/gludata/data/'

    # modify collate to repeat samples
    collate_fn_custom = modify_collate(num_samples)

    train_data = CGMData(PATH, 'train', [len_seq, len_label, len_pred])
    train_data_loader = DataLoader(train_data, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=0, 
                                drop_last=True, 
                                collate_fn = collate_fn_custom)

    val_data = CGMData(PATH, 'val', [len_seq, len_label, len_pred])
    val_data_loader = DataLoader(val_data, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=0, 
                                drop_last=True, 
                                collate_fn = collate_fn_custom)

    test_data = CGMData(PATH, 'test', [len_seq, len_label, len_pred])
    test_data_loader = DataLoader(test_data, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=0, 
                                drop_last=True,
                                collate_fn = collate_fn_custom)

    return train_data_loader, val_data_loader, test_data_loader

def build_model(device, d_model, n_heads, d_fcn, r_drop, activ, 
                    num_enc_layers, num_dec_layers, distil, len_pred):
    model = Gluformer(d_model=d_model, 
                    n_heads=n_heads, 
                    d_fcn=d_fcn, 
                    r_drop=r_drop, 
                    activ=activ, 
                    num_enc_layers=num_enc_layers, 
                    num_dec_layers=num_dec_layers,
                    distil=distil, 
                    len_pred=len_pred)
    model.train()
    model = model.to(device)

    return model

@click.command()
@click.option('--model_path', default="model_best.pth", help='save model here')
@click.option('--gpu_index', default=0, help='index of gpu to use for training')
@click.option('--loss_name', default="mixture", help='name of loss to train model')
@click.option('--num_samples', default=1, help='number of samples from posterior')
@click.option('--epochs', default=1, help='number of epochs')
@click.option('--stop_epochs', default=10, help='number of epochs for early stopping')
@click.option('--lrate', default=0.0002, help='learning rate of optimizer')
@click.option('--batch_size', default=32, help='batch size for SGD')
@click.option('--alpha', default=0, help='penalty for variance')
@click.option('--len_pred', default=12, help='length to predict')
@click.option('--len_label', default=60, help='length to feed to decoder')
@click.option('--len_seq', default=180, help='length of lookback')
@click.option('--d_model', default=512, help='model dimensions')
@click.option('--n_heads', default=12, help='number of attention heads')
@click.option('--d_fcn', default=2048, help='dimension of fully-connected layer')
@click.option('--r_drop', default=0.1, help='dropout rate')
@click.option('--activ', default="relu", help='activation function')
@click.option('--num_enc_layers', default=2, help='number of encoder layers')
@click.option('--num_dec_layers', default=1, help='number of decoder layers')
@click.option('--distil', default=True, help='distill or not between encoding')
def training(model_path, gpu_index, loss_name, num_samples, epochs, stop_epochs, lrate, batch_size, 
                alpha, len_pred, len_label, len_seq,
                d_model, n_heads, d_fcn, r_drop, activ,
                num_enc_layers, num_dec_layers, distil):
    # define consts -- experimental observations
    UPPER = 402
    LOWER = 38
    SCALE_1 = 5
    SCALE_2 = 2

    # determine device type
    device = torch.device('cuda:'+str(gpu_index)) if torch.cuda.is_available() else torch.device('cpu')

    # load data
    train_data_loader, val_data_loader, test_data_loader = load_data(num_samples, batch_size, len_pred, len_label, len_seq)
    # define model
    model = build_model(device, d_model, n_heads, d_fcn, r_drop, activ, 
                    num_enc_layers, num_dec_layers, distil, len_pred)

    # define loss and optimizer
    criterion = ""
    if loss_name == "mixture":
        criterion =  ExpLikeliLoss(num_samples=num_samples, alpha = alpha)
    else:
        criterion = nn.MSELoss()
    model_optim = torch.optim.Adam(model.parameters(), lr=lrate, betas=(0, 0.9))

    # define params for training
    TRAIN_STEPS = len(train_data_loader)
    early_stop = EarlyStop(stop_epochs, 0)

    for epoch in range(epochs):
        iter_count = 0
        train_loss = []
        
        epoch_time = time.time()
        curr_time = time.time()
        
        for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_data_loader):
            iter_count += 1
            # zero-out grad
            model_optim.zero_grad()
            pred, true, logvar = process_batch(subj_id = subj_id, 
                                    batch_x=batch_x, 
                                    batch_y=batch_y, 
                                    batch_x_mark=batch_x_mark, 
                                    batch_y_mark=batch_y_mark, 
                                    len_pred=len_pred, 
                                    len_label=len_label, 
                                    model=model, 
                                    device=device)
            loss = 0
            if loss_name == "mixture":
                loss = criterion(pred, true, logvar)
            else:
                loss = criterion(pred, true)
            train_loss.append(float(loss.item()))
            # print every 100
            if (i+1) % 100==0:
                print("\t iters: {0}, epoch: {1} | loss: {2:.7f} | variance: {3:.7f}".format(i + 1, epoch + 1, loss.item(), np.exp(logvar.detach().cpu().numpy())[0]))
                speed = (time.time() - curr_time) / iter_count
                left_time = speed * ((epochs - epoch) * TRAIN_STEPS - i)
                print('\t speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                curr_time = time.time()
            
            loss.backward()
            model_optim.step()
        # compute average train loss
        train_loss = np.average(train_loss)

        # compute validation / test loss + metric
        with torch.no_grad():
            val_loss = []
            for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_data_loader):
                pred, true, logvar = process_batch(subj_id = subj_id, 
                                        batch_x=batch_x, 
                                        batch_y=batch_y, 
                                        batch_x_mark=batch_x_mark, 
                                        batch_y_mark=batch_y_mark, 
                                        len_pred=len_pred, 
                                        len_label=len_label, 
                                        model=model, 
                                        device=device)
                if loss_name == "mixture":
                    loss = criterion(pred, true, logvar)
                else:
                    loss = criterion(pred, true)
                val_loss.append(float(loss.item()))
            val_loss = np.average(val_loss)

            test_metric = {3: [], 6: [], 9: [], 12:[]}; test_loss = []
            for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
                pred, true, logvar = process_batch(subj_id = subj_id, 
                                        batch_x=batch_x, 
                                        batch_y=batch_y, 
                                        batch_x_mark=batch_x_mark, 
                                        batch_y_mark=batch_y_mark, 
                                        len_pred=len_pred, 
                                        len_label=len_label, 
                                        model=model, 
                                        device=device)
                if loss_name == "mixture":
                    loss = criterion(pred, true, logvar)
                else:
                    loss = criterion(pred, true)
                test_loss.append(float(loss.item()))

                # compute metrix: APE
                pred = pred.detach().cpu().numpy(); true = true.detach().cpu().numpy()
                # transform back to data space
                pred = (pred + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
                true = (true + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
                # arrange in proper shape
                pred = pred.transpose((1,0,2)).reshape((pred.shape[1], -1, num_samples)).transpose((1, 0, 2))
                pred = np.mean(pred, axis=2)
                true = true.transpose((1,0,2)).reshape((true.shape[1], -1, num_samples)).transpose((1, 0, 2))[:, :, 0]
                # compute APE: 15 mins (3 points), 30 mins (6 points), 45 mins (9 points), full (12 points)
                for i in [3,6,9,12]:
                    test_metric[i].append(np.mean(np.abs(true[:, :i] - pred[:, :i]) / true[:, :i]))
            test_loss = np.average(test_loss)
            for i in [3,6,9,12]:
                    test_metric[i] = np.median(test_metric[i])
        
        # check early stopping
        early_stop(val_loss, model, model_path)
        if early_stop.stop:
            print("Early stopping...")
            break

        # update lr
        # adjust_learning_rate(model_optim, epoch, lr)
        
        print("Epoch: {0} Time: {1} Steps: {2}".format(epoch+1, time.time() - epoch_time, TRAIN_STEPS))
        print("Train Loss: {0:.7f} | Val Loss: {1:.7f} | Test Loss: {2:.7f}".format(train_loss, val_loss, test_loss))
        print("Test Loss (15 mins): {0:.7f} | (30 mins): {1:.7f} | (45 mins): {2:.7f} | (60 mins): {3:.7f}".format(
            test_metric[3], test_metric[6], test_metric[9], test_metric[12]))

if __name__ == '__main__':
    training()  
