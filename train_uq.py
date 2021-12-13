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
from utils.test import *


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
@click.option('--num_samples', default=1, help='number of samples from posterior')
@click.option('--epochs', default=1, help='number of epochs')
@click.option('--batch_size', default=32, help='batch size for SGD')
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
def training(num_samples, epochs, batch_size, len_pred, len_label, len_seq,
                d_model, n_heads, d_fcn, r_drop, activ,
                num_enc_layers, num_dec_layers, distil):
    # define consts -- experimental observations
    UPPER = 402
    LOWER = 38
    SCALE_1 = 5
    SCALE_2 = 2
    # determine device type
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load data
    train_data_loader, val_data_loader, test_data_loader = load_data(num_samples, batch_size, len_pred, len_label, len_seq)
    # define model
    model = build_model(device, d_model, n_heads, d_fcn, r_drop, activ, 
                    num_enc_layers, num_dec_layers, distil, len_pred)

    # define loss and optimizer
    lr = 0.0002
    criterion =  ExpLikeliLoss(num_samples=num_samples)
    model_optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.9))

    # define params for training
    PATH_MODEL = os.getcwd() + "/model_best.pth"
    TRAIN_STEPS = len(train_data_loader)
    early_stop = EarlyStop(20, 0)

    for epoch in range(epochs):
        iter_count = 0
        train_loss = []
        
        epoch_time = time.time()
        curr_time = time.time()
        
        for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_data_loader):
            iter_count += 1
            # zero-out grad
            model_optim.zero_grad()

            pred, true = process_batch(subj_id = subj_id, 
                                    batch_x=batch_x, 
                                    batch_y=batch_y, 
                                    batch_x_mark=batch_x_mark, 
                                    batch_y_mark=batch_y_mark, 
                                    len_pred=len_pred, 
                                    len_label=len_label, 
                                    model=model, 
                                    device=device)
            loss = criterion(pred, true)
            train_loss.append(loss.item())

            # print every 100
            if (i+1) % 100==0:
                print("\t iters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - curr_time) / iter_count
                left_time = speed * ((EPOCHS - epoch) * TRAIN_STEPS - i)
                print('\t speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                curr_time = time.time()
            
            loss.backward()
            model_optim.step()

        # compute average train loss
        train_loss = np.average(train_loss)

        # compute validation loss
        val_loss = []
        for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_data_loader):
            pred, true = process_batch(subj_id = subj_id, 
                                    batch_x=batch_x, 
                                    batch_y=batch_y, 
                                    batch_x_mark=batch_x_mark, 
                                    batch_y_mark=batch_y_mark, 
                                    len_pred=len_pred, 
                                    len_label=len_label, 
                                    model=model, 
                                    device=device)
            pred = pred.detach().cpu().numpy(); true = true.detach().cpu().numpy()
            # transform back
            pred = (pred + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
            true = (true + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER

            pred = pred.transpose((1,0,2)).reshape((pred.shape[1], -1, num_samples)).transpose((1, 0, 2))
            pred = np.mean(pred, axis=2)
            true = true.transpose((1,0,2)).reshape((true.shape[1], -1, num_samples)).transpose((1, 0, 2))[:, :, 0]
            # compute APE
            ape_val = np.mean(np.abs(true - pred) / true)
            val_loss.append(ape_val)
        val_loss = np.median(np.array(val_loss))
        
        # compute test loss
        test_loss_3 = []; test_loss_6 = []; test_loss_9 = []; test_loss_12 = []
        for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
            pred, true = process_batch(subj_id = subj_id, 
                                    batch_x=batch_x, 
                                    batch_y=batch_y, 
                                    batch_x_mark=batch_x_mark, 
                                    batch_y_mark=batch_y_mark, 
                                    len_pred=len_pred, 
                                    len_label=len_label, 
                                    model=model, 
                                    device=device)
            pred = pred.detach().cpu().numpy(); true = true.detach().cpu().numpy()
            # transform back
            pred = (pred + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
            true = (true + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER

            pred = pred.transpose((1,0,2)).reshape((pred.shape[1], -1, num_samples)).transpose((1, 0, 2))
            pred = np.mean(pred, axis=2)
            true = true.transpose((1,0,2)).reshape((true.shape[1], -1, num_samples)).transpose((1, 0, 2))[:, :, 0]
            # compute APE: 15 mins (3 points)
            ape_3 = np.mean(np.abs(true[:, :3] - pred[:, :3]) / true[:, :3])
            # compute APE: 30 mins (6 points)
            ape_6 = np.mean(np.abs(true[:, :6] - pred[:, :6]) / true[:, :6])
            # compute APE: 45 mins (9 points)
            ape_9 = np.mean(np.abs(true[:, :9] - pred[:, :9]) / true[:, :9])
            # compute APE: full
            ape_12 = np.mean(np.abs(true - pred) / true)

            test_loss_3.append(ape_3)
            test_loss_6.append(ape_6)
            test_loss_9.append(ape_9)
            test_loss_12.append(ape_12)
        test_loss_3 = np.median(np.array(test_loss_3))
        test_loss_6 = np.median(np.array(test_loss_6))
        test_loss_9 = np.median(np.array(test_loss_9))
        test_loss_12 = np.median(np.array(test_loss_12))
        
        # check early stopping
        early_stop(val_loss, model, PATH_MODEL)
        if early_stop.stop:
            print("Early stopping...")
            break

        # update lr
        # adjust_learning_rate(model_optim, epoch, lr)
        
        print("Epoch: {} cost time: {}".format(epoch+1, time.time() - epoch_time))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f}".format(
            epoch + 1, TRAIN_STEPS, train_loss, val_loss))
        print("Test loss for 15 mins: {0:.7f}, for 30 mins: {1:.7f}, for 45 mins: : {2:.7f}, for 60 mins: {3:.7f}".format(
            test_loss_3, test_loss_6, test_loss_9, test_loss_12))

if __name__ == '__main__':
    training()  