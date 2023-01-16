import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import click
from scipy.special import logsumexp
from scipy.stats import norm

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
                                drop_last=False,
                                collate_fn = collate_fn_custom)

    test_data = CGMData(PATH, 'test', [len_seq, len_label, len_pred])
    test_data_loader = DataLoader(test_data, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=0, 
                                drop_last=False,
                                collate_fn = collate_fn_custom)

    return train_data_loader, test_data_loader

def build_model(model_path, device, d_model, n_heads, d_fcn, r_drop, activ, 
                    num_enc_layers, num_dec_layers, distil, len_seq, len_pred):
    model = Gluformer(d_model=d_model, 
                    n_heads=n_heads, 
                    d_fcn=d_fcn, 
                    r_drop=r_drop, 
                    activ=activ, 
                    num_enc_layers=num_enc_layers, 
                    num_dec_layers=num_dec_layers,
                    distil=distil, 
                    len_seq=len_seq,
                    len_pred=len_pred)
    model.load_state_dict(torch.load(model_path))
    model.train()
    model = model.to(device)

    return model

def calculate_rmse(pred, true, i, j):
    HYPO_THR = 70
    HYPER_THR = 180

    true = true[:, 0]; pred = np.mean(pred, axis=1)
    if j == "event":
        if np.any(true >= HYPER_THR) or np.any(true <= HYPO_THR):
            return [np.sqrt(np.mean((pred[:i] - true[:i])**2))]
        else:
            return []
    if j == "hypo":
        if np.any(true <= HYPO_THR):
            return [np.sqrt(np.mean((pred[:i] - true[:i])**2))]
        else:
            return []
    if j == "hyper":
        if np.any(true >= HYPER_THR):
            return [np.sqrt(np.mean((pred[:i] - true[:i])**2))]
        else:
            return []
    if j == "full":
        return [np.sqrt(np.mean((pred[:i] - true[:i])**2))]
    return []

def calculate_ape(pred, true, i, j):
    HYPO_THR = 70
    HYPER_THR = 180

    # median since ape corresponds to median
    true = true[:, 0]; pred = np.median(pred, axis=1)
    if j == "event":
        if np.any(true >= HYPER_THR) or np.any(true <= HYPO_THR):
            return [np.mean(np.abs(pred[:i] - true[:i]) / np.abs(true[:i]))]
        else:
            return []
    if j == "hypo":
        if np.any(true <= HYPO_THR):
            return [np.mean(np.abs(pred[:i] - true[:i]) / np.abs(true[:i]))]
        else:
            return []
    if j == "hyper":
        if np.any(true >= HYPER_THR):
            return [np.mean(np.abs(pred[:i] - true[:i]) / np.abs(true[:i]))]
        else:
            return []
    if j == "full":
        return [np.mean(np.abs(pred[:i] - true[:i]) / np.abs(true[:i]))]
    return []

@click.command()
@click.option('--trial_id', help='trial id')
@click.option('--gpu_index', default=0, help='index of gpu to use for training')
@click.option('--loss_name', default="mixture", help='name of loss to train model')
@click.option('--num_samples', default=1, help='number of samples from posterior')
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
def test(trial_id, loss_name,
                gpu_index, num_samples, len_pred, len_label, len_seq,
                d_model, n_heads, d_fcn, r_drop, activ,
                num_enc_layers, num_dec_layers, distil):
    # define consts -- experimental observations
    UPPER = 402
    LOWER = 38
    SCALE_1 = 5
    SCALE_2 = 2
    BATCH_SIZE=1

    # define paths
    if not os.path.exists(f'./trials/{trial_id}'):
        os.makedirs(f'./trials/{trial_id}')
    model_path = os.path.join(f'./trials/{trial_id}', "model_best.pth")

    # determine device type
    device = torch.device('cuda:'+str(gpu_index)) if torch.cuda.is_available() else torch.device('cpu')
    # load data
    if loss_name != "mixture":
        assert num_samples == 1
    train_data_loader, test_data_loader = load_data(num_samples, BATCH_SIZE, len_pred, len_label, len_seq)
    # define model
    model = build_model(model_path, device, d_model, n_heads, d_fcn, r_drop, activ, 
                    num_enc_layers, num_dec_layers, distil, len_seq, len_pred)
    if loss_name != "mixture":
        model.eval()
    else:
        model.train()

    horizons = [3, 6, 9, 12]
    events = ["event", "hypo", "hyper", "full"]
    ape = {i: {event: [] for event in events} for i in horizons}
    rmse = {i: {event: [] for event in events} for i in horizons}
    calibration = [[] for i in range(len_pred)]
    likelihoods = []

    # save predictions from 1 iteration for the following random samples
    SAMPLES = [1, 1230, 2340, 3001, 4001, 5001, 6540, 7001, 8012, 9200, 10980, 11012]
    curr_sample = 0
    save_pred_mean = np.empty((len(SAMPLES), len_pred, num_samples))
    save_pred_var = np.empty((len(SAMPLES), num_samples))
    save_true = np.empty((len(SAMPLES), len_pred, num_samples))
    save_train_true = {i: [] for i in SAMPLES}
    save_inp = np.empty((len(SAMPLES), len_seq))
    save_train_inp = {i: [] for i in SAMPLES}
    save_subj_ids_and_times = {}

    # extract subject ids and time stamps for each sample
    for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
         if i in SAMPLES:
            id = subj_id.detach().cpu().numpy()[0]
            time = batch_y_mark.detach().cpu().numpy()[0, :, 3:].reshape(-1).tolist() # hour and minute
            element = (id, *time)
            save_subj_ids_and_times[element] = i
    # extract training curves with same subject ids and time stamps
    for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_data_loader):
        id = subj_id.detach().cpu().numpy()[0]
        time = batch_y_mark.detach().cpu().numpy()[0, :, 3:].reshape(-1).tolist()
        element = (id, *time)
        if element in save_subj_ids_and_times.keys():
            save_train_inp[save_subj_ids_and_times[element]].append(batch_x.detach().cpu().numpy()[0, :, 0])
            save_train_true[save_subj_ids_and_times[element]].append(batch_y.detach().cpu().numpy()[0, -len_pred:, 0]) # cut to prediction length
    # reshape / rescale training curves
    for i in SAMPLES:
        save_train_inp[i] = (np.array(save_train_inp[i]) + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
        save_train_true[i] = (np.array(save_train_true[i]) + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER

    # iterate through test data
    # for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
    #     pred, true, logvar = process_batch(subj_id=subj_id, 
    #                                         batch_x=batch_x, 
    #                                         batch_y=batch_y, 
    #                                         batch_x_mark=batch_x_mark, 
    #                                         batch_y_mark=batch_y_mark, 
    #                                         len_pred=len_pred, 
    #                                         len_label=len_label, 
    #                                         model=model, 
    #                                         device=device)
    #     pred = pred.detach().cpu().numpy(); true = true.detach().cpu().numpy()
    #     logvar = logvar.detach().cpu().numpy(); batch_x = batch_x.detach().cpu().numpy()
    #     subj_id = subj_id.detach().cpu().numpy(); batch_x_mark = batch_x_mark.detach().cpu().numpy()

    #     # calculate log-likelihood
    #     likelihood = 0 
    #     if loss_name == "mixture":
    #         # reshape mean and true as (len_pred, num_samples)
    #         pred = pred.transpose((1,0,2)).reshape((pred.shape[1], -1, num_samples)).transpose((1, 0, 2))[0, :, :]
    #         true = true.transpose((1,0,2)).reshape((true.shape[1], -1, num_samples)).transpose((1, 0, 2))[0, :, :]
    #         # reshape logvar as (num_samples)
    #         logvar = logvar.squeeze()
    #         # calculate log-likelihood
    #         likelihood = -0.5 * pred.shape[0] * (np.log(2*np.pi) + logvar)
    #         likelihood += -0.5 * np.sum(np.square(pred - true), axis=0) * np.exp(-logvar)
    #         likelihood = logsumexp(likelihood) - np.log(num_samples)
    #     else:
    #         # reshape mean and true as (len_pred, num_samples)
    #         pred = pred[0, :, :]
    #         true = true[0, :, :]
    #         # calculate log-likelihood
    #         likelihood = np.mean((pred - true)**2)
    #     likelihoods.append(likelihood)

    #     # save samples + predictions for plotting 
    #     if i in SAMPLES:
    #         save_pred_mean[curr_sample, :, :] = (pred + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
    #         save_true[curr_sample, :, :] = (true + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
    #         save_inp[curr_sample, :] = (batch_x[0][:, 0] + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
    #         if loss_name == "mixture":
    #             scale = ((UPPER - LOWER) / (SCALE_1 * SCALE_2)) ** 2
    #             save_pred_var[curr_sample, :] = scale * np.exp(logvar)
    #         curr_sample = curr_sample + 1

    #     # transform data back
    #     pred = (pred + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
    #     true = (true + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
    #     # calculate ape / rmse for 3, 6, 9, 12 points AND full, event, hypo, hyper data
    #     for horizon in horizons:
    #         for event in ["full", "event", "hypo", "hyper"]:
    #             # calculate ape/rmse -> take median/mean over num_samples (inside function)+
    #             ape[horizon][event] += calculate_ape(pred, true, horizon, event)
    #             rmse[horizon][event] += calculate_rmse(pred, true, horizon, event)
        
    #     # calculate calibration and sharpness (full data) for mixture model
    #     if loss_name == "mixture":
    #         for i in range(len_pred):
    #             ps = [norm.cdf(true[i, 0], pred[i, j], np.sqrt(np.exp(logvar[j]))) 
    #                     for j in range(num_samples)]
    #             p = np.average(ps)
    #             calibration[i].append(p)
    
    # for horizon in horizons:
    #     for event in ["full", "event", "hypo", "hyper"]:
    #         ape_horizon_event = np.median(ape[horizon][event])
    #         rmse_horizon_event = np.median(rmse[horizon][event])
    #         print(f"APE for {event} {5*horizon} minutes: {ape_horizon_event:.6f}")
    #         print(f"RMSE for {event} {5*horizon} minutes: {rmse_horizon_event:.6f}")

    if not os.path.exists('./cache/visualize_glucose/'):
        os.makedirs('./cache/visualize_glucose/')
    if loss_name=="mixture":
        print("Log likelihood: {0}".format(np.sum(likelihoods)))
        print("Average log likelihood: {0}".format(np.mean(likelihoods)))
        np.save('./cache/visualize_glucose/input.npy', save_inp)
        np.save('./cache/visualize_glucose/true.npy', save_true)
        np.save('./cache/visualize_glucose/pred_mean_infmixt.npy', save_pred_mean)
        np.save('./cache/visualize_glucose/pred_var_infmixt.npy', save_pred_var)
        np.save('./cache/visualize_glucose/calibration_infmixt.npy', calibration)
        for i in save_train_inp.keys():
            np.save(f'./cache/visualize_glucose/train_input_{i}.npy', save_train_inp[i])
            np.save(f'./cache/visualize_glucose/train_true_{i}.npy', save_train_true[i])

    # else:
    #     varhat = np.mean(likelihoods)
    #     likelihood = -0.5*len_pred- 0.5*len_pred*np.log(2*np.pi*varhat)
    #     scale = ((UPPER - LOWER) / (SCALE_1 * SCALE_2)) ** 2
    #     varhat = scale * varhat
    #     print("Average log likelihood: {0}".format(likelihood))
    #     print("Variance MLE: {0}".format(varhat))
    #     np.save('./cache/visualize_glucose/input.npy', save_inp)
    #     np.save('./cache/visualize_glucose/true.npy', save_true)
    #     np.save('./cache/visualize_glucose/pred_mean_norm.npy', save_pred_mean)
    #     np.save('./cache/visualize_glucose/pred_var_norm.npy', np.array([varhat]))
    
if __name__ == '__main__':
    test()  