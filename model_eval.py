import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import click
from scipy.special import logsumexp
from statsmodels.distributions.empirical_distribution import ECDF

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

    test_data = CGMData(PATH, 'test', [len_seq, len_label, len_pred])
    test_data_loader = DataLoader(test_data, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=0, 
                                drop_last=False,
                                collate_fn = collate_fn_custom)

    return test_data_loader

def build_model(model_path, device, d_model, n_heads, d_fcn, r_drop, activ, 
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

    true = true[:, 0]; pred = np.mean(pred, axis=1)
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

def plot_calibration(calibration):
    calibration_matrix = np.empty((11, 13))
    probs = np.linspace(0, 1, 11)
    for i in range(12):
        for j in range(len(probs)):
            calibration_matrix[j, i+1] = np.mean(np.array(calibration[i]) <= probs[j])
    calibration_matrix[:, 0] = probs
    calibration_data = pd.DataFrame(calibration_matrix)
    calibration_data.columns = ["Expected Confidence"] + [str(i * 5) + " minutes" for i in range(1, 13)]
    calibration_data = calibration_data.melt(id_vars=["Expected Confidence"], var_name="Time", value_name="Observed Confidence")

    sns.set_theme()
    sns.set_context("paper")
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(calibration_data, col="Time", hue="Time", palette="tab20c",
                        col_wrap=6, height=2)
    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "Expected Confidence", "Observed Confidence", marker="o")
    # Adjust the tick positions and labels
    grid.set(xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
            xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    # PLot diagonal 45 lines
    for ax in grid.axes.flat:
        x = np.linspace(0, 1, 11)
        y = x
        ax.plot(x, y, linestyle=':', color='gray')
    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)
    plt.savefig('calibration.pdf', dpi=300)

# def plot_prediction():
#     # define functions for plotting predictions
#     def subplt(fig, index, x, y, yhat):
#         ax = fig.add_subplot(2, 6, index)
#         quants = np.quantile(yhat[0, :, :], q=[0.025, 0.975], axis=1) 
#         mean = np.mean(yhat[0, :, :], axis=1)
#         ax.plot(range(1,13), y[0, :], label = "True")
#         ax.plot(range(1,13), mean, label = "Predicted")
#         ax.plot(range(-180, 0, 1), x, label = "Input")
#         if index > 6:
#             ax.set(xlabel="Time")
#         if index == 1 or index == 7:
#             ax.set(ylabel="Glucose (mg/dL)")
#         ax.fill_between(range(1,13), quants[0], quants[1], alpha=0.3, label = "95% CI")
#         ax.legend(loc='upper left')

#     plt.style.use("seaborn")
#     fig = plt.figure()
#     fig.set_size_inches(18, 6)
#     fig.subplots_adjust(hspace=0.2, wspace=0.4)
#     # plot selected samples 
#     sample = [1000, 3000, 4105, 
#             5000, 6000, 7000, 
#             7500, 9000, 10000,
#             11000, 12000, 12500]
#     for i in range(1, 13):
#         # subplt(fig, i, inp[sample[i-1]][1][0, :, 0], trues[sample[i-1]], preds[sample[i-1]])
#     # plt.tight_layout()
#     # plt.savefig('predictions.pdf', dpi=300)

def plot_sharpness(sharpness):
    sharpness_values = np.array([np.mean(sharpness[i]) for i in range(12)])
    ax = plt.figure()
    ax = sns.lineplot(x = range(1, 13), y = sharpness_values, marker="o")
    ax.set(xlabel="Time", ylabel="Variance")
    plt.savefig('sharpness.pdf', dpi=300)


@click.command()
@click.option('--model_path', default="./model_best.pth", help='path to the model dict file')
@click.option('--num_samples', default=1, help='number of samples from posterior')
@click.option('--variance', default=0.3, help='variance of normals')
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
def test(model_path, num_samples,
                variance, len_pred, len_label, len_seq,
                d_model, n_heads, d_fcn, r_drop, activ,
                num_enc_layers, num_dec_layers, distil):
    # define consts -- experimental observations
    UPPER = 402
    LOWER = 38
    SCALE_1 = 5
    SCALE_2 = 2
    BATCH_SIZE=1

    # determine device type
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load data
    test_data_loader = load_data(num_samples, BATCH_SIZE, len_pred, len_label, len_seq)
    # define model
    model = build_model(model_path, device, d_model, n_heads, d_fcn, r_drop, activ, 
                    num_enc_layers, num_dec_layers, distil, len_pred)

    ape = {3: {"hypo": [], "hyper": [], "event": [], "full": []},
        6: {"hypo": [], "hyper": [], "event": [], "full": []},
        9: {"hypo": [], "hyper": [], "event": [], "full": []},
        12: {"hypo": [], "hyper": [], "event": [], "full": []}}
    rmse = {3: {"hypo": [], "hyper": [], "event": [], "full": []},
        6: {"hypo": [], "hyper": [], "event": [], "full": []},
        9: {"hypo": [], "hyper": [], "event": [], "full": []},
        12: {"hypo": [], "hyper": [], "event": [], "full": []}}
    calibration = [[] for i in range(12)]; sharpness = [[] for i in range(12)]
    likelihoods = []

    for i, (subj_id, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
        pred, true = process_batch(subj_id=subj_id, 
                                    batch_x=batch_x, 
                                    batch_y=batch_y, 
                                    batch_x_mark=batch_x_mark, 
                                    batch_y_mark=batch_y_mark, 
                                    len_pred=len_pred, 
                                    len_label=len_label, 
                                    model=model, 
                                    device=device)
        pred = pred.detach().cpu().numpy(); true = true.detach().cpu().numpy()
        # reshape such that samples are last dim 
        pred = pred.transpose((1,0,2)).reshape((pred.shape[1], -1, num_samples)).transpose((1, 0, 2))[0, :, :]
        true = true.transpose((1,0,2)).reshape((true.shape[1], -1, num_samples)).transpose((1, 0, 2))[0, :, :]

        # calculate likelihood
        likelihood = logsumexp(np.sum(((-1 / (2 * variance)) * (pred - true)**2), axis=0)) + np.log(1/num_samples) - (num_samples / 2) * np.log(2*np.pi*variance)
        likelihoods.append(likelihood)

        # transform back
        pred = (pred + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER
        true = (true + SCALE_1) / (SCALE_1 * SCALE_2) * (UPPER - LOWER) + LOWER

        # calculate ape / rmse for 3, 6, 9, 12 points AND full, event, hypo, hyper data
        for i in [3,6,9,12]:
            for j in ["full", "event", "hypo", "hyper"]:
                ape[i][j] += calculate_ape(pred, true, i, j)
                rmse[i][j] += calculate_rmse(pred, true, i, j)
        
        # calculate calibration and sharpness (full data)
        for i in range(12):
            ecdf = ECDF(pred[i, :])
            p = ecdf(true[i, 0])
            calibration[i].append(p)
            sharpness[i].append(np.var(pred[i, :]))
    
    for i in [3,6,9,12]:
        for j in ["full", "event", "hypo", "hyper"]:
            print("APE for " + j + " " + str(i) + " :{0:.6f}".format(np.median(ape[i][j])))
            print("RMSE for " + j + " " + str(i) + " :{0:.6f}".format(np.median(rmse[i][j])))

    print("Log likelihood: {0}".format(np.sum(likelihoods)))
    print("Average log likelihood: {0}".format(np.mean(likelihoods)))

    # plot_calibration(calibration)
    # plot_sharpness(sharpness)
    # plot_prediction()

if __name__ == '__main__':
    test()  
        
        


    

