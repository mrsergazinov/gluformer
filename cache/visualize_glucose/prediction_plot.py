import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


def subplt(fig, index, x, y, yhat, varhat=-1):
    HISTORY = 20
    ax = fig.add_subplot(2, 6, index)
    mean = np.mean(yhat, axis=1)
    quants = 0
    ax.plot(range(1,13), y[:, 0], label = "Observed")
    ax.plot(range(1,13), mean, label = "Predicted")
    ax.plot(range(-HISTORY, 0, 1), x[-HISTORY:], label = "Input")
    if varhat == -1:
        quants = np.quantile(yhat, q=[0.025, 0.975], axis=1)
        ax.fill_between(range(1,13), quants[0], quants[1], alpha=0.3, label = "95% CI")
    else:
        quants = [mean - 2*np.sqrt(varhat), mean + 2*np.sqrt(varhat)]
        ax.fill_between(range(1,13), quants[0], quants[1], alpha=0.3, color='darkorange', label = "95% CI")
    if index == 1 or index == 7:
        ax.set(ylabel="Glucose (mg/dL)")

if __name__ == '__main__':
    inp = np.load('./input.npy')
    true = np.load('./true.npy')
    pred_wo = np.load('./pred_wo.npy')
    pred_w = np.load('./pred_w.npy') 
    varhat = 706.8018483386448 # MLE estimate of variance

    plt.style.use("seaborn")
    params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

    fig = plt.figure()
    fig.set_size_inches(18, 6)
    # plot selected samples
    samples=[0, 1, 3, 5, 8, 10]*2
    for i in range(len(samples)):
        if i > 5:
            subplt(fig, i+1, inp[samples[i]], true[samples[i]], pred_w[samples[i]])
        else:
            subplt(fig, i+1, inp[samples[i]], true[samples[i]], pred_wo[samples[i]], varhat)

    fig.subplots_adjust(hspace=0.2, wspace=0.4, bottom=0.12)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    y_box, y_label = fig.axes[0].get_legend_handles_labels()
    y_box = [y_box[3]]; y_label = [y_label[3]]
    fig.legend(lines+y_box, labels+y_label, loc = "lower center", ncol = 5)
    
    plt.savefig('./predictions_combined.pdf', dpi=300)