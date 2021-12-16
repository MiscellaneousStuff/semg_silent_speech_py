# MIT License
# 
# Copyright (c) 2021 Tada Makepeace
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Methods to plot different types of data used for sEMG Silent Speech datasets
and models."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

def plot_pred_y_emg_features(pred, y, epoch_idx):
    fig, ax = plt.subplots(2)

    plot_emg_features(pred, ax[0], "Prediction", epoch_idx)
    plot_emg_features(y, ax[1], "Actual", epoch_idx)

    return fig

def plot_emg_features(emg_features, ax, text, epoch_idx):
    epoch_txt = "" if epoch_idx==None else f", Epoch: {epoch_idx}"
    ax.set_title(f"EMG Feature: {text}{epoch_txt}")
    for i in range(emg_features.shape[1]):
        cur = np.squeeze(emg_features)[:,i::5]
        ax.plot(cur) # , label=electrode_labels[i])
    ax.legend()
    return ax

def plot_mel_spectrograms(pred, y, epoch_idx=None):
    fig, ax = plt.subplots(2)

    epoch_txt = "" if epoch_idx==None else f", Epoch: {epoch_idx}"

    ax[0].set_title(f"Mel Spectogram (Predicted){epoch_txt}")
    pred = np.swapaxes(pred, 0, 1)
    cax = ax[0].imshow(pred, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

    ax[1].set_title(f"Mel Spectogram (Actual){epoch_txt}")
    y = np.swapaxes(y, 0, 1)
    cax = ax[1].imshow(y, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

    return fig

def stack_emg_features(data):
    data = data.cpu().float().detach().numpy()

    # NOTE: These are the default audio features used in "Digital Voicing of Silent Speech"
    # Loop over each second of predicted `emg_features`
    new_data = data[0]
    for i in range(1, data.shape[0]):
        new_data = np.vstack((new_data, data[i]))
    
    return new_data

def stack_mel_spectogram(data):
    data = data.cpu().float().detach().numpy()

    # Loop over each second of `audio_features`
    new_data = data[0]
    for i in range(1, data.shape[0]):
        new_data = np.vstack((new_data, data[i]))
    
    return new_data