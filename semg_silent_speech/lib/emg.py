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
"""Utility functions for loading raw EMG data into a suitable format for
the machine learning models."""

import os
import numpy as np
import scipy

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)
    
def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def load_utterance(base_dir, index, limit_length=False, powerline_freq=60):
    idx = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{idx}_emg.npy'))
    before = os.path.join(base_dir, f'{idx-1}_emg.npy')
    after = os.path.join(base_dir, f'{idx+1}_emg.npy')

    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])
    
    emg = []
    for emg_channel in range(raw_emg.shape[1]):
        x = np.concatenate([raw_emg_before[:,emg_channel],
                            raw_emg[:,emg_channel],
                            raw_emg_after[:,emg_channel]])
        x = notch_harmonics(x, powerline_freq, 1000)
        x = remove_drift(x, 1000)
        x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0]]
        x = subsample(x, 600, 1000)
        x = subsample(x, 600, 1000)

        emg.append(x)
    
    emg = np.stack(emg, 1)
    return emg