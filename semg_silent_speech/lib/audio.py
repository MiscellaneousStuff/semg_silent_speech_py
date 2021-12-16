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
"""Utility functions for loading raw audio data into a suitable format for
the machine learning models."""

import soundfile as sf
import librosa
import numpy as np

def load_audio(fname,
    mel_spectograms,
    max_frames=None,
    sampling_rate=22_050,
    n_mel_channels=80,
    filter_length=1_024,
    win_length=1_024,
    hop_length=256,
    n_mfcc=26
    ):
    audio, r = sf.read(fname)
    assert r == 16_000 # Audio sampling rate expected to be 16kHz

    # Ensures single channel only
    if len(audio.shape) > 1:
        audio = audio[:,0]
    
    if not mel_spectograms:
        audio_features = librosa.feature.mfcc(
            audio,
            sr=sampling_rate,
            n_mfcc=n_mfcc,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length,
            center=False).T
    else:
        audio_features = librosa.feature.melspectrogram(
            audio, 
            sr=sampling_rate,
            n_mels=n_mel_channels,
            center=False,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length).T
        audio_features = np.log(audio_features + 1e-5)
    
    audio_discrete = librosa.core.mu_compress(audio, mu=255, quantize=True) + 128

    if max_frames is not None and audio_features.shape[0] > max_frames:
        audio_features = audio_features[:max_frames,:]
    
    audio_length = \
            hop_length * audio_features.shape[0] + \
            (win_length - hop_length)
    
    audio_discrete = audio_discrete[:audio_length]

    return audio_features.astype(np.float32), audio_discrete, audio