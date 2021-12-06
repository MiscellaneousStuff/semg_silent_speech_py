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
"""The library and base class for defining sEMG Silent Speech datasets.

This generic class is provided for other datasets to inherit from as future
datasets might have different characteristics to the Digital Voicing of Silent
Speech dataset.
"""

import torch
import abc

class sEMGUtterance(object):
    """Base utterance object which encapsulates a single utterance within
    an sEMG Silent Speech dataset. This is expected to contain vocalised
    and silent EMG data, audio data, the audio text and other related
    metadata. To define an utterance for a specific dataset, just subclass
    this.
    
    Attributes:
        vocal_emg_features: One or multiple EMG features (if applicable)
            for the utterance during the vocalised modality. Should be
            aligned with the audio.
        silent_emg_features: One or multiple EMG features (if applicable)
            for the utterance during the silent modality. Will not be
            aligned with the audio.
        text: Text representation of the audio data.
        audio_raw: Raw audio data.
        audio_features: MFCC, mel_spectogram or other audio features.
    
    May contain other attributes as well depending on the dataset. This is
    just the list of of minimal attributes per utterance.
    """

    @property
    def voiced_emg_features(self):
        return self._voiced_emg_features
    
    @property
    def silent_emg_features(self):
        return self._silent_emg_features

    @property
    def text(self):
        return self._text

    @property
    def audio_raw(self):
        return self._audio_raw

    @property
    def audio_features(self):
        return self._audio_features


class sEMGDataset(torch.utils.data.Dataset):
    """Base dataset object to configure an sEMG Silent Speech dataset. To
    define a dataset, just subclass this.

    Attributes:
        name: The name of the dataset.
        root_dir: The root directory of the dataset.
    """

    @property
    def name(self):
        return self.__class__.__name__
    
    """
    @property
    def num_features(self):
        return self.get_num_features()
    
    @property
    def num_speech_features(self):
        return self.get_num_speech_features()
    """
    
    """
    @abc.abstractmethod
    def get_num_speech_features(self):
        pass

    @abc.abstractmethod
    def get_num_features(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass
    """