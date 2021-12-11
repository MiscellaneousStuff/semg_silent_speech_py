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
"""Define the Digital Voicing of Silent Speech dataset."""

import os
import torch
import json
import numpy as np
import librosa

from semg_silent_speech.datasets  import lib
from semg_silent_speech.lib.emg   import load_utterance
from semg_silent_speech.lib.audio import load_audio
from semg_silent_speech.lib.utils import double_average, split_fixed_length

electrode_labels = [
    "Cheek above mouth",
    "Chin",
    "Below chin",
    "Throat",
    "Mid-jaw",
    "Cheek below mouth",
    "High cheek",
    "Back of cheek"
]

class DigitalVoicingUtterance(lib.sEMGUtterance):
    """Encapsulation of an sEMG Silent Speech utterance within the Digital
    Voicing dataset by David Gaddy and Dan Klein."""
    def __init__(self, voiced_emg_features, silent_emg_features, text, audio_discrete,
        audio_features, chunks, audio_raw):
        self._voiced_emg_features = voiced_emg_features
        self._silent_emg_features = silent_emg_features
        self._text                = text
        self._audio_discrete      = audio_discrete
        self._audio_features      = audio_features
        self.chunks               = chunks
        self._audio_raw           = audio_raw

    def get_dict(self):
        return {
            "text": self.text,
            "chunks": self.chunks,
            "voiced_emg_features": self._voiced_emg_features,
            "silent_emg_features": self._silent_emg_features,
            "audio_features": self._audio_features,
            "audio_discrete": self._audio_discrete,
        }
    
    def __str__(self):
        return self.text


class DigitalVoicingDataset(lib.sEMGDataset):
    """Encapsulation of the sEMG Silent Speech dataset released along with the
    Digital Voicing of Silent Speech dataset by David Gaddy and Dan Klein."""
    name = "Digital Voicing"

    def __init__(self,
        root_dir,
        idx_only=None,
        mel_spectograms=True,
        limit_length=False,
        add_stft_features=False,
        dataset_type=lib.sEMGDatasetType.TRAIN):

        # Union between idx list and target idx list
        files = os.listdir(root_dir)
        idx_s = set([int(fi.split("_")[0]) for fi in files])
        if idx_only:
            idx_s = idx_s.intersection(set(idx_only))
        
        if lib.sEMGDatasetType.DEV:
            print(len(idx_s), dataset_type)

        utterances = []
        sentences = {}
        books = {}

        for idx in idx_s:
            info_fi  = f"{idx}_info.json"
            emg_fi   = f"{idx}_emg.npy"
            audio_fi = f"{idx}_audio_clean.flac"

            with open(os.path.join(root_dir, info_fi)) as f:
                data = json.loads(f.read())
                sentence_idx = data["sentence_index"]
                text = data["text"]
                book = data["book"]
                chunks = data["chunks"]

                if sentence_idx == -1:
                    continue
                else:
                    if not book in books:
                        books[data["book"]] = data["book"]
                    if not text in sentences:
                        sentences[sentence_idx] = text
            
            emg_data = load_utterance(root_dir, idx, limit_length=limit_length)
            emg_data = emg_data - emg_data.mean(axis=0, keepdims=True)
            emg_features = []

            for i in range(emg_data.shape[1]):
                x   = emg_data[:,i]
                w   = double_average(x)
                p   = x - w
                r   = np.abs(p)

                w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
                p_w = librosa.feature.rms(w, frame_length=16, hop_length=6, center=False)
                p_w = np.squeeze(p_w, 0)
                p_r = librosa.feature.rms(r, frame_length=16, hop_length=6, center=False)
                p_r = np.squeeze(p_r, 0)
                z_p = librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False)
                z_p = np.squeeze(z_p, 0)
                r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)

                emg_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))

                if add_stft_features:
                    s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))
                    emg_features.append(s.T)
            
            emg_features = np.concatenate(emg_features, axis=1)
            emg_features = emg_features.astype(np.float32)

            audio_features, audio_discrete, audio_raw = \
                load_audio(
                    os.path.join(root_dir, audio_fi),
                    mel_spectograms,
                    max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf')))

            utterances.append(DigitalVoicingUtterance(
                emg_features,
                emg_features,
                text,
                audio_discrete,
                audio_features,
                chunks,
                audio_raw))

        self.utterances = utterances
        print(len(self.utterances), dataset_type)
        self.sentences = sentences
        self.books = books

        self.num_features = self.utterances[0].voiced_emg_features.shape[1]
        self.num_speech_features = self.utterances[0].audio_features.shape[1]

    """
    @property
    def num_features(self):
        return self.utterances[0].emg_features.shape[1]
    
    @property
    def num_speech_features(self):
        return self.utterances[0].audio_features.shape[1]
    """

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, i):
        return self.utterances[i].get_dict()
    
    @staticmethod
    def collate_fixed_length(batch):
        batch_size = len(batch)
        audio_features = torch.cat(
            [torch.from_numpy(utter['audio_features']) for utter in batch], 0)
        audio_discrete = torch.cat(
            [torch.from_numpy(utter["audio_discrete"]) for utter in batch], 0)

        # Only voiced emg features for now
        emg_features = torch.cat(
            [torch.from_numpy(utter['voiced_emg_features']) for utter in batch], 0)

        mb = batch_size * 8

        return {
            "audio_features": split_fixed_length(audio_features, 100)[:mb],
            "voiced_emg_features":   split_fixed_length(emg_features, 100)[:mb],
            "audio_discrete": split_fixed_length(audio_discrete, 16000)[:mb]}