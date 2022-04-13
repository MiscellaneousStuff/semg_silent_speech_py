# MIT License
# 
# Copyright (c) 2022 Tada Makepeace
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
"""Define the Digital Voicing of Silent Speech dataset for the automatic
speech recongnition task."""

import os
import json
import numpy as np
import librosa
import jiwer

import torch
import torch.nn as nn

from unidecode import unidecode

from semg_silent_speech.datasets  import lib
from semg_silent_speech.lib.emg   import load_utterance
from semg_silent_speech.lib.utils import double_average, split_fixed_length
from semg_silent_speech.lib.asr   import get_encoder, normalise_text

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

"""
Original pre-processing for `Digital Voicing of Silent Speech` paper
and EMG augmentation task.
def get_emg_features(emg_data, add_stft_features=False):
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

    return emg_features
"""

def get_emg_features(emg_data, add_stft_features=False):
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

    return emg_features

class DigitalVoicingASRUtterance(lib.sEMGUtterance):
    """Encapsulation of an sEMG Silent Speech utterance containing the EMG
    data and text transcription."""
    def __init__(self, voiced_emg_features, text, chunks, session_id):
        self._voiced_emg_features = voiced_emg_features
        self._text                = text
        self._chunks              = chunks
        self._session_id          = session_id

    def __str__(self):
        return self.text


class DigitalVoicingASRDataset(lib.sEMGDataset):
    """Encapsulation of the sEMG Silent Speech dataset released along with the
    Digital Voicing of Silent Speech dataset by David Gaddy and Dan Klein.
    However this class only uses the EMG data and the text transcription
    as this is a text prediction task, not audio synthesis task.
    
    This class further generalises between the original ground truth dataset
    and EMG data which has been synthesized on a model trained to create
    genuine samples from the same distribution as the ground truth dataset."""
    name = "Digital Voicing ASR"

    def __init__(self,
        root_dir=None,
        add_stft_features=False,
        encoder_vocab=None,
        dataset_type=lib.sEMGDatasetType.TRAIN):
        
        self.utterances = []
        if encoder_vocab:
            self.encoder = get_encoder(encoder_vocab)
        else:
            self.encoder = get_encoder()

        self.dataset_type = dataset_type
        
        """
        top_dirs = [
            "closed_vocab/voiced/"
            "voiced_parallel_data/",
            "nonparallel_data/"]
        """

        top_dirs = [
            "closed_vocab/voiced/"]
        num_sessions = 0

        done = False

        for top_dir in top_dirs:
            print(os.path.join(root_dir, top_dir))
            sub_dirs = os.listdir(os.path.join(root_dir, top_dir))

            for sub_dir in sub_dirs:
                print(sub_dir)

                num_sessions += 1
                cur_fis   = os.listdir(os.path.join(root_dir, top_dir, sub_dir))
                cur_infos = list(filter(lambda fi: fi.endswith(".json"), cur_fis))

                for info_fi in cur_infos:
                    done = False
                    if not done:
                        info_path = os.path.join(root_dir, top_dir, sub_dir, info_fi)

                        with open(info_path) as f:
                            info = json.loads(f.read())
                            sentence_idx = info["sentence_index"]

                            if sentence_idx != -1:
                                file_idx = info_fi.split("_")[0]
                                cur_path = os.path.join(root_dir, top_dir, sub_dir)
                                session_id = num_sessions

                                chunks = info["chunks"]
                                emg_data = \
                                    get_emg_features(
                                        load_utterance(cur_path, file_idx),
                                        add_stft_features)
                                print("EMG SHAPE:", emg_data.shape)
                                text = unidecode(info["text"])
                                
                                utter = DigitalVoicingASRUtterance(
                                    emg_data,
                                    text,
                                    chunks,
                                    session_id)
                                self.utterances.append(utter)
                    else:
                        done = True

        self.num_features = self.utterances[0]._voiced_emg_features.shape[1]
        self.num_sessions = num_sessions
        
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, i):
        return self.utterances[i]

    def data_processing(self, batch, data_type="train"):
        emg_data_s    = []
        session_ids   = []
        labels        = []
        input_lengths = []
        label_lengths = []

        for example in batch:
            emg_data = torch.from_numpy(example._voiced_emg_features)
            text = example._text

            try:
                label = normalise_text(text)
                #print("label:", len(label))
                label = self.encoder.batch_encode(label)
            except Exception as e:
                label = self.encoder.batch_encode([" "])
                #print("error:", e, text, len(text), normalise_text(text))

            session_id = np.full(emg_data.shape[0], example._session_id, dtype=np.int64)

            emg_data_s.append(emg_data)
            session_ids.append(session_id)
            labels.append(label)
            input_lengths.append(emg_data.shape[0]//2) # Half the number of EMG features?
            label_lengths.append(len(label))

        emg_data_s = \
            nn.utils.rnn.pad_sequence(emg_data_s, batch_first=True) # .transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        # Get session_ids tensor
        session_ids = [torch.from_numpy(ex) for ex in session_ids]
        session_ids = nn.utils.rnn.pad_sequence(session_ids, batch_first=True)
        
        """
        new_sess_ids = session_ids[0]
        for sess_ids in session_ids[1:]:
            new_sess_ids = torch.cat((new_sess_ids, sess_ids), axis=0)
        session_ids = new_sess_ids
        """

        #print("lens:", len(emg_data_s), len(session_ids), len(labels))
        #print("types:", type(emg_data_s), type(session_ids), type(labels))
        #print("shapes:", emg_data_s.shape, session_ids.shape, labels.shape)
        return emg_data_s, torch.tensor([]), labels, input_lengths, label_lengths