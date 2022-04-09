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

from semg_silent_speech.datasets  import lib

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


class DigitalVoicingASRUtterance(lib.sEMGUtterance):
    """Encapsulation of an sEMG Silent Speech utterance containing the EMG
    data and text transcription."""
    def __init__(self, voiced_emg_features, silent_emg_features, text, chunks):
        self._voiced_emg_features = voiced_emg_features
        self._silent_emg_features = silent_emg_features
        self._text                = text
        self._chunks              = chunks

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
        idx_only=None,
        dataset_type=lib.sEMGDatasetType.TRAIN):
        
        self.dataset_type = dataset_type

        top_dirs = [
            "/closed_vocab/voiced/",
            "nonparallel_data/",
            "voiced_parallel_data/"]
    
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, i):
        return self.utterances[i]