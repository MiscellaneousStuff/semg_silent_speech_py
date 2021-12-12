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
"""Evaluates the performance of a model against the test set using ASR
to determine the text content of audio features. This text content is compared
against the ground truth and the WER is calculated."""

import os
import torch
import random
import numpy as np
import json

from absl import flags
from absl import app

from semg_silent_speech.datasets.digital_voicing import DigitalVoicingDataset
from semg_silent_speech.models.digital_voicing import DigitalVoicingModel
from semg_silent_speech.datasets.lib import sEMGDatasetType
from semg_silent_speech.models.lib import SequenceLayerType
from semg_silent_speech.bin.train import test
from semg_silent_speech.bin.text_to_mel import save_output

FLAGS = flags.FLAGS

# flags.DEFINE_string("waveglow_checkpoint", None, "Path to a waveglow *.pt model checkpoint")
# flags.DEFINE_string("checkpoint_path", "", "(Optional) Existing model to continue training")
# flags.DEFINE_string("root_dir", None, "Root directory of the EMG dataset")
# flags.DEFINE_bool("add_stft_features", False, "Use short-time fourier transform EMG features")
# flags.DEFINE_bool("use_transformer", False, "Use transformer layer for sequence layer")
# flags.DEFINE_float("dropout", 0.0, "Dropout value")
# flags.DEFINE_integer("n_layers", 3, "Number of layers")
# flags.DEFINE_integer("model_size", 1024, "Number of hidden dimensions")
flags.DEFINE_string("audio_output_dir", None, "Directory to save audio files to")
# flags.mark_flag_as_required("waveglow_checkpoint")
# flags.mark_flag_as_required("checkpoint_path")
# flags.mark_flag_as_required("root_dir")
flags.mark_flag_as_required("audio_output_dir")

def evaluate(testset, audio_dir):
    pass

def main(unused_argv):
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("testset.json") as f:
        idx_s = json.loads(f.read())

    testset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=idx_s["test"],
        dataset_type=sEMGDatasetType.TEST,
        add_stft_features=FLAGS.add_stft_features)

    model = DigitalVoicingModel(
        ins=testset.num_features,
        model_size=FLAGS.model_size,
        n_layers=FLAGS.n_layers,
        dropout=FLAGS.dropout,
        outs=testset.num_speech_features,
        sequence_layer=SequenceLayerType.TRANSFORMER \
             if FLAGS.use_transformer \
             else SequenceLayerType.LSTM).to(device)

    model.load_state_dict(torch.load(FLAGS.checkpoint_path))

    # def save_output(waveglow_path, mel_spectogram, sr, audio_path, denoise=False):

    for i, datapoint in enumerate(testset):
        emg = datapoint["voiced_emg_features"]
        emg = torch.tensor(emg).to(device)
        emg = torch.unsqueeze(emg, 0)
        mel = model(emg)
        mel = torch.swapaxes(mel, 1, 2)

        save_output(FLAGS.waveglow_checkpoint,
                    mel,
                    22_050,
                    os.path.join(FLAGS.audio_output_dir, f"example_output_{i}.wav"),
                    denoise=False)

    error = test(model, testset, device, epoch_idx=-1)
    print(error)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)