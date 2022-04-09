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
"""Train a speech recognition model with can classify sEMG data recorded during
silent speech into text."""

import random
import numpy as np

import torch

from absl import flags
from absl import app

from semg_silent_speech.datasets.digital_voicing import DigitalVoicingDataset
from semg_silent_speech.models.digital_voicing_asr import DigitalVoicingASRModel
from semg_silent_speech.datasets.lib import sEMGDatasetType

FLAGS = flags.FLAGS

flags.DEFINE_string("root_dir", None, "Root directory of the EMG dataset")
flags.DEFINE_boolean('no_session_embed', False, "Don't use a session embedding")
flags.DEFINE_integer("random_seed", 1, "Seed value for all libraries")
flags.DEFINE_integer("epochs", 100, "Number of training epochs")

def train(trainset, devset, device, n_epochs=100):
    model = DigitalVoicingASRModel()

def main(unused_argv):
    # Set random seed using NumPy and Torch
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=[10],
        dataset_type=sEMGDatasetType.TRAIN,
        add_stft_features=False)

    devset = DigitalVoicingDataset()

    train(trainset=trainset,
          devset=devset,
          device=device,
          n_epochs=FLAGS.epochs)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)