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
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from absl import flags
from absl import app

from semg_silent_speech.datasets.digital_voicing_asr import DigitalVoicingASRDataset
from semg_silent_speech.models.digital_voicing_asr import DigitalVoicingASRModel
from semg_silent_speech.datasets.lib import sEMGDatasetType

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("root_dir", None, "Root directory of the EMG dataset")
flags.DEFINE_boolean('no_session_embed', False, "Don't use a session embedding")
flags.DEFINE_integer("random_seed", 1, "Seed value for all libraries")
flags.DEFINE_integer("epochs", 100, "Number of training epochs")
flags.DEFINE_integer("rnn_dim", 512, "RNN hidden state dimension size")
flags.DEFINE_integer("num_workers", 0, "Number of threads to async load data into model")
flags.DEFINE_float("learning_rate", 5e-4, "Initial learning rate")
flags.DEFINE_integer("batch_size", 32, "Number of examples per single mini-batch")

def train(trainset, devset, device, n_epochs=100):
    dataloader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        pin_memory=(device=="cuda"),
        collate_fn=lambda x: trainset.data_processing(x, "train"),
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers)

    model = DigitalVoicingASRModel(
        ins=trainset.num_features,
        rnn_dim=512,
        n_class=len(trainset.encoder.vocab),
        num_sessions=trainset.num_sessions,
        session_embed=not FLAGS.no_session_embed).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate)
    criterion = nn.CTCLoss(blank=28).to(device)

    best_validation = float("inf")

    data_len = len(dataloader)

    losses_s = []
    for epoch_idx in range(n_epochs):
        losses = []
        start = time.time()

        for batch_idx, _data in enumerate(dataloader):
            emg_data_s, session_ids, labels, input_lengths, label_lengths = _data
            emg_data_s, session_ids, labels = \
                emg_data_s.to(device), session_ids.to(session_ids), labels.to(device)

            optimizer.zero_grad()

            output = model(emg_data_s, session_ids)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)

            loss = criterion(output, labels, input_lengths, label_lengths)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx * len(emg_data_s), data_len,
                    100. * batch_idx / len(dataloader), loss.item()))

            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        losses_s.append(train_loss)
        print("EPOCH, TRAIN_LOSS:", epoch_idx, train_loss)

        import matplotlib.pyplot as plt
        plt.plot(losses_s)
        plt.savefig("train_loss.png")

def main(unused_argv):
    # Set random seed using NumPy and Torch
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = DigitalVoicingASRDataset(
        root_dir=FLAGS.root_dir,
        dataset_type=sEMGDatasetType.TRAIN,
        add_stft_features=False)

    # print("feats, sess_count:", trainset.num_features, trainset.num_sessions)

    train(trainset=trainset,
          devset=trainset,
          device=device,
          n_epochs=FLAGS.epochs)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)