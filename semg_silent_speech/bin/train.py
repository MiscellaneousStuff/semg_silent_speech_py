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
"""Train a transduction or data augmentation model."""


import numpy as np
import time
import random

import torch
import torch.nn.functional as F

from absl import flags
from absl import app

from semg_silent_speech.datasets.digital_voicing import DigitalVoicingDataset
from semg_silent_speech.models.digital_voicing import DigitalVoicingModel

FLAGS = flags.FLAGS

flags.DEFINE_string("root_dir", None, "Root directory of the EMG dataset")
flags.DEFINE_float("dropout", 0.0, "Dropout value")
flags.DEFINE_integer("model_size", 1024, "Number of hidden dimensions")
flags.DEFINE_integer("batch_size", 32, "Training batch size")
flags.DEFINE_integer("n_layers", 3, "Number of layers")
flags.DEFINE_integer("idx_only", -1, "Train on a single sample only")
flags.DEFINE_float("learning_rate", 0.001, "Step size during backpropagation")
flags.DEFINE_integer("random_seed", 1, "Seed value for all libraries")
flags.DEFINE_integer("n_epochs", 100, "Number of training epochs")
flags.DEFINE_string("checkpoint_path", "", "(Optional) Existing model to continue training")
flags.DEFINE_float("datasize_fraction", 1.0, "Percentage of the entire dataset to train on")
flags.mark_flag_as_required("root_dir")

def train(trainset, devset, device, n_epochs=100, checkpoint_path=None):
    training_subset = torch.utils.data.Subset(
        trainset,
        list(range(int(len(trainset) * FLAGS.datasize_fraction))))

    dataloader = torch.utils.data.DataLoader(
        training_subset,
        shuffle=True,
        pin_memory=(device=="cuda"),
        collate_fn=devset.collate_fixed_length,
        batch_size=FLAGS.batch_size)

    model = DigitalVoicingModel(
        ins=devset.num_features,
        model_size=FLAGS.model_size,
        n_layers=FLAGS.n_layers,
        dropout=FLAGS.dropout,
        outs=devset.num_speech_features).to(device)
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    optim = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    best_validation = float("inf")
    best_loss = float("inf")

    losses_s = []
    for epoch_idx in range(n_epochs):
        losses = []
        start = time.time()
        for batch in dataloader:
            optim.zero_grad()
            X = batch["voiced_emg_features"].to(device) # NOTE: Only voiced for now
            y = batch["audio_features"].to(device)

            pred = model(X)
            if pred.shape[0] != y.shape[0]:
                min_first_dim = min(pred.shape[0], y.shape[0])
                if pred.shape[0] != min_first_dim:
                    pred = pred[min_first_dim:, :, :]
                if y.shape[0] != min_first_dim:
                    y = y[min_first_dim:, :, :]
            
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())

            loss.backward()
            optim.step()
        
        train_loss = np.mean(losses)

        end = time.time() - start
        print(f"epoch: {epoch_idx+1}, loss: {train_loss:.4f}, tm: {end}")

        losses_s.append(train_loss)

def main(unused_argv):
    # Set random seed using NumPy and Torch
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)

    # Get training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_only = FLAGS.idx_only if FLAGS.idx_only != -1 else None

    trainset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=idx_only)

    devset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=idx_only)

    train(trainset=trainset,
          devset=devset,
          device=device,
          n_epochs=FLAGS.n_epochs,
          checkpoint_path=FLAGS.checkpoint_path)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)