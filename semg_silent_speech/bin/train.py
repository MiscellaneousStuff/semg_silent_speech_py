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
import json

import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler

from absl import flags
from absl import app

from semg_silent_speech.datasets.digital_voicing import DigitalVoicingDataset
from semg_silent_speech.datasets.lib import sEMGDatasetType
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
flags.DEFINE_bool("data_augment_model", False, "Enable this to train a speech feature to EMG model instead")
flags.DEFINE_bool("amp", False, "Enables automated mixed precision.")
flags.DEFINE_integer("learning_rate_patience", 5, "Learning rate decay patience")
flags.mark_flag_as_required("root_dir")

def test(model, testset, device):
    model.eval()

    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)

    losses = []
    with torch.no_grad():
        i = 0
        for example in dataloader:
            if not FLAGS.data_augment_model:
                X = example["voiced_emg_features"].to(device) # NOTE: Only voiced for now
                y = example["audio_features"].to(device)
            else:
                X = example["audio_features"].to(device)
                y = example["voiced_emg_features"].to(device) # NOTE: Only voiced for now

            with torch.autocast(
                enabled=FLAGS.amp,
                dtype=torch.bfloat16,
                device_type="cuda"):
                pred = model(X)
                if pred.shape[0] != y.shape[0]:
                    min_first_dim = min(pred.shape[0], y.shape[0])
                    if pred.shape[0] != min_first_dim:
                        pred = pred[min_first_dim:, :, :]
                    if y.shape[0] != min_first_dim:
                        y = y[min_first_dim:, :, :]
                
                loss = F.mse_loss(pred, y)
                losses.append(loss.item())
            i += 1

    model.train()

    return np.mean(losses)

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
        ins=devset.num_features \
            if not FLAGS.data_augment_model \
            else devset.num_speech_features,
        model_size=FLAGS.model_size,
        n_layers=FLAGS.n_layers,
        dropout=FLAGS.dropout,
        outs=devset.num_speech_features \
             if not FLAGS.data_augment_model \
             else devset.num_features).to(device)
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    optim = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(\
        optim, 'min', 0.5, patience=FLAGS.learning_rate_patience)

    best_validation = float("inf")
    best_loss = float("inf")

    scaler = GradScaler()

    losses_s = []
    for epoch_idx in range(n_epochs):
        losses = []
        start = time.time()
        for batch in dataloader:
            optim.zero_grad()

            if not FLAGS.data_augment_model:
                X = batch["voiced_emg_features"].to(device) # NOTE: Only voiced for now
                y = batch["audio_features"].to(device)
            else:
                X = batch["audio_features"].to(device)
                y = batch["voiced_emg_features"].to(device) # NOTE: Only voiced for now

            with torch.autocast(
                enabled=FLAGS.amp,
                dtype=torch.bfloat16,
                device_type="cuda"):
                pred = model(X)
                if pred.shape[0] != y.shape[0]:
                    min_first_dim = min(pred.shape[0], y.shape[0])
                    if pred.shape[0] != min_first_dim:
                        pred = pred[min_first_dim:, :, :]
                    if y.shape[0] != min_first_dim:
                        y = y[min_first_dim:, :, :]
                
                loss = F.mse_loss(pred, y)
                losses.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        
        val_start = time.time()
        val = test(model, devset, device)
        lr_sched.step(val)

        train_loss = np.mean(losses)

        end = time.time() - start
        val_end = time.time() - val_start
        print(f"epoch: {epoch_idx+1}, vloss: {val}, loss: {train_loss:.4f}, tm: {end}, val_tm: {val_end}")

        losses_s.append(train_loss)

def main(unused_argv):
    # Set random seed using NumPy and Torch
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)

    # Get trainset indices
    with open("testset.json") as f:
        idx_s = json.loads(f.read())

    # Get training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_only = [FLAGS.idx_only] if FLAGS.idx_only != -1 else None

    print('Loading train...')
    trainset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=idx_s["train"] if not idx_only else idx_only,
        dataset_type=sEMGDatasetType.TRAIN)

    print('Loading dev...')
    devset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=idx_s["dev"] if not idx_only else idx_only,
        dataset_type=sEMGDatasetType.DEV)

    train(trainset=trainset,
          devset=devset,
          device=device,
          n_epochs=FLAGS.n_epochs,
          checkpoint_path=FLAGS.checkpoint_path)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)