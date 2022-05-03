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
"""Tries to generate ^E'v `emg_features` samples using the EMG augment model
with ground truth Av `audio_features` as the input."""

import numpy as np
import time
import random
import json
import os
import re

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler

from absl import flags
from absl import app

from semg_silent_speech.datasets.digital_voicing import DigitalVoicingDataset
from semg_silent_speech.datasets.lib import sEMGDatasetType
from semg_silent_speech.models.digital_voicing import DigitalVoicingModel
from semg_silent_speech.models.lib import SequenceLayerType
from semg_silent_speech.lib import plotting

FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir", None, "Root directory of the EMG dataset")
flags.DEFINE_string("augment_model_path", None, "Pre-trained data augment model to visualise")
flags.DEFINE_float("dropout", 0.0, "Dropout value")
flags.DEFINE_integer("model_size", 1024, "Number of hidden dimensions")
flags.DEFINE_integer("batch_size", 32, "Training batch size")
flags.DEFINE_integer("n_layers", 3, "Number of layers")
flags.DEFINE_bool("data_augment_model", True, "Enable this to train a speech feature to EMG model instead")
flags.DEFINE_bool("amp", False, "Enables automated mixed precision.")
flags.DEFINE_bool("add_stft_features", False, "Use short-time fourier transform EMG features")
flags.DEFINE_bool("use_transformer", False, "Use transformer layer for sequence layer")
flags.DEFINE_integer("random_seed", 1, "Seed value for all libraries")
flags.DEFINE_string("dataset", "train", "Dataset slice to use for visualisation")
flags.mark_flag_as_required("root_dir")
flags.mark_flag_as_required("augment_model_path")

def visualise(devset, device, augment_model_path):
    dataloader = torch.utils.data.DataLoader(devset, batch_size=1)

    model = DigitalVoicingModel(
        ins=devset.num_features \
            if not FLAGS.data_augment_model \
            else devset.num_speech_features,
        model_size=FLAGS.model_size,
        n_layers=FLAGS.n_layers,
        dropout=FLAGS.dropout,
        outs=devset.num_speech_features \
             if not FLAGS.data_augment_model \
             else devset.num_features,
        sequence_layer=SequenceLayerType.TRANSFORMER \
             if FLAGS.use_transformer \
             else SequenceLayerType.LSTM).to(device)

    model.load_state_dict(torch.load(augment_model_path))
    model.eval()

    i = 0
    losses = []
    with torch.no_grad():
        for example in dataloader:
            print(f"Plotting EMG augment: {i}")
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
            
            fig = plotting.plot_pred_y_emg_features(
                plotting.stack_emg_features(pred),
                plotting.stack_emg_features(y),
                epoch_idx=f'{i} - "{example["text"][0]}"')
            fig.savefig(f"./augment_vs_ground_visuals/example_augment_{i}.png")
            i += 1
            plt.close()
            
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

    devset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=idx_s[FLAGS.dataset],
        dataset_type=sEMGDatasetType.DEV,
        add_stft_features=FLAGS.add_stft_features)

    visualise(devset, device, FLAGS.augment_model_path)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)