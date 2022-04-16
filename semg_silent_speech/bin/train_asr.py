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
from jiwer import wer, cer
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler

import neptune.new as neptune

from absl import flags
from absl import app

from semg_silent_speech.datasets.digital_voicing_asr import DigitalVoicingASRDataset
from semg_silent_speech.models.digital_voicing_asr import DigitalVoicingASRModel
from semg_silent_speech.datasets.lib import sEMGDatasetType
from semg_silent_speech.lib.asr import GreedyDecoder
from semg_silent_speech.lib import plotting

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("root_dir", None, "Root directory of the EMG dataset")
flags.DEFINE_boolean('no_session_embed', False, "Don't use a session embedding")
flags.DEFINE_integer("random_seed", 1, "Seed value for all libraries")
flags.DEFINE_integer("epochs", 100, "Number of training epochs")
flags.DEFINE_integer("rnn_dim", 512, "RNN hidden state dimension size")
flags.DEFINE_integer("num_workers", 1, "Number of threads to async load data into model")
flags.DEFINE_float("learning_rate", 5e-4, "Initial learning rate")
flags.DEFINE_integer("batch_size", 32, "Number of examples per single mini-batch")
flags.DEFINE_float("dropout", 0.1, "Dropout percentage")
flags.DEFINE_bool("amp", True, "Use mixed precision training")
flags.DEFINE_bool("add_stft_features", False, "Add stft features for EMG input")
flags.DEFINE_integer("n_rnn_layers", 5, "Number of BiRNN layers")
flags.DEFINE_string("neptune_token", "", "(Optional) Neptune.ai logging token")
flags.DEFINE_string("neptune_project", "", "(Optional) Neptune.ai project name")
flags.DEFINE_string("encoder_vocab", "", "(Optional) Define the encoder vocabulary")
flags.DEFINE_integer('learning_rate_patience', 5, 'Learning rate decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'Steps of linear warmup')
flags.DEFINE_bool("sanity_test", False, "Trains and evals on one same utterance.")
flags.DEFINE_float('l2', 1e-7, 'weight decay')
flags.DEFINE_float('reduce_lr_factor', 0.5, "Factor to reduce LR by on plateau")
flags.mark_flag_as_required("root_dir")

def test(model, device, test_loader, encoder, criterion, run, blank_label):
    model.eval()
    test_loss = 0

    test_cer, test_wer = [], []

    with torch.no_grad():
        for batch_idx, _data in enumerate(test_loader):
            emg_data_s, session_ids, labels, input_lengths, label_lengths = _data
            emg_data_s, session_ids, labels = \
                emg_data_s.to(device), session_ids.to(session_ids), labels.to(device)
            
            with torch.autocast(
                enabled=FLAGS.amp,
                dtype=torch.bfloat16,
                device_type="cuda"):

                output = model(emg_data_s, session_ids)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)

                loss = criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = \
                GreedyDecoder(output.transpose(0, 1), labels, label_lengths, encoder, blank_label)
            
            print("Targets:", decoded_targets)
            print("Preds:", decoded_preds)

            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    run["test_loss"].log(test_loss)
    run["cer"].log(avg_cer)
    run["wer"].log(avg_wer)
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    return test_loss, avg_wer

def train(dataset, device, run, n_epochs=100):
    dataset_len = int(len(dataset) * 1.0)
    train_split = int(dataset_len * 0.9)
    test_split  = dataset_len - train_split

    if not FLAGS.sanity_test:
        train_dataset, test_dataset = \
            torch.utils.data.random_split(dataset, [train_split, test_split])
        # train_dataset = torch.utils.data.Subset(dataset, range(len(dataset)))
        # test_dataset  = torch.utils.data.Subset(dataset, range(len(dataset)))
    else:
        sanity_test_num = 50
        run["sanity_test_num"] = sanity_test_num
        train_dataset = torch.utils.data.Subset(dataset, range(sanity_test_num))
        test_dataset  = torch.utils.data.Subset(dataset, range(sanity_test_num))

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=(device=="cuda"),
        collate_fn=lambda x: dataset.data_processing(x, "train"),
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=(device=="cuda"),
        collate_fn=lambda x: dataset.data_processing(x, "test"),
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers)

    model = DigitalVoicingASRModel(
        ins=dataset.num_features,
        rnn_dim=FLAGS.rnn_dim,
        n_rnn_layers=FLAGS.n_rnn_layers,
        n_class=len(dataset.encoder.vocab),
        num_sessions=dataset.num_sessions,
        session_embed=not FLAGS.no_session_embed,
        dropout=FLAGS.dropout).to(device)

    blank_label = 0
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=FLAGS.learning_rate,
                                  weight_decay=FLAGS.l2)
    criterion = nn.CTCLoss(blank=blank_label).to(device)
    
    """
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS.learning_rate, 
                                            steps_per_epoch=int(len(dataloader)),
                                            epochs=FLAGS.epochs,
                                            anneal_strategy='linear')
    """

    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(\
        optimizer, 'min', FLAGS.reduce_lr_factor, patience=FLAGS.learning_rate_patience)
    

    run["lr_scheduler"] = type(scheduler)

    def set_lr(new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    best_validation = float("inf")

    data_len = len(dataloader)

    scaler = GradScaler()

    losses_s = []
    lr_s = []
    batch_count = 0
    for epoch_idx in range(n_epochs):
        model.train()
        
        losses = []
        for batch_idx, _data in enumerate(dataloader):
            emg_data_s, session_ids, labels, input_lengths, label_lengths = _data
            emg_data_s, session_ids, labels = \
                emg_data_s.to(device), session_ids.to(session_ids), labels.to(device)

            # optimizer.zero_grad()
            schedule_lr(batch_count)

            with torch.autocast(
                enabled=FLAGS.amp,
                dtype=torch.bfloat16,
                device_type="cuda"):

                output = model(emg_data_s, session_ids)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)

                loss = criterion(output, labels, input_lengths, label_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # OneCycleLR ONLY
            # scheduler.step()

            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx * len(emg_data_s), data_len,
                    100. * batch_idx / len(dataloader), loss.item()))

            losses.append(loss.item())
            batch_count += 1

        train_loss = sum(losses) / len(losses)
        losses_s.append(train_loss)
        run["loss"].log(train_loss)
        run["learning_rate"].log(optimizer.param_groups[0]['lr'])
        # run["learning_rate"].log(FLAGS.learning_rate)
        print("EPOCH, TRAIN_LOSS:", epoch_idx, train_loss)

        val_loss, avg_wer = \
            test(model, device, testloader, dataset.encoder, criterion, run, blank_label)
        scheduler.step(val_loss)

def main(unused_argv):
    # Set Neptune.ai handler
    run = neptune.init(project=FLAGS.neptune_project,
                       api_token=FLAGS.neptune_token)

    dataset = DigitalVoicingASRDataset(
        root_dir=FLAGS.root_dir,
        dataset_type=sEMGDatasetType.TRAIN,
        encoder_vocab=FLAGS.encoder_vocab,
        add_stft_features=FLAGS.add_stft_features)

    # Log which directories are being used in each experiment
    run["top_dirs"] = dataset.top_dirs
    run["target_sub_dirs"] = dataset.target_sub_dirs

    # Log parameters
    run["parameters"] = {
        "root_dir": FLAGS.root_dir,
        "dropout": FLAGS.dropout,
        "learning_rate": FLAGS.learning_rate,
        "n_rnn_layers": FLAGS.n_rnn_layers,
        "amp": FLAGS.amp,
        "batch_size": FLAGS.batch_size,
        "random_seed": FLAGS.random_seed,
        "epochs": FLAGS.epochs,
        "rnn_dim": FLAGS.rnn_dim,
        "no_session_embed": FLAGS.no_session_embed,
        "neptune_token": FLAGS.neptune_token,
        "neptune_project": FLAGS.neptune_project,
        "encoder_vocab": FLAGS.encoder_vocab,
        "add_stft_features": FLAGS.add_stft_features,
        "learning_rate_patience": FLAGS.learning_rate_patience,
        "sanity_test": FLAGS.sanity_test,
        "learning_rate_warmup": FLAGS.learning_rate_warmup,
        "l2": FLAGS.l2,
        "reduce_lr_factor": FLAGS.reduce_lr_factor,
        "dataset_len": len(dataset)
    }

    # Set random seed using NumPy and Torch
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print("feats, sess_count:", trainset.num_features, trainset.num_sessions)

    train(dataset=dataset,
          device=device,
          run=run,
          n_epochs=FLAGS.epochs)

    run.stop()

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)