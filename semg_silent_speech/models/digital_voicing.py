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
"""Original model used in the Digital Voicing of Silent Speech paper by
David Gaddy and Dan Klein.

Model is adapted to use WaveGlow as the vocoder instead of the original
WaveNet model."""

import torch.nn as nn
import torch.nn.functional as F

from semg_silent_speech.models.transformer import TransformerEncoderLayer
from semg_silent_speech.models.lib import SequenceLayerType


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class DigitalVoicingSynthModel(nn.Module):
    def __init__(self,
                 ins,
                 model_size,
                 n_layers,
                 dropout,
                 outs,
                 n_feats,
                 n_cnn_layers,
                 reconstruct_outs=None,
                 sequence_layer=SequenceLayerType.LSTM,
                 stride=2):
        super().__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, model_size)
        self.dropout = dropout
        self.sequence_layer = sequence_layer
        self.reconstruct_outs = reconstruct_outs        
        if sequence_layer == SequenceLayerType.LSTM:
            self.lstm = \
                nn.LSTM(
                    ins, model_size, batch_first=True,
                    bidirectional=True, num_layers=n_layers,
                    dropout=dropout)
            self.w1 = nn.Linear(model_size * 2, outs)
            if reconstruct_outs:
                self.w2 = nn.Linear(model_size * 2, reconstruct_outs)

    def forward(self, x):
        if self.sequence_layer == SequenceLayerType.LSTM:
            x = self.cnn(x)
            x = self.rescnn_layers(x)
            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
            x = x.transpose(1, 2)

            x = F.dropout(x, self.dropout, training=self.training)
            x, _ = self.lstm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if not self.reconstruct_outs:
                return self.w1(x), None
            else:
                return self.w1(x), self.w2(x)


class DigitalVoicingModel(nn.Module):
    def __init__(self,
                 ins,
                 model_size,
                 n_layers,
                 dropout,
                 outs,
                 reconstruct_outs=None,
                 sequence_layer=SequenceLayerType.LSTM):
        super().__init__()
        self.dropout = dropout
        self.sequence_layer = sequence_layer
        self.reconstruct_outs = reconstruct_outs
        if sequence_layer == SequenceLayerType.LSTM:
            self.lstm = \
                nn.LSTM(
                    ins, model_size, batch_first=True,
                    bidirectional=True, num_layers=n_layers,
                    dropout=dropout)
            self.w1 = nn.Linear(model_size * 2, outs)
            if reconstruct_outs:
                self.w2 = nn.Linear(model_size * 2, reconstruct_outs)
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model=model_size,        
                nhead=8,
                relative_positional=True,
                relative_positional_distance=100,
                dim_feedforward=3072)
            self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
            self.w_out = nn.Linear(model_size, outs)

    def forward(self, x):
        if self.sequence_layer == SequenceLayerType.LSTM:
            x = F.dropout(x, self.dropout, training=self.training)
            x, _ = self.lstm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if not self.reconstruct_outs:
                return self.w1(x), None
            else:
                return self.w1(x), self.w2(x)
        elif self.sequence_layer == SequenceLayerType.TRANSFORMER:
            x = self.transformer(x)
            return self.w_out(x)