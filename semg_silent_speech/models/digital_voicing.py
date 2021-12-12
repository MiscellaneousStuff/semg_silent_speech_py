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
"""Original model used in the Digital Voicing of Silent Speech paper by
David Gaddy and Dan Klein.

Model is adapted to use WaveGlow as the vocoder instead of the original
WaveNet model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from semg_silent_speech.models.transformer import TransformerEncoderLayer
from semg_silent_speech.models.lib import SequenceLayerType


class DigitalVoicingModel(nn.Module):
    def __init__(self,
                 ins,
                 model_size,
                 n_layers,
                 dropout,
                 outs,
                 sequence_layer=SequenceLayerType.LSTM):
        super().__init__()
        self.dropout = dropout
        self.sequence_layer=sequence_layer
        if sequence_layer == SequenceLayerType.LSTM:
            self.lstm = \
                nn.LSTM(
                    ins, model_size, batch_first=True,
                    bidirectional=True, num_layers=n_layers,
                    dropout=dropout)
            self.w1 = nn.Linear(model_size * 2, outs)
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
            return self.w1(x)
        elif self.sequence_layer == SequenceLayerType.TRANSFORMER:
            x = self.transformer(x)
            return self.w_out(x)