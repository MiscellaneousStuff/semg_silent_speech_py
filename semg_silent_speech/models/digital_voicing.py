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


class DigitalVoicingModel(nn.Module):
    def __init__(self, ins, model_size, n_layers, dropout, outs):
        super().__init__()
        self.dropout = dropout
        self.lstm = \
            nn.LSTM(
                ins, model_size, batch_first=True,
                bidirectional=True, num_layers=n_layers,
                dropout=dropout)
        self.w1 = nn.Linear(model_size * 2, outs)
    
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x, _ = self.lstm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.w1(x)