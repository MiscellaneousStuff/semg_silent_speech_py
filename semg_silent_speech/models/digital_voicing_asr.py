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
"""New proposed model to classify speech from surface electromyography
(sEMG) signals during silent speech by Tada Makepeace. The model is a
mixture of a recent DeepSpeech2 model and the original transduction model
proposed in the Digital Voicing of Silent Speech paper.

This approach allows the text classification of silent speech without requiring
audio data from users which makes it more flexible. The text prediction model
can also be improved with a language model on the predictions which improves
it's performance even further.

Text output from this model can also then used with FastSpeech2 (trained
on the text and audio of the silent speech dataset) along with the WaveGlow
vocoder."""

import torch.nn as nn
import torch.nn.functional as F


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class DigitalVoicingASRModel(nn.Module):
    def __init__(self,
                 ins,
                 rnn_dim,
                 n_rnn_layers,
                 n_class,
                 num_sessions,
                 dropout=0.1,
                 session_embed=True,
                 emb_size=32):
        super().__init__()

        self.session_embed = session_embed
        if session_embed:
            self.session_emb = nn.Embedding(num_sessions, emb_size)
            self.w_emb = nn.Linear(emb_size, rnn_dim)

        """
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        """

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=ins if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x_feat, session_ids):
        if not self.session_embed:
            x = x_feat
        else:
            emb = self.session_emb(session_ids)
            x = x_feat + self.w_emb(emb)
        
        x = self.birnn_layers(x)
        x = self.classifier(x)

        return x