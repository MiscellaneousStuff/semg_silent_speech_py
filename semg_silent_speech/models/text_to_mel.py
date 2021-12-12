import sys
import os
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

from waveglow.tacotron2.model import Tacotron2
from waveglow.tacotron2.layers import TacotronSTFT, STFT
from waveglow.tacotron2.audio_processing import griffin_lim
from waveglow.tacotron2.train import load_model
from waveglow.tacotron2.text import text_to_sequence

from waveglow.denoiser import Denoiser
import waveglow.glow as glow

# NOTE: This needs to be top most because of __futures__ imports
from semg_silent_speech.models.hparams import create_hparams

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    fig.savefig("text_to_mel.png")

def infer(waveglow_checkpoint_path, tacotron_checkpoint_path, text):
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    model = load_model(hparams)
    model.load_state_dict(torch.load(tacotron_checkpoint_path)['state_dict'])
    _ = model.cuda().eval() # .half()

    waveglow = torch.load(waveglow_checkpoint_path)['model']

    waveglow.cuda().eval() # .half()
    for k in waveglow.convinv:
        k.float()

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
            mel_outputs_postnet.float().data.cpu().numpy()[0],
            alignments.float().data.cpu().numpy()[0].T))