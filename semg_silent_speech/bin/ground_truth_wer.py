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
"""Calculates the Word-Error Rate of the grouth truth audio files."""

import os
import json
import numpy as np
import soundfile as sf

from unidecode import unidecode
import jiwer
import deepspeech
import librosa

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("root_dir", None, "Directory of the ground truth audio files")
flags.mark_flag_as_required("root_dir")

def evaluate(idx_s, txts, root_dir):
    model = deepspeech.Model('deepspeech-0.7.0-models.pbmm')
    model.enableExternalScorer('deepspeech-0.7.0-models.scorer')
    predictions = []
    targets = []

    i = 0

    for idx, txt in zip(idx_s[0:10], txts[0:10]):
        cur_path = os.path.join(root_dir, f"{idx}_audio_clean.flac")
        audio, _ = librosa.load(
            cur_path,
            sr=model.sampleRate())
        audio_int16 = (audio*(2**15)).astype(np.int16)
        text = model.stt(audio_int16)
        predictions.append(text)
        target_text = unidecode(txt)
        targets.append(target_text)
        i += 1
        print(f"{i}/{len(idx_s)}", idx, text, target_text)
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    print('targets:', targets)
    print('predictions:', predictions)
    print('wer:', jiwer.wer(targets, predictions))

def main(unused_argv):
    # Get info files
    files = os.listdir(FLAGS.root_dir)
    infos = [fi if fi.endswith(".json") else None for fi in files]
    infos = list(filter(lambda x: x != None, infos))
    idx_s = [] # List of indices of valid utterances
    txts  = [] # Ground truth text values for valid utterances

    # Find recordings of valid utterances
    for fi in infos:
        cur_path = os.path.join(FLAGS.root_dir, fi)
        with open(cur_path) as f:
            cur_info = json.loads(f.read())
            cur_idx  = fi.split("_")[0]
            print(fi, cur_idx)
            if cur_info["sentence_index"] != -1:
                idx_s.append(cur_idx)
                txts.append(cur_info["text"])

    evaluate(idx_s, txts, FLAGS.root_dir)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)