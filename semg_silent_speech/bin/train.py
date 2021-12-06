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
flags.mark_flag_as_required("root_dir")

def main(unused_argv):
    idx_only = FLAGS.idx_only if FLAGS.idx_only != -1 else None

    dataset = DigitalVoicingDataset(
        root_dir=FLAGS.root_dir,
        idx_only=idx_only)

    print('len(dataset):', len(dataset))
    
    model = DigitalVoicingModel(
        ins=dataset.num_features,
        model_size=FLAGS.model_size,
        n_layers=FLAGS.n_layers,
        dropout=FLAGS.dropout,
        outs=dataset.num_speech_features)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)