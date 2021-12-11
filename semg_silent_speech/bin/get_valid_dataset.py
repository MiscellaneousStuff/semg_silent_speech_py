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
"""Returns a train, dev, test JSON file with idx's into a dataset of valid
utterances. Ignores any entries with a sentence index of `-1`"""

import os
import json
import random

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir", None, "Root directory of the EMG dataset")
flags.DEFINE_string("output_path", None, "Output path for the `testset.json` file")
flags.DEFINE_integer("train_count", 370, "Number of utterances for training")
flags.DEFINE_integer("dev_count", 30, "Number of utterances for validation")
flags.DEFINE_integer("test_count", 100, "Number of utterances for testing")
flags.DEFINE_integer("random_seed", 1, "Sets the PRNG seed used to shuffle the indices")
flags.mark_flag_as_required("root_dir")
flags.mark_flag_as_required("output_path")

def main(unused_argv):
    # Get info files
    files = os.listdir(FLAGS.root_dir)
    infos = [fi if fi.endswith(".json") else None for fi in files]
    infos = list(filter(lambda x: x != None, infos))
    valid = []

    # Find recordings of valid utterances
    for fi in infos:
        cur_path = os.path.join(FLAGS.root_dir, fi)
        with open(cur_path) as f:
            cur_info = json.loads(f.read())
            cur_idx  = fi.split("_")[0]
            print(fi, cur_idx)
            if cur_info["sentence_index"] != -1:
                valid.append(cur_idx)

    # Create testset comprised of valid (train, dev, test) idx_s
    random.seed(1)
    testset = {
        "train": [],
        "dev": [],
        "test": []
    }

    random.shuffle(valid)

    testset["train"] = valid[0:FLAGS.train_count]
    testset["dev"]   = valid[FLAGS.train_count:FLAGS.train_count + FLAGS.dev_count]
    testset["test"]  = valid[\
        FLAGS.train_count + FLAGS.dev_count:\
        FLAGS.train_count + FLAGS.dev_count + FLAGS.test_count]

    testset = json.dumps(testset, indent=4, sort_keys=True)

    print(f"Writing testset to '{FLAGS.output_path}'!")
    print(testset)
    with open(FLAGS.output_path, "w") as f:
        f.write(testset)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)