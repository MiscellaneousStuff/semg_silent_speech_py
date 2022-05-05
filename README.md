# sEMG Silent Speech - sEMG Silent Speech Python Module

sEMG Silent Speech is the Python module which contains the classes
and methods to deal with different sEMG Silent Speech datasets, models,
visualisations, utility functions and other useful functionality.

A lot of the initial code in this Python module is adapted from the
code released with the
[Improved Voicing of Silent Speech](https://arxiv.org/pdf/2106.01933v1.pdf)
paper in the
[dgaddy/silent_speech](https://github.com/dgaddy/silent_speech) GitHub repository.
Code adapted from this repository will be attributed to correctly under the MIT
License which the code is released under.

## About

This module is produced as part of the requirement for my Final Year
Project for my BSc Computer Science degree requirement at the
University of Portsmouth. This module represents the first round of experiments
for my research project which involves producing a system which considered
how to synthesize surface EMG signals during the voiced speaking condition.

# Quick Start Guide

## Get sEMG Silent Speech

### From Source

You can install sEMG Silent Speech from a local clone of the git repo:

```bash
git clone https://github.com/MiscellaneousStuff/semg_silent_speech_py.git
cd semg_silent_speech_py
git submodule init
git submodule update
pip install --upgrade semg_silent_speech_py/
```

If you also want to use the EMG data augmentation code, you'll also need
to enable `tacotron2` within WaveGlow by doing the following after running
the above code:

```bash
cd semg_silent_speech_py
cd waveglow
git submodule init
git submodule update
cd tacotron2
git submodule init
git submodule update
cd waveglow
git submodule init
git submodule update
```

## Inference

### Transduction

This model converts a sequence of EMG features into a sequence of audio features.

### EMG Augmentation

This model converts a sequence of audio features into EMG features.

## Training

### Transduction

#### Ground Truth Data Only

This training type only uses the original ground truth dataset.

#### Augmented Data Only

This training type uses only data which is produced using the EMG data synthesis
model (i.e. Data Augmentation Model). This training mode is used to benchmark
how well the augmented data alone performs.