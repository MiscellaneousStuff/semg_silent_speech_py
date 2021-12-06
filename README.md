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
University of Portsmouth.

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