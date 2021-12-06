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
"""Module setuptools script."""
from setuptools import setup

description = """sEMG Silent Speech - sEMG Silent Speech Python Module
sEMG Silent Speech is the Python module which contains the classes
and methods to deal with different sEMG Silent Speech datasets, models,
visualisations, utility functions and other useful functionality.
Read the README at https://github.com/MiscellaneousStuff/semg_silent_speech_py
for more information.
"""

setup(
    name="semg_silent_speech",
    version="1.0.0",
    description="sEMG Silent Speech Python Module",
    long_description=description,
    long_description_content_type="text/markdown",
    author="Tada Makepeace",
    author_email="up904749@myport.ac.uk",
    license="MIT License",
    keywords=["sEMG", "silent speech", "machine learning", "research"],
    url="https://github.com/MiscellaneousStuff/semg_silent_speech_py",
    packages=[
        "semg_silent_speech",
        "semg_silent_speech.lib",
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)