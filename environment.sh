#!/bin/bash
# These are the commands to build the all-in-one environment for ESM4SL.
# Some versions are specifically for linux, cuda=11.8, but can change to other version based on the settings on different computers.

conda create -n esm4sl python=3.10
conda activate esm4sl
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # numpy should have been installed in this step
pip install pandas matplotlib seaborn scikit-learn pytorch-lightning wandb transformers easydict biopython scipy peft fair-esm fvcore omegaconf tensorboard
