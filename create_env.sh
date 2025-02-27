#!/bin/bash

# Create a new conda environment named headgendata
conda create --name headgendata python==3.10 -y
conda activate headgendata
# Install ffmpeg
conda install -c conda-forge ffmpeg -y
# Install the required packages
pip install -e .
