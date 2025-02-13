#!/bin/bash

# Create a new conda environment named headgendata
conda create --name headgendata python==3.10 -y
conda activate headgendata

# Install the required packages
pip install -e .
