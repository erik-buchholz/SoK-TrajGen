#!/usr/bin/env bash

echo "Setting up environment..."
if ! command -v python3 &> /dev/null || ! command -v conda &> /dev/null; then
    echo "Python 3.10 or Conda is not installed. Installing might require sudo access."
    echo "Installing Python 3.10..."
    sudo apt install python3.10
    echo "Downloading Conda..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniconda.sh
    echo "Installing Conda..."
    bash miniconda.sh -b -p "$HOME/miniconda"
    echo "Initialising Conda..."
    source "$HOME/miniconda/bin/activate"
    conda init
fi
VENV_NAME="stg"  # Change to your liking
# Update Conda
conda update -n base -c defaults conda
# Create environment
conda env create --name $VENV_NAME -f environment.yml
# Activate environment
conda activate $VENV_NAME
echo "Running Unittests..."
if FORCE_TESTS=false RUN_SLOW_TESTS=true python3 -m unittest discover -s test; then
    echo "Setup Successful. Done."
    echo "To activate the environment, run 'conda activate $VENV_NAME'"
else
    echo "Error - Setup unsuccessful."
fi