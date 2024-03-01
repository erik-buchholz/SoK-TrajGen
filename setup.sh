#!/usr/bin/env bash

echo "Setting up environment..."
if ! command -v python3 &> /dev/null || ! python3 -c "import venv" &> /dev/null; then
    echo "Either Python 3 is not installed or the venv module is missing. Installing them requires sudo access."
    sudo apt install python3.10 python3.10-venv
fi
VENV_NAME="venv"
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Running Unittests..."
if FORCE_TESTS=false RUN_SLOW_TESTS=true python3 -m unittest discover -s test; then
    echo "Setup Successful. Done."
    echo "To activate the environment, run 'source $VENV_NAME/bin/activate'"
else
    echo "Error - Setup unsuccessful."
fi
