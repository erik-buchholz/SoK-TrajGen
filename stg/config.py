#!/usr/bin/env python3
"""Contains all configuration parameters."""
import os
import sys
from pathlib import Path
from enum import Enum


# General Settings
class Backend(Enum):
    tensorflow = 'tensorflow'
    pytorch = 'pytorch'


backend = Backend.pytorch

# Directories
BASE_DIR = str(Path(__file__).parent.parent.resolve()) + '/'
TMP_DIR = BASE_DIR + 'tmp/'
LOG_DIR = BASE_DIR + 'logs/'
RESULT_DIR = BASE_DIR + 'results/'
TENSORBOARD_DIR = TMP_DIR + 'tensorboard/'
PARAM_PATH = BASE_DIR + 'parameters/'

# TensorFlow Fixes
env_dir = sys.prefix  # Get the current conda/venv environment directory
# Libdevice not found fix:
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={os.path.join(env_dir, 'lib')}"
# Reduce all the Keras/TensorFlow info messages (only show warning and above)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
# Link TensorRT
site_packages_path = os.path.join(env_dir, 'lib', 'python3.10', 'site-packages')
tensorrt_path = os.path.join(site_packages_path, 'tensorrt')
lib_path = os.path.join(env_dir, 'lib')

os.environ['LD_LIBRARY_PATH'] = f"{site_packages_path}:{tensorrt_path}:{lib_path}"
