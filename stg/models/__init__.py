#!/usr/bin/env python3
""" """
from enum import Enum

from .ar_rnn import AR_RNN
from .start_rnn import StartRNN
from .noise_trajgan import Noise_TrajGAN
from .cnn_gan import CNN_GAN
from .rgan import RGAN
from .geotrajgan import GeoTrajGAN


class RNNModels(str, Enum):
    """Enumeration of RNN-based models."""
    AR_RNN = 'ar_rnn'
    START_RNN = 'start_rnn'


RNN_MODEL_CLASSES = {
    RNNModels.AR_RNN: AR_RNN,
    RNNModels.START_RNN: StartRNN
}
