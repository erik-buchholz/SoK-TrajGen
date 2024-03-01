#!/usr/bin/env python3
""" """
from unittest import TestCase

import numpy as np

# noinspection PyUnresolvedReferences
from stg import config  # Required to suppress TensorFlow warnings
import tensorflow as tf

# Deactivate GPU for testing
tf.config.set_visible_devices([], 'GPU')

from stg.models.utils import l1_regularizer
from stg.ml_tf.lstm_trajgan import LSTM_TrajGAN
from stg.models.lstm_trajgan import LSTM_TrajGAN as torchGAN
from stg.utils.helpers import count_parameters_torch

# @unittest.skip("Very slow due to TensorFlow imports.")
class Test(TestCase):

    def test_parameter_count(self):
        """
        Verify that the TF and Torch implementations have the same parameters
        Note: Both the discriminator and the generator will have an additional
        400 parameters due to PyTorch's LSTM implementation using an additional
        400 parameters.
        """
        ref = (0.0, 0.0)
        sf = (1.0, 1.0)
        tf_gan = LSTM_TrajGAN(ref, sf, 100)
        torch_gan = torchGAN(ref, sf)
        self.assertEqual(
            count_parameters_torch(torch_gan.gen),
            tf_gan.generator.count_params() + 400
        )
        self.assertEqual(
            count_parameters_torch(torch_gan.dis),
            tf_gan.discriminator.count_params() + 400
        )

    def test_l1_regularizer(self):
        import tensorflow as tf
        import torch
        n = np.random.randn(10)
        self.assertAlmostEqual(
            l1_regularizer(torch.tensor(n), 0.01).numpy(),
            tf.keras.regularizers.l1(l1=0.01)(n).numpy(),
            4
        )
        self.assertAlmostEqual(
            l1_regularizer(torch.tensor(n), 0.02).numpy(),
            tf.keras.regularizers.l1(l1=0.02)(n).numpy(),
            4
        )
