#!/usr/bin/env python3
""" """
import logging
import os
from unittest import TestCase

import torch

# Reduce all the Keras/TensorFlow info messages (only show warning and above)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import numpy as np

import stg.models.traj_loss as pt_loss
from stg.ml_tf import loss as tf_loss

# Deactivate GPU for testing
tf.config.set_visible_devices([], 'GPU')

log = logging.getLogger()
log.setLevel(logging.ERROR)


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        import tensorflow as tf
        self.ptl = pt_loss.TrajLoss()
        self.y_pred = np.array([0.3, 0.3, 0.3])
        self.y_true = np.array([1, 0, 1])
        self.real = [
            np.array([
                [
                    [1, 1],  # t1
                    [2, 2]  # t2
                ],  # sample 1
                [
                    [0, 0],
                    [11, 11]
                ],  # sample 2
                [
                    [1, 9],
                    [2, 8]
                ]  # sample 3
            ]),  # latlon
            np.array([
                [
                    np.eye(24)[9],  # t1
                    np.eye(24)[10]  # t2
                ],  # sample 1
                [
                    np.eye(24)[11],
                    np.eye(24)[12]
                ],  # sample 2
                [
                    np.eye(24)[13],
                    np.eye(24)[14]
                ]  # sample 3
            ]),  # hour
            np.array([
                [
                    np.eye(7)[5],  # t1
                    np.eye(7)[5]  # t2
                ],  # sample 1
                [
                    np.eye(7)[4],
                    np.eye(7)[4]
                ],  # sample 2
                [
                    np.eye(7)[6],
                    np.eye(7)[6]
                ]  # sample 3
            ]),  # dow
            np.array([
                [
                    np.eye(10)[0],  # t1
                    np.eye(10)[1]  # t2
                ],  # sample 1
                [
                    np.eye(10)[1],
                    np.eye(10)[2]
                ],  # sample 2
                [
                    np.eye(10)[8],
                    np.eye(10)[9]
                ]  # sample 3
            ]),  # cat
        ]
        self.gen = [
            np.array([
                [
                    [8, 8],  # t1
                    [8, 8]  # t2
                ],  # sample 1
                [
                    [10, 10],
                    [12, 12]
                ],  # sample 2
                [
                    [5, 5],
                    [5, 5]
                ]  # sample 3
            ]),  # latlon
            np.array([
                [
                    np.eye(24)[11],  # t1
                    np.eye(24)[11]  # t2
                ],  # sample 1
                [
                    np.eye(24)[5],
                    np.eye(24)[5]
                ],  # sample 2
                [
                    np.eye(24)[5],
                    np.eye(24)[5]
                ]  # sample 3
            ]),  # hour
            np.array([
                [
                    np.eye(7)[6],  # t1
                    np.eye(7)[5]  # t2
                ],  # sample 1
                [
                    np.eye(7)[4],
                    np.eye(7)[4]
                ],  # sample 2
                [
                    np.eye(7)[2],
                    np.eye(7)[2]
                ]  # sample 3
            ]),  # dow
            np.array([
                [
                    np.eye(10)[0],  # t1
                    np.eye(10)[0]  # t2
                ],  # sample 1
                [
                    np.eye(10)[0],
                    np.eye(10)[0]
                ],  # sample 2
                [
                    np.eye(10)[0],
                    np.eye(10)[0]
                ]  # sample 3
            ]),  # cat
        ]
        self.real_pt = [torch.tensor(x, dtype=torch.float32) for x in self.real]
        self.gen_pt = [torch.tensor(x, dtype=torch.float32) for x in self.gen]
        self.real_tf = [tf.constant(x, dtype=tf.float32) for x in self.real]
        self.gen_tf = [tf.constant(x, dtype=tf.float32) for x in self.gen]
        self.tf_mask, self.tf_len = tf_loss.compute_mask_trajLoss(self.real_tf)
        self.pt_mask, self.pt_len = self.ptl.compute_mask(self.real_pt)

    def test_mask(self):
        import tensorflow as tf
        tf_mask, tf_len = tf_loss.compute_mask_trajLoss(self.real_tf)
        pt_mask, pt_len = self.ptl.compute_mask(self.real_pt)
        np.testing.assert_allclose(
            tf_mask,
            pt_mask
        )
        tf_len = tf.squeeze(tf_len)
        np.testing.assert_allclose(
            tf_len,
            pt_len
        )

    def test_trajLoss(self):
        tfs = tf_loss.trajLoss(
            y_true=self.y_true,
            y_pred=self.y_pred,
            real_traj=self.real_tf,
            gen_traj=self.gen_tf,
            use_regularizer=True
        )
        pts = self.ptl(
            torch.tensor(self.y_true, dtype=torch.float32),
            torch.tensor(self.y_pred, dtype=torch.float32),
            self.real_pt,
            self.gen_pt
        )
        log.debug(tfs, pts)
        self.assertEqual(
            tfs,
            pts
        )

    def test_bce_loss(self):
        tf_bce = tf_loss.compute_bce(self.y_true, self.y_pred, use_regularizer=True)
        pt_bce = self.ptl.bce_loss(
            torch.tensor(self.y_pred, dtype=torch.float32),
            torch.tensor(self.y_true, dtype=torch.float32)
        )
        log.debug(tf_bce, pt_bce)
        self.assertAlmostEqual(
            tf_bce.numpy(),
            pt_bce.numpy(),
            6
        )

    def test_latlon_loss(self):
        tf_latlon = tf_loss.compute_latlon_loss(
            real_traj=self.real_tf,
            gen_traj=self.gen_tf,
            mask=self.tf_mask,
            traj_length=self.tf_len
        )
        pt_latlon = self.ptl.latlon_loss(
            real_trajs=self.real_pt,
            gen_trajs=self.gen_pt,
            mask=self.pt_mask,
            traj_len=self.pt_len
        )
        log.debug(tf_latlon, pt_latlon)
        self.assertAlmostEqual(
            tf_latlon,
            pt_latlon
        )

    def test_cat_loss(self):
        tf_cat = tf_loss.compute_cat_loss(
            real_traj=self.real_tf,
            gen_traj=self.gen_tf,
            mask=self.tf_mask,
            traj_length=self.tf_len
        )
        pt_cat = self.ptl.cat_loss(
            real_trajs=self.real_pt,
            gen_trajs=self.gen_pt,
            mask=self.pt_mask,
            traj_len=self.pt_len
        )
        log.debug(tf_cat, pt_cat)
        self.assertAlmostEqual(
            tf_cat,
            pt_cat
        )

    def test_hour_loss(self):
        tf_hour = tf_loss.compute_hour_loss(
            real_traj=self.real_tf,
            gen_traj=self.gen_tf,
            mask=self.tf_mask,
            traj_length=self.tf_len
        )
        pt_hour = self.ptl.hour_loss(
            real_trajs=self.real_pt,
            gen_trajs=self.gen_pt,
            mask=self.pt_mask,
            traj_len=self.pt_len
        )
        log.debug(tf_hour, pt_hour)
        self.assertAlmostEqual(
            tf_hour,
            pt_hour
        )

    def test_day_loss(self):
        tf_dow = tf_loss.compute_dow_loss(
            real_traj=self.real_tf,
            gen_traj=self.gen_tf,
            mask=self.tf_mask,
            traj_length=self.tf_len
        ).numpy()
        pt_dow = self.ptl.dow_loss(
            real_trajs=self.real_pt,
            gen_trajs=self.gen_pt,
            mask=self.pt_mask,
            traj_len=self.pt_len
        ).numpy()
        log.debug(tf_dow, pt_dow)
        self.assertAlmostEqual(
            tf_dow,
            pt_dow,
            6
        )


class TestGetTrajLoss(TestCase):

    def test_with_all_features_and_uniform_weight(self):
        features = ['category', 'hour', 'day']
        weight = 2
        expected_params = {'p_bce': 1, 'p_latlon': 10, 'p_cat': 2, 'p_dow': 2, 'p_hour': 2}

        loss = pt_loss.get_trajLoss(features, weight)

        for param, value in expected_params.items():
            with self.subTest(param=param, value=value):
                self.assertEqual(getattr(loss, param), value)

    def test_with_missing_features_and_individual_weights(self):
        features = ['category', 'day']
        weights = [3, 4]
        expected_params = {'p_bce': 1, 'p_latlon': 10, 'p_cat': 3, 'p_dow': 4, 'p_hour': 0}

        loss = pt_loss.get_trajLoss(features, weights)

        for param, value in expected_params.items():
            with self.subTest(param=param, value=value):
                self.assertEqual(getattr(loss, param), value)

    def test_with_incorrect_weights_length_raises_value_error(self):
        features = ['category', 'hour', 'day']
        weights = [1, 2]  # Incorrect length

        with self.assertRaises(ValueError):
            pt_loss.get_trajLoss(features, weights)

    def test_with_no_features_should_set_all_to_zero_except_defaults(self):
        features = []
        expected_params = {'p_bce': 1, 'p_latlon': 10, 'p_cat': 0, 'p_dow': 0, 'p_hour': 0}

        loss = pt_loss.get_trajLoss(features)

        for param, value in expected_params.items():
            with self.subTest(param=param, value=value):
                self.assertEqual(getattr(loss, param), value)
