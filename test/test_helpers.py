#!/usr/bin/env python3
""" """
from unittest import TestCase

from stg.utils import helpers


class Test(TestCase):
    def test_compute_inverse_power(self):
        self.assertEqual(
            5,
            helpers.compute_inverse_power(1 / 60000)
        )
        self.assertEqual(
            8,
            helpers.compute_inverse_power(1 / 100000000)
        )

    def test_compute_delta(self):
        self.assertLessEqual(
            helpers.compute_delta(50000),
            1 / 50000 ** 1.1
        )
        self.assertLessEqual(
            helpers.compute_delta(55555),
            1 / 50000 ** 1.1
        )
        self.assertLessEqual(
            helpers.compute_delta(100000),
            1 / 100000 ** 1.1,
        )
        self.assertLessEqual(
            helpers.compute_delta(60000) ** 1.1,
            1e-5
        )
