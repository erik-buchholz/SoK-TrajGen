#!/usr/bin/env python3
""" """
import unittest

from stg.datasets import DictProperty


class TestDictProperty(unittest.TestCase):

    def setUp(self):
        # Sample creation function
        def creation_fn(key):
            self.parent_dict[key] = f"Generated {key}"

        self.parent_dict = {"a": 1}
        self.keys = ["a", "b", "c"]
        self.dp = DictProperty(self.parent_dict, self.keys, creation_fn)

    def test_get_item(self):
        # Existing item
        self.assertEqual(self.dp["a"], 1)
        # Non-existing but in the allowed keys, should trigger the creation function
        self.assertEqual(self.dp["b"], "Generated b")
        # Not in the allowed keys
        with self.assertRaises(KeyError):
            _ = self.dp["z"]

    def test_set_item(self):
        with self.assertRaises(NotImplementedError):
            self.dp["a"] = 2

    def test_del_item(self):
        with self.assertRaises(NotImplementedError):
            del self.dp["a"]

    def test_len(self):
        self.assertEqual(len(self.dp), 3)

    def test_contains(self):
        # Existing key
        self.assertIn("a", self.dp)
        # Non-existing but in the allowed keys
        self.assertIn("b", self.dp)
        # Not in the allowed keys
        self.assertNotIn("z", self.dp)

    def test_keys(self):
        self.assertSetEqual(self.dp.keys(), {"a", "b", "c"})

    def test_values(self):
        # Since b and c haven't been accessed yet, the creation function hasn't been triggered for them.
        self.assertEqual(set(self.dp.values()), {1, "Generated b", "Generated c"})

    def test_get_method(self):
        # Existing key
        self.assertEqual(self.dp.get("a"), 1)
        # Non-existing but in the allowed keys
        self.assertEqual(self.dp.get("b"), "Generated b")
        # Not in the allowed keys with default
        self.assertEqual(self.dp.get("z", "default"), "default")
        # Not in the allowed keys without default
        self.assertIsNone(self.dp.get("z"))

    def test_pop(self):
        with self.assertRaises(NotImplementedError):
            self.dp.pop("a")
