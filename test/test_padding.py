import unittest

import numpy as np
import torch

from stg.datasets.padding import pre_pad, pad_feature_first, ZeroPadding


class TestPaddingFunctions(unittest.TestCase):

    def setUp(self):
        # Setup some sample data to be used across tests
        self.single_feature_data = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]])
        ]

    def test_pre_pad_single_sequence(self):
        sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        padded_sequence = pre_pad([sequence], batch_first=True)
        self.assertEqual(padded_sequence.shape, (1, 3, 2))
        self.assertTrue(torch.equal(padded_sequence, sequence.unsqueeze(0)))

    def test_pre_pad_multiple_sequences(self):
        padded_sequence = pre_pad(self.single_feature_data, batch_first=True)
        expected_sequence = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [5.0, 6.0]]
        ])
        self.assertEqual(padded_sequence.shape, (2, 2, 2))
        self.assertTrue(torch.equal(padded_sequence, expected_sequence))

    def test_pre_pad_with_multiple_padding_values(self):
        sequences = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            torch.tensor([[7.0, 8.0]]),
            torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        ]
        padded_sequence = pre_pad(sequences, batch_first=True)
        expected_sequence = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[0.0, 0.0], [0.0, 0.0], [7.0, 8.0]],
            [[0.0, 0.0], [9.0, 10.0], [11.0, 12.0]]
        ])
        self.assertEqual(padded_sequence.shape, (3, 3, 2))
        self.assertTrue(torch.equal(padded_sequence, expected_sequence))

    def test_invalid_padding_type(self):
        with self.assertRaises(ValueError):
            pad_feature_first(self.single_feature_data, padding_type='invalid')

    def test_pre_pad_feature_first_single_feature(self):
        padded_batch = pad_feature_first(self.single_feature_data, padding_type='pre')
        expected_batch = [
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1
                    [3.0, 4.0]  # Sequence 2
                ],  # Sample 1
                [
                    [0.0, 0.0],  # Sequence 1
                    [5.0, 6.0]  # Sequence 2
                ]  # Sample 2
            ])  # Feature 1
        ]
        self.assertEqual(len(padded_batch), 1)
        self.assertEqual(padded_batch[0].shape, (2, 2, 2))
        self.assertTrue(torch.equal(padded_batch[0], expected_batch[0]))

    def test_pre_pad_feature_first_single_feature_multiple_values(self):
        sequences = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            torch.tensor([[7.0, 8.0]]),
            torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        ]
        padded_batch = pad_feature_first(sequences, padding_type='pre')
        expected_batch = [
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1
                    [3.0, 4.0],  # Sequence 2
                    [5.0, 6.0]  # Sequence 3
                ],  # Sample 1
                [
                    [0.0, 0.0],  # Sequence 1
                    [0.0, 0.0],  # Sequence 2
                    [7.0, 8.0]  # Sequence 3
                ],  # Sample 2
                [
                    [0.0, 0.0],  # Sequence 1
                    [9.0, 10.0],  # Sequence 2
                    [11.0, 12.0]  # Sequence 3
                ]  # Sample 3
            ])  # Feature 1
        ]
        self.assertEqual(len(padded_batch), 1)
        self.assertEqual(padded_batch[0].shape, (3, 3, 2))
        self.assertTrue(torch.equal(padded_batch[0], expected_batch[0]))

    def test_post_pad_feature_first_single_feature_multiple_values(self):
        sequences = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            torch.tensor([[7.0, 8.0]]),
            torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        ]
        padded_batch = pad_feature_first(sequences, padding_type='post')
        expected_batch = [
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1
                    [3.0, 4.0],  # Sequence 2
                    [5.0, 6.0]  # Sequence 3
                ],  # Sample 1
                [
                    [7.0, 8.0],  # Sequence 1
                    [0.0, 0.0],  # Sequence 2
                    [0.0, 0.0]  # Sequence 3
                ],  # Sample 2
                [
                    [9.0, 10.0],  # Sequence 1
                    [11.0, 12.0],  # Sequence 2
                    [0.0, 0.0]  # Sequence 3
                ]  # Sample 3
            ])  # Feature 1
        ]
        self.assertEqual(len(padded_batch), 1)
        self.assertEqual(padded_batch[0].shape, (3, 3, 2))
        self.assertTrue(torch.equal(padded_batch[0], expected_batch[0]))

    def test_post_pad_feature_first_single_feature(self):
        padded_batch = pad_feature_first(self.single_feature_data, padding_type='post')
        expected_batch = [
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1
                    [3.0, 4.0]  # Sequence 2
                ],  # Sample 1
                [
                    [5.0, 6.0],  # Sequence 1
                    [0.0, 0.0]  # Sequence 2
                ]  # Sample 2
            ])  # Feature 1
        ]
        self.assertEqual(len(padded_batch), 1)
        self.assertEqual(padded_batch[0].shape, (2, 2, 2))
        self.assertTrue(torch.equal(padded_batch[0], expected_batch[0]))

    def test_pre_pad_feature_first_multiple_features(self):
        sequence = [
            # Sample 1
            [
                torch.tensor([
                    [1.0, 2.0],  # Sequence 1 for Feature 1
                    [3.0, 4.0],  # Sequence 2 for Feature 1
                    [5.0, 6.0]  # Sequence 3 for Feature 1
                ]),
                torch.tensor([
                    [1.0, 2.0, 3.0, 4.0],  # Sequence 1 for Feature 2
                    [5.0, 6.0, 7.0, 8.0],  # Sequence 2 for Feature 2
                    [9.0, 10.0, 11.0, 12.0]  # Sequence 3 for Feature 2
                ]),
                torch.tensor([
                    [1.0],  # Sequence 1 for Feature 3
                    [2.0],  # Sequence 2 for Feature 3
                    [3.0]  # Sequence 3 for Feature 3
                ])
            ],

            # Sample 2
            [
                torch.tensor([
                    [7.0, 8.0]  # Sequence 3 for Feature 1
                ]),
                torch.tensor([
                    [13.0, 14.0, 15.0, 16.0]  # Sequence 3 for Feature 2
                ]),
                torch.tensor([
                    [4.0]  # Sequence 3 for Feature 3
                ])
            ],

            # Sample 3
            [
                torch.tensor([
                    [9.0, 10.0],  # Sequence 2 for Feature 1
                    [11.0, 12.0]  # Sequence 3 for Feature 1
                ]),
                torch.tensor([
                    [17.0, 18.0, 19.0, 20.0],  # Sequence 2 for Feature 2
                    [21.0, 22.0, 23.0, 24.0]  # Sequence 3 for Feature 2
                ]),
                torch.tensor([
                    [5.0],  # Sequence 2 for Feature 3
                    [6.0]  # Sequence 3 for Feature 3
                ])
            ]
        ]

        expected_batch = [
            # Feature 1: Positional data with dimension 2
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1
                    [3.0, 4.0],  # Sequence 2
                    [5.0, 6.0]  # Sequence 3
                ],  # Sample 1
                [
                    [0.0, 0.0],  # Sequence 1 (Padded)
                    [0.0, 0.0],  # Sequence 2 (Padded)
                    [7.0, 8.0]  # Sequence 3
                ],  # Sample 2
                [
                    [0.0, 0.0],  # Sequence 1 (Padded)
                    [9.0, 10.0],  # Sequence 2
                    [11.0, 12.0]  # Sequence 3
                ]  # Sample 3
            ]),

            # Feature 2: Some abstract data with dimension 4
            torch.tensor([
                [
                    [1.0, 2.0, 3.0, 4.0],  # Sequence 1
                    [5.0, 6.0, 7.0, 8.0],  # Sequence 2
                    [9.0, 10.0, 11.0, 12.0]  # Sequence 3
                ],  # Sample 1
                [
                    [0.0, 0.0, 0.0, 0.0],  # Sequence 1 (Padded)
                    [0.0, 0.0, 0.0, 0.0],  # Sequence 2 (Padded)
                    [13.0, 14.0, 15.0, 16.0]  # Sequence 3
                ],  # Sample 2
                [
                    [0.0, 0.0, 0.0, 0.0],  # Sequence 1 (Padded)
                    [17.0, 18.0, 19.0, 20.0],  # Sequence 2
                    [21.0, 22.0, 23.0, 24.0]  # Sequence 3
                ]  # Sample 3
            ]),

            # Feature 3: Speed data with dimension 1
            torch.tensor([
                [
                    [1.0],  # Sequence 1
                    [2.0],  # Sequence 2
                    [3.0]  # Sequence 3
                ],  # Sample 1
                [
                    [0.0],  # Sequence 1 (Padded)
                    [0.0],  # Sequence 2 (Padded)
                    [4.0]  # Sequence 3
                ],  # Sample 2
                [
                    [0.0],  # Sequence 1 (Padded)
                    [5.0],  # Sequence 2
                    [6.0]  # Sequence 3
                ]  # Sample 3
            ])
        ]

        padded_batch = pad_feature_first(sequence, padding_type='pre')
        self.assertEqual(len(padded_batch), 3)
        for i in range(len(padded_batch)):
            self.assertEqual(padded_batch[i].shape, expected_batch[i].shape)
            self.assertTrue(torch.equal(padded_batch[i], expected_batch[i]))

    def test_post_pad_feature_first_multiple_features(self):
        sequence = [
            # Sample 1
            [
                torch.tensor([
                    [1.0, 2.0],  # Sequence 1 for Feature 1
                    [3.0, 4.0],  # Sequence 2 for Feature 1
                    [5.0, 6.0]  # Sequence 3 for Feature 1
                ]),
                torch.tensor([
                    [1.0, 2.0, 3.0, 4.0],  # Sequence 1 for Feature 2
                    [5.0, 6.0, 7.0, 8.0],  # Sequence 2 for Feature 2
                    [9.0, 10.0, 11.0, 12.0]  # Sequence 3 for Feature 2
                ]),
                torch.tensor([
                    [1.0],  # Sequence 1 for Feature 3
                    [2.0],  # Sequence 2 for Feature 3
                    [3.0]  # Sequence 3 for Feature 3
                ])
            ],

            # Sample 2
            [
                torch.tensor([
                    [7.0, 8.0]  # Sequence 3 for Feature 1
                ]),
                torch.tensor([
                    [13.0, 14.0, 15.0, 16.0]  # Sequence 3 for Feature 2
                ]),
                torch.tensor([
                    [4.0]  # Sequence 3 for Feature 3
                ])
            ],

            # Sample 3
            [
                torch.tensor([
                    [9.0, 10.0],  # Sequence 2 for Feature 1
                    [11.0, 12.0]  # Sequence 3 for Feature 1
                ]),
                torch.tensor([
                    [17.0, 18.0, 19.0, 20.0],  # Sequence 2 for Feature 2
                    [21.0, 22.0, 23.0, 24.0]  # Sequence 3 for Feature 2
                ]),
                torch.tensor([
                    [5.0],  # Sequence 2 for Feature 3
                    [6.0]  # Sequence 3 for Feature 3
                ])
            ]
        ]

        expected_batch = [
            # Feature 1: Positional data with dimension 2
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1
                    [3.0, 4.0],  # Sequence 2
                    [5.0, 6.0]  # Sequence 3
                ],  # Sample 1
                [
                    [7.0, 8.0],  # Sequence 1
                    [0.0, 0.0],  # Sequence 2 (Padded)
                    [0.0, 0.0]  # Sequence 3 (Padded)
                ],  # Sample 2
                [
                    [9.0, 10.0],  # Sequence 1
                    [11.0, 12.0],  # Sequence 2
                    [0.0, 0.0]  # Sequence 3 (Padded)
                ]  # Sample 3
            ]),

            # Feature 2: Some abstract data with dimension 4
            torch.tensor([
                [
                    [1.0, 2.0, 3.0, 4.0],  # Sequence 1
                    [5.0, 6.0, 7.0, 8.0],  # Sequence 2
                    [9.0, 10.0, 11.0, 12.0]  # Sequence 3
                ],  # Sample 1
                [
                    [13.0, 14.0, 15.0, 16.0],  # Sequence 1
                    [0.0, 0.0, 0.0, 0.0],  # Sequence 2 (Padded)
                    [0.0, 0.0, 0.0, 0.0]  # Sequence 3 (Padded)
                ],  # Sample 2
                [
                    [17.0, 18.0, 19.0, 20.0],  # Sequence 1
                    [21.0, 22.0, 23.0, 24.0],  # Sequence 2
                    [0.0, 0.0, 0.0, 0.0]  # Sequence
                ]  # Sample 3
            ]),

            # Feature 3: Speed data with dimension 1
            torch.tensor([
                [
                    [1.0],  # Sequence 1
                    [2.0],  # Sequence 2
                    [3.0]  # Sequence 3
                ],  # Sample 1
                [
                    [4.0],  # Sequence 1
                    [0.0],  # Sequence 2 (Padded)
                    [0.0]  # Sequence 3 (Padded)
                ],  # Sample 2
                [
                    [5.0],  # Sequence 1
                    [6.0],  # Sequence 2
                    [0.0]  # Sequence 3 (Padded)
                ]  # Sample 3
            ])
        ]

        padded_batch = pad_feature_first(sequence, padding_type='post')
        self.assertEqual(len(padded_batch), 3)
        for i in range(len(padded_batch)):
            self.assertEqual(padded_batch[i].shape, expected_batch[i].shape)
            self.assertTrue(torch.equal(padded_batch[i], expected_batch[i]))

    def test_post_pad_feature_first_two_features(self):
        sequence = [
            # Sample 1
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1 for Feature 1
                    [3.0, 4.0],  # Sequence 2 for Feature 1
                    [5.0, 6.0]  # Sequence 3 for Feature 1
                ],
                [
                    [7.0, 8.0],  # Sequence 1 for Feature 2
                    [9.0, 10.0],  # Sequence 2 for Feature 2
                    [11.0, 12.0]  # Sequence 3 for Feature 2
                ]
            ]),
            # Sample 2
            torch.tensor([
                [
                    [13.0, 14.0]  # Sequence 3 for Feature 1
                ],
                [
                    [15.0, 16.0]  # Sequence 3 for Feature 2
                ]
            ]),
            # Sample 3
            torch.tensor([
                [
                    [17.0, 18.0],  # Sequence 2 for Feature 1
                    [19.0, 20.0]  # Sequence 3 for Feature 1
                ],
                [
                    [21.0, 22.0],  # Sequence 2 for Feature 2
                    [23.0, 24.0]  # Sequence 3 for Feature 2
                ]
            ])
        ]

        expected_batch = [
            # Feature 1
            torch.tensor([
                [
                    [1.0, 2.0],  # Sequence 1
                    [3.0, 4.0],  # Sequence 2
                    [5.0, 6.0]  # Sequence 3
                ],  # Sample 1
                [
                    [13.0, 14.0],  # Sequence 1
                    [0.0, 0.0],  # Sequence 2 (Padded)
                    [0.0, 0.0]  # Sequence 3 (Padded)
                ],  # Sample 2
                [
                    [17.0, 18.0],  # Sequence 1
                    [19.0, 20.0],  # Sequence 2
                    [0.0, 0.0]  # Sequence 3 (Padded)
                ]  # Sample 3
            ]),
            # Feature 2
            torch.tensor([
                [
                    [7.0, 8.0],  # Sequence 1
                    [9.0, 10.0],  # Sequence 2
                    [11.0, 12.0]  # Sequence 3
                ],  # Sample 1
                [
                    [15.0, 16.0],  # Sequence 1
                    [0.0, 0.0],  # Sequence 2 (Padded)
                    [0.0, 0.0]  # Sequence 3 (Padded)
                ],  # Sample 2
                [
                    [21.0, 22.0],  # Sequence 1
                    [23.0, 24.0],  # Sequence 2
                    [0.0, 0.0]  # Sequence 3 (Padded)
                ]  # Sample 3
            ])
        ]

        padded_batch = pad_feature_first(sequence, padding_type='post')
        self.assertEqual(len(padded_batch), 2)
        for i in range(len(padded_batch)):
            self.assertEqual(padded_batch[i].shape, expected_batch[i].shape)
            self.assertTrue(torch.equal(padded_batch[i], expected_batch[i]))


class TestZeroPadding(unittest.TestCase):

    def test_initialization(self):
        # Test for successful initialization
        zp = ZeroPadding()
        self.assertTrue(isinstance(zp, ZeroPadding))

        # Test for initialization with labels but without return_len
        with self.assertRaises(ValueError):
            ZeroPadding(return_len=True, return_labels=False)

        # Test for initialization with incorrect padding_type
        with self.assertRaises(ValueError):
            ZeroPadding(padding_type='invalid')

    def test_pad_empty_batch(self):
        zp = ZeroPadding()
        with self.assertRaises(ValueError):
            zp.pad([])

    def test_pad_with_return_len(self):
        zp = ZeroPadding(return_len=True, return_labels=True)
        batch = [(torch.tensor([1, 2, 3]), 'a'), (torch.tensor([4, 5]), 'b')]
        padded_batch, lengths, labels = zp.pad(batch)
        self.assertEqual(lengths, [3, 2])
        self.assertTrue(torch.equal(padded_batch[1], torch.tensor([4, 5, 0])))

    def test_pad_with_labels(self):
        zp = ZeroPadding(return_len=True, return_labels=True)
        batch = [(torch.tensor([1, 2, 3]), 'a'), (torch.tensor([4, 5]), 'b')]
        padded_batch, lengths, labels = zp.pad(batch)
        self.assertEqual(labels, ('a', 'b'))
        self.assertTrue(torch.equal(padded_batch[1], torch.tensor([4, 5, 0])))

    def test_pad_with_numpy_arrays(self):
        zp = ZeroPadding()
        batch = [np.array([1, 2, 3]), np.array([4, 5])]
        padded_batch = zp.pad(batch)
        self.assertTrue(torch.equal(padded_batch[1], torch.tensor([4, 5, 0])))

    def test_pre_padding(self):
        zp = ZeroPadding(padding_type='pre')
        batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded_batch = zp.pad(batch)
        self.assertTrue(torch.equal(padded_batch[1], torch.tensor([0, 4, 5])))

    def test_call_method(self):
        zp = ZeroPadding()
        batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded_batch = zp(batch)
        self.assertTrue(torch.equal(padded_batch[1], torch.tensor([4, 5, 0])))

    def test_pad_to_fixed_length(self):
        zp = ZeroPadding(fixed_length=4)
        batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded_batch = zp.pad(batch)
        self.assertTrue(torch.equal(
            padded_batch,
            torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        ))

    def test_prepad_to_fixed_length(self):
        zp = ZeroPadding(fixed_length=4, padding_type='pre')
        batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded_batch = zp.pad(batch)
        self.assertTrue(torch.equal(
            padded_batch,
            torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]])
        ))


if __name__ == '__main__':
    unittest.main()
