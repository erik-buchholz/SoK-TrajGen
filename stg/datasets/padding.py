#!/usr/bin/env python3
"""Different Padding implementations for PyTorch"""
from typing import Union, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pre_pad(
        sequences: Union[Tensor, List[Tensor]],
        batch_first: bool = False,
        padding_value: float = 0.0,
) -> Tensor:
    sequences = tuple(map(lambda s: s.flip(0), sequences))
    padded_sequence = pad_sequence(sequences, batch_first, padding_value)
    _seq_dim = padded_sequence.dim()
    padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    return padded_sequence


def pad_feature_first(batch: List[Union[torch.Tensor, List[torch.Tensor]]], padding_type='pre') -> List[torch.Tensor]:
    """
    Pads a batch of variable-length sequences to the same length.
    Considers LSTM-TrajGAN style batches where the second dimension is the number of features.

    :param batch: List of Tensors with shape
                  (batch_size, sequence_length, feature_dim) or
                  List of Lists of Tensors with shape
                  (batch_size, num_features, sequence_length, feature_dim)
                  feature_dim might depend on the feature,e.g.:
                  batch[0][0].shape = (sequence_length, 2), but
                    batch[0][1].shape = (sequence_length, 24), and
                    batch[0][2].shape = (sequence_length, 10)
    :param padding_type: Padding type ('pre' or 'post')
    :return: List[Tensor]: Padded and collated batch with shape:
             (num_features, batch_size, max_sequence_length, feature_dim)
             If the input was of the first form (without explicit num_features),
             the output shape will have num_features = 1.
             feature_dim can be different for each feature. Therefore, the output is a list and not a tensor.
    """

    # Check if the batch consists of single-feature tensors or multi-feature lists of tensors
    if isinstance(batch[0], torch.Tensor) and len(batch[0].shape) == 2:
        # Here, we have single-feature sequences with shape (batch_size, sequence_length, feature_dim)
        # We add an extra dimension to conform to the multi-feature structure
        # New shape: (batch_size, 1, sequence_length, feature_dim)
        batch = [b.unsqueeze(0) for b in batch]

    # Extract dimensions from the batch.
    dimensions = [[b[i] for b in batch] for i in range(len(batch[0]))]

    # Choose the appropriate padding function based on the padding type
    if padding_type == 'pre':
        padding_function = pre_pad
    elif padding_type == 'post':
        padding_function = pad_sequence
    else:
        raise ValueError(f"Unknown padding type {padding_type}.")

    # Pad each dimension separately because they might have sizes in the last dimension
    # This gives us a list of tensors:
    # Each dimension is one tensor with shape (batch_size, max_sequence_length, specific_feature_dim)
    padded_dimensions = [
        padding_function(dimension, batch_first=True, padding_value=0.0) for dimension in dimensions
    ]

    # Collate (gather) the batch in such a way that it's ordered by feature first.
    # Resultant collated_batch will be a list of tensors with shapes:
    # (batch_size, max_sequence_length, feature_dim1),
    # (batch_size, max_sequence_length, feature_dim2), ...
    collated_batch = [
        [padded_dimensions[j][i] for j in range(len(padded_dimensions))] for i in range(len(padded_dimensions[0]))
    ]

    return torch.utils.data.default_collate(collated_batch)


class ZeroPadding:
    def __init__(
            self,
            return_len: bool = False,
            return_labels: bool = False,
            padding_value: float = 0.0,
            padding_type: str = 'post',
            fixed_length: Optional[int] = None,
    ):
        """
        Initialize the ZeroPadding object.

        :param return_len: Whether to return a list containing the original lengths.
        :param return_labels: Whether the batch also contains labels which should be returned.
        :param padding_value: Value to use for padding.
        :param padding_type: Padding type ('pre' or 'post').
        :param fixed_length: If not None, all sequences will be padded to this length.
        """
        if padding_type not in ['pre', 'post']:
            raise ValueError(f"Unknown padding type {padding_type}.")
        if return_len and not return_labels:
            raise ValueError("Cannot return lengths without labels.")
            # Otherwise, it would be unclear whether a tuple of length 2 is (batch, lengths) or (batch, labels).
            # This way, it's always (batch, lengths, labels) or (batch, lengths) or just (batch).
        self.return_len = return_len
        self.return_labels = return_labels
        self.padding_value = padding_value
        self.padding_type = padding_type
        self.fixed_length: int = fixed_length

    def pad(self, batch: List[Union[torch.Tensor, np.ndarray]]) -> Union[torch.Tensor, List]:
        """
        Pad a batch of sequences to the same length with the specified padding value.

        :param batch: A list of sequences (torch.Tensor or numpy.ndarray).
        :return: The padded batch, optionally with original lengths and labels.
        """
        # Error handling for empty batch
        if not batch:
            raise ValueError("Batch is empty.")

        lengths, labels = None, None
        if self.return_labels:
            batch, labels = zip(*batch)

        # Convert numpy arrays to torch tensors if necessary
        batch = [torch.from_numpy(e) if isinstance(e, np.ndarray) else e for e in batch]

        # Get lengths if required
        if self.return_len:
            lengths = [len(e) for e in batch]

        if self.fixed_length is not None:
            # Pad to fixed length: Append sequence of fixed length
            batch.append(torch.ones((self.fixed_length, *batch[0].shape[1:]), dtype=batch[0].dtype))

        # Perform the padding
        if self.padding_type == 'pre':
            batch = pre_pad(batch, batch_first=True, padding_value=self.padding_value)
        elif self.padding_type == 'post':
            batch = pad_sequence(batch, batch_first=True, padding_value=self.padding_value)
        else:
            raise ValueError(f"Unknown padding type {self.padding_type}.")

        if self.fixed_length is not None:
            # Remove the appended sequence
            batch = batch[:-1]

        if self.fixed_length and batch.shape[1] != self.fixed_length:
            raise RuntimeError(
                f"Padding failed. Expected batch.shape[1] == {self.fixed_length}, but got {batch.shape[1]}."
            )

        # Return the necessary information based on the flags
        output = [batch]
        if self.return_len:
            output.append(lengths)
        if self.return_labels:
            output.append(labels)

        return output if len(output) > 1 else output[0]

    def __call__(self, batch: List[Union[torch.Tensor, np.ndarray]]) -> Union[torch.Tensor, List]:
        """
        Make the object callable.

        :param batch: A list of sequences (torch.Tensor or numpy.ndarray).
        :return: The padded batch, optionally with original lengths and labels.
        """
        return self.pad(batch)
