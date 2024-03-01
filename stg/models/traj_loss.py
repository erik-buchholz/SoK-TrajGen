#!/usr/bin/env python3
"""Implements trajLoss for LSTM-TrajGAN in PyTorch"""
from typing import List, Union

import torch
from torch import nn


class TrajLoss(nn.Module):
    def __init__(self,
                 p_bce=1,
                 p_latlon=10,
                 p_cat=1,
                 p_dow=1,
                 p_hour=1,
                 ):
        super().__init__()
        self.bce = torch.nn.BCELoss(reduction='mean')
        self.p_bce = p_bce
        self.p_latlon = p_latlon
        self.p_cat = p_cat
        self.p_dow = p_dow
        self.p_hour = p_hour

    def compute_mask(self, real_trajs):
        # Compute mask
        pad_value = 0.0
        mask = (real_trajs[0] != pad_value).any(dim=-1)
        traj_len = mask.sum(dim=-1)
        return mask, traj_len

    def latlon_loss(self, real_trajs, gen_trajs, mask, traj_len):
        latlon = torch.square(gen_trajs[0] - real_trajs[0])
        # Expand the last dimension to two in order to prepare for multiplication
        # (batch_size, sequence) --reshape--> (batch_size, sequence, 1) --expand--> (batch_size, sequence, 2)
        latlon_mask = mask.reshape((*mask.shape, 1)).expand(*mask.shape, 2)
        masked_latlon_full = (latlon_mask * latlon).sum(dim=1).sum(dim=1)
        masked_latlon_mse = (masked_latlon_full / traj_len).sum()
        return masked_latlon_mse

    def hour_loss(self, real_trajs, gen_trajs, mask, traj_len):
        # According to PyTorch Doc, we have to permute the sequences, such that classes are the second
        # dimension and the real labels should not be provided as one hot but rather as class labels
        # Cf. https://stackoverflow.com/questions/72091572/how-to-compute-cross-entropy-loss-for-sequences
        ce_hour = torch.nn.functional.cross_entropy(gen_trajs[1].permute(0, 2, 1), real_trajs[1].argmax(dim=-1),
                                                    reduction='none')
        ce_hour_masked = ce_hour * mask
        ce_hour_mean = (ce_hour_masked / traj_len.reshape(-1, 1)).sum()
        return ce_hour_mean

    def dow_loss(self, real_trajs, gen_trajs, mask, traj_len):
        ce_dow = torch.nn.functional.cross_entropy(gen_trajs[2].permute(0, 2, 1), real_trajs[2].argmax(dim=-1),
                                                   reduction='none')
        ce_dow_masked = ce_dow * mask
        ce_dow_mean = (ce_dow_masked / traj_len.reshape(-1, 1)).sum()
        return ce_dow_mean

    def cat_loss(self, real_trajs, gen_trajs, mask, traj_len):
        ce_category = torch.nn.functional.cross_entropy(gen_trajs[3].permute(0, 2, 1), real_trajs[3].argmax(dim=-1),
                                                        reduction='none')
        ce_category_masked = ce_category * mask
        ce_category_mean = (ce_category_masked / traj_len.reshape(-1, 1)).sum()
        return ce_category_mean

    def bce_loss(self, y_pred, y_true):
        return self.bce(y_pred, y_true)
        # return torch.nn.functional.binary_cross_entropy(y_pred, y_true, reduction='sum')

    def forward(self, y_true, y_pred, real_trajs, gen_trajs, verbose: bool = False, pbar=None):
        mask, traj_len = self.compute_mask(real_trajs)

        # Consider the critic's judgement
        bce_loss = self.bce_loss(y_pred, y_true) if self.p_bce > 0 else 0

        masked_latlon_mse = self.latlon_loss(real_trajs, gen_trajs, mask, traj_len) if self.p_latlon > 0 else 0

        ce_category_mean = self.cat_loss(real_trajs, gen_trajs, mask, traj_len) if self.p_cat > 0 else 0
        ce_dow_mean = self.dow_loss(real_trajs, gen_trajs, mask, traj_len) if self.p_dow > 0 else 0
        ce_hour_mean = self.hour_loss(real_trajs, gen_trajs, mask, traj_len) if self.p_hour > 0 else 0

        if verbose:
            msg = (
                f"Loss components:\t"
                f"LATLON: {self.p_latlon * float(masked_latlon_mse.cpu()):.4f};\t"
                f"BCE: {self.p_bce * float(bce_loss.cpu()):.4f};\t"
                f"CAT: {self.p_cat * float(ce_category_mean.cpu()):.4f};\t"
                f"DOW: {self.p_dow * float(ce_dow_mean.cpu()):.4f};\t"
                f"HOUR: {self.p_hour * float(ce_hour_mean.cpu()):.4f};\t"
            )
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)

        return bce_loss * self.p_bce + masked_latlon_mse * self.p_latlon + \
            ce_category_mean * self.p_cat + ce_dow_mean * self.p_dow + ce_hour_mean * self.p_hour


def get_trajLoss(features: List[str], weight: Union[int, List[int]] = 1) -> TrajLoss:
    """
    Creates a TrajLoss instance with weights adapted to the provided features.
    Weights for features not included in the 'features' list will be set to 0.

    :param features: A list of features to be considered for the loss function.
    :param weight: An optional weight parameter that can either be an integer,
                   which applies the same weight to all features, or a list of integers
                   providing a distinct weight for each feature.
                   If a list is provided, it must be the same length as 'features'.
    :return: An instance of TrajLoss with the specified feature weights.
    """

    # Define default parameters for the TrajLoss constructor.
    loss_params = {'p_bce': 1, 'p_latlon': 10, 'p_cat': 0, 'p_dow': 0, 'p_hour': 0}

    # Define a mapping of feature keys to TrajLoss parameter names.
    feature_params = {
        'category': 'p_cat',
        'hour': 'p_hour',
        'day': 'p_dow'
    }

    # Ensure that the weight is a list with the same length as features, or a single integer.
    if isinstance(weight, int):
        weight_dict = {feature: weight for feature in features}
    elif isinstance(weight, list) and len(weight) == len(features):
        weight_dict = dict(zip(features, weight))
    else:
        raise ValueError("Weight must be an integer or a list of integers with the same length as features.")

    # Update the loss_params with weights for included features.
    for feature in features:
        param = feature_params.get(feature)
        if param:
            loss_params[param] = weight_dict[feature]

    # Instantiate the TrajLoss class with the prepared parameters.
    return TrajLoss(**loss_params)
