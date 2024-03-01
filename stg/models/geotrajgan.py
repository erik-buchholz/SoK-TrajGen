"""
# GeoTrajGAN

This model is heavily based on GeoPointGAN [1].
However, we tried to adapt the model to generate trajectories instead of point clouds.

[1] T. Cunningham, K. Klemmer, H. Wen, and H. Ferhatosmanoglu,
“GeoPointGAN: Synthetic Spatial Data with Local Label Differential Privacy.”
arXiv, May 18, 2022. doi: 10.48550/arXiv.2205.08886.
"""
import logging
from contextlib import nullcontext
from datetime import datetime
from typing import List, Optional, Union

import torch
import torch.nn as nn
from IPython import display
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tabulate import tabulate
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_ntbk

from stg.datasets import Datasets
from stg.models.base_gan import BaseGAN
from stg.models.layers import Activation, Norm, CustomLinear, NormOption, ActivationOption
from stg.models.utils import get_rnn, merge_bilstm_output, compute_mask_from_lengths, MergeMode, Optimizer, \
    validate_loss_options, get_optimizer, prepare_param_path, split_data, combine_losses, LossCombination, \
    compute_gradient_penalty, NullWriter, clip_discriminator_weights, visualize_mnist_sequential, \
    visualize_trajectory_samples
from stg.utils.helpers import count_parameters_torch

log = logging.getLogger()

DEFAULT_NORM = NormOption.LAYER  # Original GeoPointGAN: batch1d
DEFAULT_ACTIVATION = ActivationOption.RELU  # Original GeoPointGAN: RelU
MAX_PHYSICAL_BATCH_SIZE = 512


class STNkd(nn.Module):
    """Spatial Transformer Network for k-dimensional points"""

    def __init__(self, k: int = 2, norm: NormOption = DEFAULT_NORM, activation: ActivationOption = DEFAULT_ACTIVATION,
                 conv_only: bool = False):
        """
        Spatial Transformer Network for k-dimensional points
        :param k: Dimensionality of the points
        :param norm: Type of normalization, see NORM_OPTIONS
        :param activation: Type of activation, see ACTIVATION_OPTIONS
        :param conv_only: Replace all fully connected layers with 1D convolutions with kernel size 1
        """
        mode = 'conv_only' if conv_only else 'standard'
        log.debug(f"STN runs in {mode} mode and uses {norm} normalization and {activation} as activation.")
        self.conv_only = conv_only
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        if conv_only:
            # Replace all fully connected layers with conv1D layers with kernel size 1
            # This performs basically the same operation as a fully connected layer, but
            # requires the input to have shape (N, C, L) instead of (N, C) and is initialized differently.
            self.fc1 = nn.Conv1d(1024, 512, 1)
            self.fc2 = nn.Conv1d(512, 256, 1)
            self.fc3 = nn.Conv1d(256, 128, 1)
            self.fc4 = nn.Conv1d(128, k * k, 1)
        else:
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, k * k)
        self.activation = Activation(mode=activation)

        self.norm1 = Norm(32, norm_type=norm)
        self.norm2 = Norm(64, norm_type=norm)
        self.norm3 = Norm(128, norm_type=norm)
        self.norm4 = Norm(512, norm_type=norm)
        self.norm5 = Norm(1024, norm_type=norm)
        # The following Norms might either be called after 1DConv or Linear.
        # Hence, input shape is important.
        self.norm6 = Norm(512, norm_type=norm, channels_last=(not conv_only))
        self.norm7 = Norm(256, norm_type=norm, channels_last=(not conv_only))
        self.norm8 = Norm(128, norm_type=norm, channels_last=(not conv_only))

        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Spatial Transformer Network
        :param x: Expects x.shape = (N, k, L) for batch size N, sequence length L, and num features k
        :return: x with x.shape = (N, k*k, L)
        """
        # noinspection PyArgumentList
        log.debug(f"Entering STN Module with: {x.shape}, x.stride: {x.stride()}")
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.activation(self.norm3(self.conv3(x)))
        x = self.activation(self.norm4(self.conv4(x)))
        x = self.activation(self.norm5(self.conv5(x)))

        # x = torch.max(x, 2, keepdim=True)[0]  # I think this line is obsolete

        if not self.conv_only:
            x = x.transpose(2, 1)

        x = self.activation(self.norm6(self.fc1(x)))
        x = self.activation(self.norm7(self.fc2(x)))
        x = self.activation(self.norm8(self.fc3(x)))
        x = self.fc4(x)  # Output Shape: (N, k*k, L) for conv_only=True and (N, L, k*k) for conv_only=False
        log.debug(f"Last Layer: {x.shape}")

        if not self.conv_only:
            x = x.transpose(2, 1)
        # x.shape = (N, k*k, L) for both cases

        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).unsqueeze(-1)
        # identity.shape = (1, k*k, 1)
        x = x + identity.expand_as(x)

        log.debug(f"Leaving STN Module with: {x.shape}, x.stride: {x.stride()}")
        return x


class TrajNetfeat(nn.Module):

    def __init__(self, code_nfts: int = 2048, n_dim: int = 2, stn: bool = True, norm: NormOption = DEFAULT_NORM,
                 activation: ActivationOption = DEFAULT_ACTIVATION, conv_only: bool = False):
        """
        TrajNet encoder
        :param code_nfts: Latent feature dimensionality at bottleneck / code layer
        :param n_dim: Feature dimensionality of the input points
        :param stn: Use Spatial Transformer Network
        :param norm: Which normalization to use (batch1d, layer, none)
        :param activation: Which activation to use (relu, leaky_relu, elu, tanh, none)
        :param conv_only: Replace Linear Layers with Conv1D Layers
        :returns: output.shape = (N, L, code_nfts) for batch size N, sequence length L, and num features k
        """
        log.debug(f"TrajNet uses {norm} normalization and {activation} as activation.")
        super(TrajNetfeat, self).__init__()
        self.n_dim = n_dim
        self.use_stn = stn

        if stn:
            self.stn = STNkd(k=n_dim, norm=norm, activation=activation, conv_only=conv_only)
        self.code_nfts = code_nfts
        self.conv1 = torch.nn.Conv1d(n_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, code_nfts, 1)

        self.norm1 = Norm(64, norm_type=norm)
        self.norm2 = Norm(128, norm_type=norm)
        self.norm3 = Norm(code_nfts, norm_type=norm)
        self.activation = Activation(mode=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Expects x.shape = (N, L/1, n_dim)
        :return: shape = (N, L, code_nfts)
        """
        log.debug(f"Entering TrajNet with: {x.shape}")

        # x.shape = (N, L, n_dim)
        if self.use_stn:
            batch_size, L, n_dim = x.shape
            stn_out = self.stn(x.transpose(2, 1))
            log.debug(f"stn_out.shape = {stn_out.shape}, stn_out.stride = {stn_out.stride()}")
            stn_out = stn_out.view(batch_size, self.n_dim, self.n_dim, L)  # stn_out.shape = (N, 2, 2, L)
            # noinspection PyArgumentList
            log.debug(f"x.shape = {x.shape}, x.stride = {x.stride()}")
            # x = x.view(-1, 1, self.n_dim)  # x.shape = (N * L, 1, 2)
            log.debug(f"Multiplying: x = {x.shape}, stn_out = {stn_out.shape}")
            # x = torch.bmm(x, stn_out)  # matrix multiplication (N * L, 1, 2) x (N * L, 2, 2) = (N * L, 1, 2)
            x = torch.einsum('bli,bikl->blk', x, stn_out)  # Allows to remove reshape
            log.debug(f"Multiplication Result: {x.shape}")  # x.shape = (N, L, 2)
            # x = x.view(batch_size, L, n_dim)
        x = x.transpose(2, 1)  # x.shape = (N, 2, L)
        log.debug(f"Transposed Result: {x.shape}")
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))

        output = x.transpose(2, 1)  # output.shape = (N, L, code_nfts)
        log.debug(f"Leaving TrajNet with: {output.shape}")
        return output


class TrajNet_Generator(nn.Module):
    """ 
        Generator with TrajNet Encoder, MLP Decoder
    """

    def __str__(self):
        """
        Write generator's parameters as table
        :return:
        """
        table = [["n_dim", self.out_dim], ["Uses PointNet", self.stn], ["Code Size", self.code_nfts],
                 ["Normalization", self.norm], ["Activation", str(self.activation)], ["Mode", self.mode],
                 ["Sequential Mode", self.sequential], ["Use LSTM", self.use_lstm], ]
        if self.use_lstm:
            table.extend([["LSTM Latent Dim", self.lstm_latent_dim], ["Bidirectional LSTM", self.bi_lstm],
                          ["Merge Mode", self.merge_mode], ])
        table.append(["Parameters", f"{count_parameters_torch(self, False):,}"])
        return tabulate(table, tablefmt="github", headers=["Parameter", "Value"])

    def __init__(self, code_nfts: int = 2048, n_dim: int = 2, stn=True, norm: NormOption = DEFAULT_NORM,
                 activation: ActivationOption = DEFAULT_ACTIVATION, conv_only: bool = False, sequential: bool = False,
                 use_lstm: bool = False, bi_lstm: bool = True, merge_mode: MergeMode = 'sum',
                 lstm_latent_dim: int = 100):
        """
        TrajNet Generator
        :param code_nfts: Latent space size
        :param n_dim: Input dimensionality
        :param stn: Use Spatial Transformer Network
        :param norm: Which normalization to use (batch1d, layer, none)
        :param activation: Which activation to use (relu, leaky_relu, elu, tanh, none)
        :param conv_only: Replace Linear Layers with Conv1D Layers
        :param sequential: Use sequential model
        :param use_lstm: Use LSTM
        :param bi_lstm: Use BiLSTM if LSTM is used
        :param merge_mode: How to merge the output of the LSTM (sum, concat, average, mul)
        :param lstm_latent_dim: Latent dimensionality of the LSTM
        """
        super(TrajNet_Generator, self).__init__()

        # Parameters
        self.code_nfts = code_nfts
        self.n_dim = n_dim
        self.out_dim = n_dim
        self.stn = stn
        self.activation = Activation(mode=activation)
        self.norm = norm
        self.mode = 'linear' if not conv_only else 'conv1d'
        self.sequential = sequential
        self.use_lstm = use_lstm
        self.noise_dim = 2
        self.bi_lstm = bi_lstm
        self.merge_mode = merge_mode
        self.lstm_latent_dim = lstm_latent_dim

        # Layers in order
        if self.use_lstm:
            self.n_dim = self.lstm_latent_dim
            self.rnn = get_rnn('lstm', self.noise_dim, self.n_dim, bidirectional=bi_lstm)
        else:
            self.rnn = None
        self.trajnet = TrajNetfeat(code_nfts, n_dim=self.n_dim, stn=stn, norm=norm, activation=activation,
                                   conv_only=conv_only)
        self.encoder = nn.Sequential(CustomLinear(code_nfts, code_nfts, mode=self.mode),
                                     Norm(code_nfts, norm_type=norm, channels_last=(not conv_only)), self.activation)
        self.decoder = nn.Sequential(CustomLinear(code_nfts, code_nfts, mode=self.mode),
                                     Norm(code_nfts, norm_type=norm, channels_last=(not conv_only)), self.activation,
                                     CustomLinear(code_nfts, code_nfts // 2, mode=self.mode),
                                     Norm(code_nfts // 2, norm_type=norm, channels_last=(not conv_only)),
                                     self.activation,
                                     CustomLinear(code_nfts // 2, self.out_dim, mode=self.mode), nn.Tanh())

        # Plotable Gradients for Tensorboard
        self.plotable_gradients = {"Encoder": self.encoder[0].weight, "Output": self.decoder[-2].weight}
        # Add LSTM Gradients
        # Note we only plot the forward pass weights for BiLSTM
        if self.rnn is not None:
            self.plotable_gradients["LSTM_hh"] = self.rnn.weight_hh_l0
            self.plotable_gradients["LSTM_ih"] = self.rnn.weight_ih_l0

        log.info("Generator:\n" + str(self))

    def forward(self, x, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        """
        :param x: (N, self.noise_dim = 2, 1) or (N, seq_len, self.noise_dim)
        :param lengths: Lengths of the sequences in x
        :return: (N, L, self.out_dim)
        """
        log.debug("#" * 80)
        log.debug(f"Generator received: {x.shape}")  # x.shape = (N, seq_len, self.noise_dim)

        if self.use_lstm:
            # Add LSTM & packed padding
            if lengths is not None:
                x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
            x, _ = self.rnn(x)  # x.shape = (N, seq_len, self.n_dim = 2)
            if lengths is not None:
                x, output_lengths = pad_packed_sequence(x, batch_first=True)
            if self.bi_lstm:
                x = merge_bilstm_output(x, mode=self.merge_mode)
            log.debug(f"LSTM output: {x.shape}")
        elif lengths is not None:
            # Manual Masking of Padding values instead of using Packed Padding
            mask = compute_mask_from_lengths(x, lengths)
            # Pre-Masking --> Prevent padding values to influence computation & Gradients
            x = x * mask.unsqueeze(-1)

        if not self.sequential:
            # Flatten. We need a reshape in case the LSTM is used because the LSTM processes the points
            # one after another and not in parallel
            x = x.reshape(-1, 1, self.n_dim)  # x.shape = (N * seq_len, 1, self.n_dim)

        # TrajNet
        log.debug(f"Entering TrajNet: x.shape = {x.shape}")
        # input.shape = (N, L, n_dim), if self.sequential, or (N * L, 1, n_dim) if not self.sequential
        x = self.trajnet(x)  # output.shape = (N, L, code_nfts) or (N * L, 1, code_nfts)

        if self.mode == 'conv1d':
            # Put lengths last
            x = x.transpose(2, 1)  # x.shape = (N, code_nfts, L)

        # Encoder
        log.debug(f"Entering Encoder: x.shape = {x.shape}")
        code = self.encoder(x)

        # Decoder
        log.debug(f"Entering Decoder: code.shape = {code.shape}")
        x = self.decoder(code)

        if self.mode == 'conv1d':
            x = x.transpose(2, 1)  # x.shape = (N, L, self.out_dim)
        # else: x.shape = (N * L, 1, self.out_dim)

        log.debug(f"Generator returned: {x.shape}")
        log.debug("#" * 80)
        # Make x contiguous to detect errors through bad reshaping early
        return x.contiguous()  # , code


class PointDiscriminator(nn.Module):

    def __init__(self, n_dim: int = 2, code_nfts: int = 256, stn: bool = True, norm: NormOption = DEFAULT_NORM,
                 activation: ActivationOption = DEFAULT_ACTIVATION, conv_only: bool = False):
        super().__init__()
        # Set Parameters
        self.n_dim = n_dim
        self.code_nfts = code_nfts
        self.stn = stn
        self.activation = Activation(mode=activation)
        self.norm = norm
        self.mode = 'linear' if not conv_only else 'conv1d'

        # Define Layers in order
        self.point_cls = nn.Sequential(CustomLinear(code_nfts, code_nfts, mode=self.mode),
                                       Norm(code_nfts, norm_type=norm, channels_last=(not conv_only)), self.activation,
                                       CustomLinear(code_nfts, 1, mode=self.mode), )

        self.plotable_gradients = {"Linear": self.point_cls[0].weight, "Output": self.point_cls[-1].weight}

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if self.mode == 'conv1d':
            x = x.transpose(2, 1)  # x.shape = (N, code_nfts, seq_len)
        x = self.point_cls(x)  # x.shape = (N, seq_len, 1)

        # Output Mask
        if mask is not None:
            # Post-Masking --> Mask outputs
            x = x.view(-1, 1) * mask.view(-1, 1)
        else:
            x = x.view(-1, 1)
        # x.shape = (N * seq_len, 1)

        return x


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, norm: NormOption = DEFAULT_NORM,
                 activation: ActivationOption = DEFAULT_ACTIVATION,
                 bi_lstm: bool = True, stn: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.norm = norm
        self.bi_lstm = bi_lstm
        self.activation = Activation(mode=activation)
        self.use_stn = stn

        # Define Layers in order
        if self.use_stn:
            self.stn = STNkd(k=self.input_dim, norm=norm, activation=activation)
        self.traj_rnn = get_rnn('lstm', input_dim, self.latent_dim, bidirectional=self.bi_lstm)
        self.sequence_cls = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim),
                                          Norm(self.latent_dim, norm_type=self.norm, channels_last=True),
                                          self.activation,
                                          nn.Linear(self.latent_dim, 1), )

        self.plotable_gradients = {"LSTM_hh": self.traj_rnn.weight_hh_l0, "LSTM_ih": self.traj_rnn.weight_ih_l0}

    def forward(self, x: torch.Tensor, lengths: List[int] or torch.Tensor = None):
        # x.shape = (N, seq_len, input_dim)
        if self.use_stn:
            batch_size, L, F = x.shape
            stn = self.stn(x.transpose(2, 1))
            log.debug(f"stn.shape = {stn.shape}, stn.stride = {stn.stride()}")
            stn = stn.view(batch_size, F, F, L)
            log.debug(f"Multiplying: x = {x.shape}, stn_out = {stn.shape}")
            x = torch.einsum('bli,bikl->blk', x, stn)  # Allows to remove reshape
            log.debug(f"Multiplication Result: {x.shape}")  # x.shape = (N, L, F)

        if lengths is not None:
            y = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        else:
            y = x
        _, (final_hidden_state, _) = self.traj_rnn(y)
        if self.bi_lstm:
            # Merge hidden states of both directions
            final_hidden_state = torch.sum(final_hidden_state, dim=0)
        else:
            final_hidden_state = final_hidden_state.view(-1, self.latent_dim)
        log.debug(f"[DIS] Final Hidden State: {final_hidden_state.shape}")  # (N, code_nfts)
        output = self.sequence_cls(final_hidden_state)
        log.debug(f"[DIS] Sequential Feedback: {output.shape}")
        return output


class TrajNet_Discriminator(nn.Module):

    def __str__(self) -> str:
        """
        Write discriminator's parameters as table
        :return: str
        """
        table = [["n_dim", self.n_dim],
                 ["Uses STN", self.stn],
                 ["Code Size", self.code_nfts],
                 ["Normalization", self.norm],
                 ["Activation", str(self.activation)],
                 ["Mode", self.mode],
                 ["Sequential Mode", self.sequential],
                 ["LSTM at Start", self.use_lstm],
                 ["Trajectory Critic", self.traj_feedback],
                 ["Trajectory STN", self.traj_stn]
                 ]
        if self.traj_feedback:
            table.append(
                ["Shared PointNet", self.share_pointnet]
            )
        if self.use_lstm or self.traj_feedback:
            table.extend([
                ["LSTM Latent Dim", self.lstm_latent_dim],
                ["Bidirectional LSTM", self.bi_lstm],
                ["Merge Mode", self.merge_mode],
            ])
        table.extend(
            [
                ["Parameter Count:", f"{count_parameters_torch(self, False):,}"]
            ]
        )
        return tabulate(table, tablefmt="github", headers=["Parameter", "Value"])

    def __init__(self, code_nfts: int = 2048, n_dim: int = 2, stn: bool = True, norm: NormOption = DEFAULT_NORM,
                 activation: ActivationOption = DEFAULT_ACTIVATION, conv_only: bool = False, sequential: bool = False,
                 use_lstm: bool = False, bi_lstm: bool = True, merge_mode: MergeMode = MergeMode.SUM,
                 per_traj_feedback: bool = False, share_pointnet: bool = True, lstm_latent_dim: int = None,
                 traj_stn: bool = False):
        """
        :param code_nfts: Latent space size
        :param n_dim: Feature size per point
        :param stn: Use Spatial Transformer Network for the pointnet
        :param norm: Type of normalization to use
        :param activation: Activation function to use (ReLU, LeakyReLU, Tanh)
        :param conv_only: Replace all linear layers with 1D convolutions
        :param sequential: Run discriminator in sequential mode
        :param use_lstm: Run the input through an LSTM before the per-point discriminator
        :param bi_lstm: Use a bidirectional LSTM if any LSTM is used
        :param merge_mode: How to merge the output of the LSTM (sum, concat, average, mul)
        :param per_traj_feedback: Provide a critique per trajectory in addition to the point feedback
        :param share_pointnet: Share the weights of the pointnet between the per-point and per-trajectory feedback
        :param lstm_latent_dim: Latent dimensionality of the LSTM (default: code_nfts)
        :param traj_stn: Use a Spatial Transformer Network for the per-trajectory feedback
        """
        super(TrajNet_Discriminator, self).__init__()

        # Set parameters
        self.n_dim = n_dim
        self.stn = stn
        self.traj_stn = traj_stn
        self.code_nfts = code_nfts
        self.activation = Activation(mode=activation)
        self.norm = norm
        self.mode = 'linear' if not conv_only else 'conv1d'
        self.sequential = sequential
        self.use_lstm = use_lstm
        self.lstm_latent_dim = code_nfts if lstm_latent_dim is None else lstm_latent_dim
        self.bi_lstm = bi_lstm
        self.merge_mode = merge_mode
        self.traj_feedback = per_traj_feedback
        self.share_pointnet = share_pointnet
        if self.traj_feedback and (use_lstm or not self.sequential):
            raise ValueError("[DIS] Per-Traj Feedback not supported with LSTM or Point mode.")

        if self.use_lstm:
            log.warning("[DIS] Using LSTM at the start of the Discriminator. This is deprecated.")
            self.rnn = get_rnn('lstm', self.n_dim, self.n_dim, bidirectional=self.bi_lstm)
        else:
            self.rnn = None

        # TrajNet - Potentially Shared
        self.trajNet = TrajNetfeat(code_nfts=self.code_nfts, n_dim=self.n_dim, stn=self.stn, norm=norm,
                                   activation=activation, conv_only=conv_only)

        # Mandatory Path 1 - Per Point Critique
        self.point_dis = PointDiscriminator(n_dim=self.n_dim, code_nfts=self.code_nfts, stn=self.stn,
                                            norm=self.norm, activation=activation, conv_only=conv_only)

        # Optional Path 2 - Per Trajectory Critique
        if self.traj_feedback:
            if self.share_pointnet:
                log.warning("[DIS] Sharing PointNet between per-point and per-trajectory feedback.")
                lstm_input_dim = self.code_nfts
            else:
                log.warning("[DIS] LSTM for per-trajectory feedback entirely separate from per-point feedback.")
                lstm_input_dim = self.n_dim
            self.traj_dis = TrajectoryDiscriminator(input_dim=lstm_input_dim, latent_dim=self.lstm_latent_dim,
                                                    norm=self.norm, activation=activation, bi_lstm=self.bi_lstm,
                                                    stn=self.traj_stn)
        else:
            self.traj_dis = None

        # Layers for plotting gradients in Tensorboard
        self.plotable_gradients = self.point_dis.plotable_gradients
        # Add the TrajCritic gradients if used
        if self.traj_feedback:
            self.plotable_gradients.update(self.traj_dis.plotable_gradients)

        log.info("Discriminator:\n" + str(self))

    def forward(self, x, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        """

        :param x: x.shape = (batch_size, sequence_length, # features)
        :param lengths: List containing the real length of all sequences if padding is used
        :return:
        """
        log.debug("#" * 80)
        log.debug(f"Discriminator received: {x.shape}")

        mask = None

        if self.use_lstm:
            # Add LSTM & packed padding
            if lengths is not None:
                x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
            x, _ = self.rnn(x)  # x.shape = (N, seq_len, self.n_dim = 2)
            if lengths is not None:
                x, output_lengths = pad_packed_sequence(x, batch_first=True)
            if self.bi_lstm:
                x = merge_bilstm_output(x, mode=self.merge_mode)
            log.debug(f"[DIS] LSTM output: {x.shape}")
        elif lengths is not None:
            # Manual Masking of Padding values instead of using Packed Padding
            mask = compute_mask_from_lengths(x, lengths)
            # Pre-Masking --> Prevent padding values to influence computation & Gradients
            x = x * mask.unsqueeze(-1)

        # Flatten: Reshape instead of view required b/c of LSTM
        if not self.sequential:
            x = x.reshape(-1, 1, self.n_dim)  # x.shape = (N * seq_len, 1, self.n_dim = 2)
            log.debug(f"[DIS] After Flatten: {x.shape}")

        # TrajNet might be shared
        trajNet_encoding = self.trajNet(x)  # x.shape = (N, seq_len, code_nfts)
        log.debug(f"[DIS] After trajNet: {trajNet_encoding.shape}")

        # 1. Per Point Critique
        point_classification = self.point_dis(trajNet_encoding, mask=mask)

        # 2. Per Trajectory Critique
        if self.traj_feedback:
            tcls_in = trajNet_encoding if self.share_pointnet else x
            traj_classification = self.traj_dis(tcls_in, lengths=lengths)
            output = point_classification, traj_classification
            log.debug(f"[DIS] Returning {point_classification.shape} and {traj_classification.shape}.")
        else:
            output = point_classification
            log.debug(f"[DIS] Returning {point_classification.shape}.")

        log.debug("#" * 80)
        return output


class GeoTrajGAN(BaseGAN):
    """Complete GeoTrajGAN Model"""

    def __init__(
            self,
            gpu: int,
            name: Optional[str] = None,
            n_dim: int = 2,
            latent_dim_g: int = 256,
            latent_dim_d: int = 256,
            norm: NormOption = DEFAULT_NORM,
            activation: ActivationOption = DEFAULT_ACTIVATION,
            conv_only: bool = False,  # Replace all Linear Layers by Conv1D Layers with kernel size 1
            bi_lstm: bool = True,  # Use Bi-LSTM instead of LSTM (both models)
            bi_lstm_merge_mode: MergeMode = 'sum',  # Proved to be the best option in experiments
            sequential_mode: bool = True,  # Use sequential mode instead of point mode
            generator_lstm: bool = False,  # Use LSTM in Generator
            discriminator_lstm: bool = False,  # Use LSTM in Discriminator
            use_traj_discriminator: bool = False,  # Use Trajectory Discriminator in addition to Point Discriminator
            use_stn_in_point_dis: bool = True,  # Use STN in Point Discriminator
            use_stn_in_traj_dis: bool = False,  # Use STN in Trajectory Discriminator
            share_pointnet: bool = False,  # Share the weights of the PointNet between per-point and per-trajectory
            lstm_latent_dim: int = 64,  # Latent dimensionality of LSTMs, affects both models
    ):
        """
            Initialize the model with specified parameters.

            :param gpu: The GPU identifier to be used.
            :param name: Optional name for the model.
            :param n_dim: Dimensionality of the data.
            :param latent_dim_g: Latent dimension for the generator.
            :param latent_dim_d: Latent dimension for the discriminator.
            :param norm: Normalization method to use.
            :param activation: Activation function to use.
            :param conv_only: If True, replace all Linear Layers with Conv1D Layers.
            :param bi_lstm: If True, use Bi-LSTM instead of LSTM.
            :param bi_lstm_merge_mode: Merge mode for Bi-LSTM ('sum', 'concat', etc.).
            :param sequential_mode: If True, use sequential mode instead of point mode.
            :param generator_lstm: If True, use LSTM in the generator.
            :param discriminator_lstm: If True, use LSTM in the discriminator.
            :param use_traj_discriminator: If True, use a Trajectory Discriminator.
            :param use_stn_in_point_dis: If True, use STN in Point Discriminator.
            :param use_stn_in_traj_dis: If True, use STN in Trajectory Discriminator.
            :param share_pointnet: If True, share PointNet weights between per-point and per-trajectory.
            :param lstm_latent_dim: Latent dimensionality for LSTM layers.

            :returns: None
        """
        super().__init__(name=name)

        self.dp = False  # TODO
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu > -1 else "cpu")

        # Store parameters
        self.n_dim = n_dim
        self.latent_dim_g = latent_dim_g
        self.latent_dim_d = latent_dim_d
        self.norm = norm
        self.activation = activation
        self.conv_only = conv_only
        self.bi_lstm = bi_lstm
        self.bi_lstm_merge_mode = bi_lstm_merge_mode
        self.sequential_mode = sequential_mode
        self.generator_lstm = generator_lstm
        self.discriminator_lstm = discriminator_lstm
        self.use_traj_discriminator = use_traj_discriminator
        self.use_stn_in_point_dis = use_stn_in_point_dis
        self.use_stn_in_traj_dis = use_stn_in_traj_dis
        self.share_pointnet = share_pointnet
        self.lstm_latent_dim = lstm_latent_dim


        self.gen = TrajNet_Generator(
            code_nfts=latent_dim_g,
            n_dim=n_dim,
            stn=True,  # We did not observe any good performance without STN in Generator
            norm=norm,
            activation=activation,
            conv_only=conv_only,
            sequential=sequential_mode,
            use_lstm=generator_lstm,
            bi_lstm=bi_lstm,
            merge_mode=bi_lstm_merge_mode,
            lstm_latent_dim=lstm_latent_dim,
        ).to(device=device)
        self.dis = TrajNet_Discriminator(
            code_nfts=latent_dim_d,
            n_dim=n_dim,
            stn=use_stn_in_point_dis,
            traj_stn=use_stn_in_traj_dis,
            norm=norm,
            activation=activation,
            conv_only=conv_only,
            sequential=sequential_mode,
            use_lstm=discriminator_lstm,
            bi_lstm=bi_lstm,
            merge_mode=bi_lstm_merge_mode,
            per_traj_feedback=use_traj_discriminator,
            share_pointnet=share_pointnet,
            lstm_latent_dim=lstm_latent_dim,
        ).to(device=device)

        self.noise_dim = self.gen.noise_dim

        # Print Parameter counts for both models to compare strengths
        n_dis = count_parameters_torch(self.dis, False)
        n_gen = count_parameters_torch(self.gen, False)
        log.info(f"Discriminator Parameters:\t{n_dis:,}")
        log.info(f"Generator Parameters:\t{n_gen:,}")
        log.info(f"Proportion:\t\t\t{n_dis / n_gen:.2f}")

    def forward(self, x, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        return self.gen(x, lengths=lengths)

    def training_loop(
            self,
            dataloader: DataLoader,
            epochs: int,
            dataset_name: Datasets,
            name: Optional[str] = None,
            suffix: str = '',
            # Output and Tracking
            plot_freq: int = 200,
            save_freq: int = 100,
            param_path: Optional[str] = None,
            notebook: bool = False,
            tensorboard: bool = True,
            # Optimizer Options
            opt_g: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            opt_d: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            lr_d: float = 0.001,
            lr_g: float = None,
            beta1: float = 0.5,
            beta2: float = 0.999,
            lr_milestones: Optional[List[int]] = None,
            # Loss function options
            wgan: bool = True,
            gradient_penalty: bool = True,
            lp: bool = False,
            lambda_gp: int = 10,
            clip_value: float = 0.01,
            n_critic: int = 1,
            padded_generator: bool = False,
            traj_factor: float = 1.0,
            loss_combination: LossCombination = LossCombination.SUM,
    ):

        """
        Train a GAN.

        :param dataloader: DataLoader
        :param epochs: Number of epochs
        :param lr_g: Learning Rate of Generator
        :param lr_d: Learning Rate of Discriminator (default: None -> same as Generator LR)
        :param beta1: Only used for Adam (default: 0.5)
        :param beta2: Only used for Adam (default: 0.999)
        :param clip_value: Clipping value for Discriminator, only used for WGAN without Gradient Penalty
        :param n_critic: How many times to train the discriminator per generator run
        :param plot_freq: After how many BATCHES should the model be evaluated? (0 for no plotting)
        :param save_freq: Number of EPOCHS to save model parameters (default: 100). None for no storage.
        :param param_path: Directory to contain model parameters. Files will be param_path/EPOCH_{GEN|DIS}.pth.
        :param notebook: Use the notebook version of TQDM?
        :param wgan: Use WGAN loss function?
        :param gradient_penalty: Improved WGAN Loss function? (better performance, but more expensive)
        :param lp: Use Lipschitz penalty instead of gradient penalty. See: 10.48550/arXiv.1709.08894
        :param lambda_gp: Multiplier for Gradient Penalty (default: 10)
        :param opt_g: Optimizer to use for Generator. Default 'auto' uses recommended optimizer for loss function.
        :param opt_d: Optimizer to use for Discriminator. Default 'auto' uses recommended optimizer for loss function.
        :param name: Name of the model (for TensorBoard)
        :param dataset_name: Dataset name, important for visualization
        :param suffix: Suffix for Model Name in Tensorboard
        :param padded_generator: Let the generator generate sequences of same length as real input
        :param tensorboard: Use tensorboard for tracking of the training process?
        :param lr_milestones: List of steps after which the LR is decreased by factor 0.1.
        :param traj_factor: Factor for per-trajectory judgement.
        :param loss_combination: Method to combine losses. See: combine_losses()
        :return:
        """
        # Validate inputs
        validate_loss_options(wgan=wgan, gradient_penalty=gradient_penalty, lp=lp, wdiv=False)

        # Move to device
        device = self.device

        # Determine Optimizer to use
        # Provided optimizers take precedence
        lr_g = lr_d if lr_g is None else lr_g
        if not isinstance(opt_g, torch.optim.Optimizer):
            opt_g = get_optimizer(parameters=self.gen.parameters(), choice=opt_g, lr=lr_g, beta_1=beta1, beta_2=beta2,
                                  wgan=wgan,
                                  gradient_penalty=gradient_penalty)
        if not isinstance(opt_d, torch.optim.Optimizer):
            opt_d = get_optimizer(parameters=self.dis.parameters(), choice=opt_d, lr=lr_d, beta_1=beta1, beta_2=beta2,
                                  wgan=wgan,
                                  gradient_penalty=gradient_penalty)

        # Initialize DP-SGD: TODO
        # if dp:
        #     dp_model = DPModel()
        #     dis, opt_d, dataloader = dp_model.init_dp(
        #         model=dis, optimizer=opt_d, dataloader=dataloader, epochs=epochs,
        #         max_grad_norm=max_grad_norm, target_epsilon=epsilon, delta=delta)
        #     hyperparameters['target_epsilon'] = dp_model.target_epsilon
        #     hyperparameters['delta'] = dp_model.get_delta()
        #     hyperparameters['max_grad_norm'] = dp_model.max_grad_norm
        #     hyperparameters['accountant'] = dp_model.accountant
        # else:
        #     dp_model = None

        # Use MultiStepLR:
        scheduler_D, scheduler_G = None, None
        if lr_milestones is not None:
            log.warning(f"Using Multistep LR at {lr_milestones}.")
            scheduler_D = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=lr_milestones, gamma=0.1)
            scheduler_G = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=lr_milestones, gamma=0.1)

        # Determine Name for GAN based on parameters
        # Requires Optimizers to be defined
        if name is None:
            assert type(opt_g) is not str and \
                   type(opt_d) is not str, "Optimizers need to be defined before name can be determined."
            dataset_str = dataset_name.value if isinstance(dataset_name, Datasets) else dataset_name
            name = (f"GTG:{'i' if gradient_penalty else ''}{'W' if wgan else ''}GAN"
                    f"{'-LP' if lp else ''}_G-{type(opt_g).__name__}:{lr_g}_{n_critic}"
                    f"x_D-{type(opt_d).__name__}:{lr_d}_{dataset_str.upper()}")
        log.info(f"Model Name:\t\t{name}")

        # Determine Parameter Path and create directory
        self.param_path = prepare_param_path(name=name, param_path=param_path)

        # Create Tensorboard connection
        writer = SummaryWriter('runs/' + datetime.now().strftime('%b-%d_%X_') + name) if tensorboard else NullWriter()

        # Determine progressbar function
        tqdm_func = tqdm_ntbk if notebook else tqdm

        noise_dim = self.gen.noise_dim
        # Initialize loss
        bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        batches_done, gp = 0, -1
        # Create progressbar for epochs
        pbar_epochs = tqdm_func(range(1, epochs + 1), desc="Epochs")
        for epoch in pbar_epochs:
            with BatchMemoryManager(data_loader=dataloader, max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                                    optimizer=opt_d) if self.dp else nullcontext(dataloader) as dataloader:
                # Create progressbar for batches
                pbar_batches = tqdm_func(enumerate(dataloader), leave=False, total=len(dataloader), desc="Batches")
                for i, data in pbar_batches:

                    # Unpack minibatch
                    samples, lengths, labels = split_data(data)
                    # If we deploy fake padding in the generator, we use the same lengths as of the real data
                    gen_lengths = lengths if padded_generator else None

                    if len(samples) == 0:
                        # This can occur if Poisson Sampling (DP-SGD) is combined with a small batch size
                        continue

                    # Configure input
                    real_data = samples.to(device=device, dtype=torch.float32)

                    # Reshape input (For GeoPointGAN)
                    batch_size = real_data.size(0)
                    sequence_len = real_data.size(1)
                    total_points = batch_size * sequence_len if real_data.dim() > 2 else batch_size

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    opt_d.zero_grad()
                    self.dis.train()
                    self.gen.train()

                    # Sample noise
                    z = torch.randn(size=(batch_size, sequence_len, noise_dim), device=self.device)

                    # Generate a batch of images
                    fake_data = self.gen(z, lengths=gen_lengths)

                    # Adversarial loss

                    # 1. Compute Maks for the loss
                    real_mask, fake_mask = 1., 1.
                    if lengths is not None:
                        real_mask = compute_mask_from_lengths(real_data, lengths).view(-1, 1)
                    if padded_generator:
                        compute_mask_from_lengths(fake_data, gen_lengths).view(-1, 1)

                    # 2. Compute actual loss

                    if self.use_traj_discriminator:
                        point_dis_real, traj_dis_real = self.dis(real_data, lengths=lengths)
                        point_dis_fake, traj_dis_fake = self.dis(fake_data, lengths=gen_lengths)
                    else:
                        point_dis_real = self.dis(real_data, lengths=lengths)
                        point_dis_fake = self.dis(fake_data, lengths=gen_lengths)
                        traj_dis_real, traj_dis_fake = 0, 0

                    loss_weights = [1.0, traj_factor]

                    if wgan:
                        # Wasserstein GAN
                        loss_point_D = -torch.mean(point_dis_real) + torch.mean(point_dis_fake)
                        if self.use_traj_discriminator:
                            loss_traj = -torch.mean(traj_dis_real) + torch.mean(traj_dis_fake)
                            # Combine losses
                            loss_D = combine_losses([loss_point_D, loss_traj], method=loss_combination,
                                                    weights=loss_weights,
                                                    step=batches_done)
                        else:
                            loss_D = loss_point_D
                        if gradient_penalty:
                            # With Gradient or Lipschitz Penalty --> improved Wasserstein GAN (iWGAN) or WGAN_LP
                            gp = compute_gradient_penalty(discriminator=self.dis, real=real_data, synthetic=fake_data,
                                                          lengths=lengths, lp=lp)
                            loss_D += lambda_gp * gp
                    else:
                        # Standard GAN

                        # Ground Truth
                        valid = torch.full(size=(total_points, 1), fill_value=0.9, device=device)
                        fake = torch.zeros(size=(total_points, 1), device=device)
                        # Output for padding points is always 0. Therefore, we set the corresponding target to 0.
                        masked_valid = valid * real_mask
                        masked_fake = fake * fake_mask

                        # Per Point Loss
                        loss_point_D = (bce(point_dis_real, masked_valid) +
                                        bce(point_dis_fake, masked_fake)) / 2

                        # Per Trajectory Loss
                        if self.use_traj_discriminator:
                            # Concatenate the judgments for each trajectory
                            traj_valid = torch.ones((batch_size, 1), device=device)
                            traj_fake = torch.zeros((batch_size, 1), device=device)

                            loss_traj_D = (bce(traj_dis_real, traj_valid) +
                                           bce(traj_dis_fake, traj_fake)) / 2
                            loss_D = combine_losses([loss_point_D, loss_traj_D], method=loss_combination,
                                                    weights=loss_weights, step=batches_done)
                        else:
                            loss_D = loss_point_D

                    # Test run
                    if not tensorboard:
                        exit()

                    loss_D.backward()

                    opt_d.step()
                    if lr_milestones is not None:
                        scheduler_D.step()

                    # Clip weights of discriminator
                    if wgan and not gradient_penalty:
                        clip_discriminator_weights(dis=self.dis, clip_value=clip_value)

                    # Train the generator every n_critic iterations
                    if i % n_critic == 0:
                        # -----------------
                        #  Train Generator
                        # -----------------

                        opt_g.zero_grad()

                        # Generate data
                        z = torch.randn(size=(batch_size, sequence_len, noise_dim), device=self.device)
                        fake_data = self.gen(z)

                        # Generator loss
                        if self.use_traj_discriminator:
                            point_dis, traj_dis = self.dis(fake_data, lengths=gen_lengths)
                        else:
                            point_dis = self.dis(fake_data, lengths=gen_lengths)
                            traj_dis = 0

                        if wgan:
                            # WGAN
                            loss_G = -torch.mean(point_dis)
                            if self.use_traj_discriminator:
                                loss_traj = -torch.mean(traj_dis)
                                # Combine losses
                                loss_G = combine_losses([loss_G, loss_traj], method=loss_combination,
                                                        weights=loss_weights, step=batches_done)
                        else:
                            # Standard GAN
                            masked_valid = torch.full(size=(total_points, 1), fill_value=0.9, device=device) * fake_mask
                            loss_G = bce(point_dis, masked_valid)
                            if self.use_traj_discriminator:
                                traj_valid = torch.ones((batch_size, 1), device=device)
                                loss_traj = bce(traj_dis, traj_valid)
                                loss_G = combine_losses([loss_G, loss_traj], method=loss_combination,
                                                        weights=loss_weights, step=batches_done)

                        loss_G.backward()

                        opt_g.step()
                        if lr_milestones is not None:
                            scheduler_G.step()

                        if tensorboard:
                            # Plot loss
                            writer.add_scalar("Loss/Gen", loss_G.item(), global_step=batches_done)
                            # For WGAN, one has to plot -D_loss:
                            # https://github.com/martinarjovsky/WassersteinGAN/issues/ 9#issuecomment-280989632
                            # Moreover, we might want to remove the gradient penalty to get a clearer image
                            loss_D = loss_D.item()
                            if wgan:
                                if gradient_penalty:
                                    # Remove gradient penalty and plot separately
                                    loss_D -= lambda_gp * gp
                                    writer.add_scalar("Loss/GP", gp, global_step=batches_done)
                                loss_D = -loss_D
                            writer.add_scalar("Loss/Dis", loss_D, global_step=batches_done)
                            if self.use_traj_discriminator:
                                # Print the point and trajectory losses separately
                                writer.add_scalar("Loss/Dis_Point", loss_point_D.item(), global_step=batches_done)
                                writer.add_scalar("Loss/Dis_Traj", loss_traj_D.item(), global_step=batches_done)
                            # Plot gradients
                            for name, weight in self.gen.plotable_gradients.items():
                                writer.add_scalar("Grad/Gen/" + name, weight.grad.norm(), global_step=batches_done)
                            else:
                                for name, weight in self.dis.plotable_gradients.items():
                                    writer.add_scalar("Grad/Dis/" + name,
                                                      weight.grad.norm() if weight.grad is not None else 0.0,
                                                      global_step=batches_done)

                    if plot_freq > 0 and batches_done % plot_freq == 0:
                        if not self.sequential_mode:
                            # Reshape to (N, L, F)
                            fake_data = fake_data.view(-1, sequence_len, self.n_dim)
                        if notebook:
                            # Clear output and display plot
                            display.clear_output(wait=True)
                            # Display progressbars again
                            display.display(pbar_epochs.container)
                            display.display(pbar_batches.container)
                        if 'mnist' in dataset_name:
                            visualize_mnist_sequential(
                                gen_samples=fake_data,
                                batches_done=batches_done,
                                notebook=notebook,
                                tensorboard=tensorboard,
                                writer=writer,
                            )
                        else:
                            # Trajectory Dataset
                            visualize_trajectory_samples(
                                gen_samples=fake_data,
                                real_samples=real_data,
                                real_lengths=lengths,
                                gen_lengths=gen_lengths,
                                epoch=epoch,
                                batch_i=batches_done,
                                notebook=notebook,
                                tensorboard=tensorboard,
                                writer=writer,
                            )

                    batches_done += 1

            if (save_freq > 0 and epoch % save_freq == 0) or epoch == epochs:  # Always save after last epoch
                self.save_parameters(epoch=epoch)
