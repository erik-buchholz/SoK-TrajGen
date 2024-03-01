#!/usr/bin/env python3
"""PyTorch Utilities for Machine Learning"""
import contextlib
import logging
import os

from enum import Enum
from typing import List, Optional

import numpy as np
import torch.nn
from IPython import display
from matplotlib import pyplot as plt
from torch import autograd, nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from stg import config
from stg.utils import visualise

log = logging.getLogger()


class RnnType(str, Enum):
    RNN = 'rnn'
    LSTM = 'lstm'
    GRU = 'gru'


RNN_TYPES = {
    RnnType.RNN: torch.nn.RNN,
    RnnType.LSTM: torch.nn.LSTM,
    RnnType.GRU: torch.nn.GRU
}

RNN_CELL_TYPES = {
    RnnType.RNN: torch.nn.RNNCell,
    RnnType.LSTM: torch.nn.LSTMCell,
    RnnType.GRU: torch.nn.GRUCell
}


class Optimizer(str, Enum):
    AUTO = 'auto'
    ADAM = 'adam'
    RMSPROP = 'rmsprop'
    ADAMW = 'adamw'
    SGD = 'sgd'


OPTIMIZERS = {
    Optimizer.AUTO: None,
    Optimizer.ADAM: torch.optim.Adam,
    Optimizer.RMSPROP: torch.optim.RMSprop,
    Optimizer.ADAMW: torch.optim.AdamW,
    Optimizer.SGD: torch.optim.SGD
}

MOMENTUM = {
    Optimizer.ADAM: True,
    Optimizer.RMSPROP: False,
    Optimizer.ADAMW: True,
    Optimizer.SGD: False
}


def get_rnn(rnn_type: RnnType, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0,
            bidirectional: bool = False):
    """
    Interface to get RNN layer for quick exchange in models
    :param rnn_type: type € [rnn, lstm, gru]
    :param input_size:
    :param hidden_size:
    :param num_layers: [Default: 1]
    :param dropout: probability € [0,1] [Default: 0]
    :param bidirectional: Use bidirectional RNN [Default: False]
    :return:
    """

    rnn_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout,
        bidirectional=bidirectional
    )

    return RNN_TYPES[rnn_type.lower()](**rnn_kwargs)


def get_rnn_cell(rnn_type: RnnType, input_size: int, hidden_size: int):
    """
    Interface to get RNN cell for quick exchange in models
    :param rnn_type: type € [rnn, lstm, gru]
    :param input_size:
    :param hidden_size:
    :return:
    """

    rnn_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size
    )

    return RNN_CELL_TYPES[rnn_type.lower()](**rnn_kwargs)


def get_optimizer(
        parameters,
        choice: Optimizer = Optimizer.AUTO,
        lr: float = 1e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        wgan: bool = True,
        gradient_penalty: bool = True
):
    choice = choice.lower()
    if choice != Optimizer.AUTO and choice not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer type: {str(choice)}.")

    # Resolve automatic mode
    if choice == Optimizer.AUTO:
        if wgan and not gradient_penalty:
            choice = Optimizer.RMSPROP
        else:
            choice = Optimizer.ADAMW

    if MOMENTUM[choice]:
        opt = OPTIMIZERS[choice](parameters, lr=lr, betas=(beta_1, beta_2))
    else:
        opt = OPTIMIZERS[choice](parameters, lr=lr)

    return opt


def l1_regularizer(weights, l1: float = 0.01):
    """PyTorch replacement for keras.regularizers.l1

    :return: L1 penalty term
    """
    return l1 * torch.norm(weights, p=1)


def compute_gradient_penalty(discriminator: nn.Module, real: Tensor, synthetic: Tensor,
                             lengths: list or None = None, lp: bool = False) -> Tensor:
    """
    Implementation of gradient penalty is based on [2]. Lipschitz penalty added by me based on [1].

    Usually, real and synthetic samples are provided as (batch_size, sequence_length, feature_dim) tensors.
    However, some models, particularly the LSTM-TrajGAN-based models, provide a list of tensors with
    (batch_size, sequence_length, feature_dim) tensors for each feature.

    [1] H. Petzka, A. Fischer, and D. Lukovnicov, “On the regularization of Wasserstein GANs.” arXiv, Mar. 05, 2018.
    doi: 10.48550/arXiv.1709.08894.
    [2] A. Sankar, “Demystified: Wasserstein GAN with Gradient Penalty,” Medium. Accessed: May 02, 2023. [Online].
    Available: https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead

    :param discriminator: torch.nn.Module
    :param real: (batch_size, sequence_length, feature_dim) or (num_features, batch_size, sequence_length, feature_dim)
    :param synthetic: (batch_size, sequence_length, feature_dim) or (num_features, batch_size, sequence_length, feature_dim)
    :param lengths: List with length of the samples before padding
    :param lp: Use Lipschitz penalty [1] instead of gradient penalty.
    :return: Gradient/Lipschitz Penalty
    """
    device = real.device if isinstance(real, Tensor) else real[0].device

    # Random weight term for interpolation between real and fake samples

    # Differentiate between single and multi-feature input
    if isinstance(real, Tensor) and isinstance(synthetic, Tensor):
        # Trajectories provided as (batch_size, sequence_length, feature_dim) tensors
        batch_size = len(real)
        alpha = torch.rand(batch_size, 1, 1).to(device=device)
        alpha = alpha.expand_as(real)
        # Get random interpolation between real and fake samples
        interpolation = alpha * real + (1 - alpha) * synthetic
        interpolates = autograd.Variable(interpolation, requires_grad=True).to(device=device)
    elif isinstance(real, list) and isinstance(synthetic, list):
        batch_size = len(real[0])
        alpha = torch.rand(batch_size, 1, 1).to(device)
        interpolates = []
        for i, feature in enumerate(real):
            alpha_expanded = alpha.expand_as(feature)
            interpolation = alpha_expanded * feature + (1 - alpha_expanded) * synthetic[i]
            interpolates.append(autograd.Variable(interpolation, requires_grad=True).to(device))
    else:
        raise ValueError("Real and synthetic samples must be either both tensors or both lists.")

    # Check if the RNN uses cuDNN backend and if so, disable cuDNN for this part of the code
    # Necessary b/c of `NotImplementedError: the derivative for '_cudnn_rnn_backward' is not implemented.`
    if contains_rnn(discriminator):
        if device.type == 'cpu':
            torch._C._set_mkldnn_enabled(False)
            context = contextlib.nullcontext()
        else:
            context = torch.backends.cudnn.flags(enabled=False)
    else:
        context = contextlib.nullcontext()
    with context:
        # Get logits for interpolated samples
        d_interpolates = discriminator(interpolates) if lengths is None else discriminator(interpolates,
                                                                                           lengths=lengths)
    # If discriminator returns tuple, concatenate
    if isinstance(d_interpolates, tuple):
        d_interpolates = torch.cat(d_interpolates, dim=0)
    grad_outputs = torch.ones_like(d_interpolates).to(device=device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)

    if lp:
        # Lipschitz Penalty
        gradient_penalty = (torch.clamp(gradients.norm(2, dim=1) - 1., min=0, max=None) ** 2).mean()
    else:
        # Gradient Penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    log.debug("Unweighted Gradient Penalty: ", gradient_penalty.item())
    return gradient_penalty


def compute_wdiv_penalty(real: Tensor, fake: Tensor, real_validity: Tensor, fake_validity: Tensor, p=6, k=2):
    DEVICE = real.device
    real_grad_outputs = torch.ones_like(real_validity, dtype=torch.float32, requires_grad=False, device=DEVICE)
    fake_grad_outputs = torch.ones_like(fake_validity, dtype=torch.float32, requires_grad=False, device=DEVICE)

    real_gradient = torch.autograd.grad(
        outputs=real_validity,
        inputs=real,
        grad_outputs=real_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    fake_gradient = torch.autograd.grad(
        outputs=fake_validity,
        inputs=fake,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    real_gradient_norm = real_gradient.reshape(real_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_gradient_norm = fake_gradient.reshape(fake_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)

    div_gp = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
    return div_gp


def save_models(gen: nn.Module, dis: nn.Module or None, param_path: str, epoch: int) -> None:
    if '{EPOCH}' not in param_path or '{MODEL}' not in param_path or param_path[-4:] != '.pth':
        raise ValueError(
            "Parameter Path has to end in '.pth' and contain {MODEL} and {EPOCH}. Provided: " +
            str(param_path)
        )
    torch.save(gen.state_dict(), param_path.format(EPOCH=epoch, MODEL='GEN'))
    if dis is not None:
        torch.save(dis.state_dict(), param_path.format(EPOCH=epoch, MODEL='DIS'))
    log.info(f"Saved models to '{param_path}'.")


def save_model(model: nn.Module, param_path: str, epoch: int, model_str: str = None, verbose: bool = True) -> None:
    if '{EPOCH}' not in param_path or param_path[-4:] != '.pth':
        raise ValueError(
            "Parameter Path has to end in '.pth' and contain {EPOCH}. Provided: " +
            str(param_path)
        )
    if model_str is None and '{MODEL}' in param_path:
        raise ValueError(
            "Parameter Path must not contain {MODEL} if no model_str is provided. Provided: " +
            str(param_path)
        )
    elif model_str is not None and '{MODEL}' not in param_path:
        raise ValueError(
            "Parameter Path must contain {MODEL} if a model_str is provided. Provided: " +
            str(param_path)
        )
    if model_str is None:
        torch.save(model.state_dict(), param_path.format(EPOCH=epoch))
    else:
        torch.save(model.state_dict(), param_path.format(EPOCH=epoch, MODEL=model_str))
    if verbose:
        log.info(f"Saved model to '{param_path}'.")


def load_models(gen: nn.Module, dis: nn.Module, param_path: str, epoch: int, device: str = None) -> None:
    """
    Load model parameters from file.
    :param gen: Generator/Model 1
    :param dis: Discriminator. If None, only first model is loaded.
    :param param_path: Path to parameter file. Must contain {EPOCH} and {MODEL} placeholders.
    :param epoch: Epoch to load.
    :param device: Device to load the model to. If None, model is loaded to CPU.
    :return:
    """
    if '{EPOCH}' not in param_path or '{MODEL}' not in param_path or param_path[-4:] != '.pth':
        raise ValueError(
            "Parameter Path has to end in '.pth' and contain {MODEL} and {EPOCH}. Provided: " +
            str(param_path)
        )
    load_model(model=gen, param_path=param_path, epoch=epoch, model_str='GEN', device=device, verbose=True)
    if dis is not None:
        load_model(model=dis, param_path=param_path, epoch=epoch, model_str='DIS', device=device, verbose=True)


def load_model(model: nn.Module, param_path: str, epoch: int, device: str, model_str: Optional[str] = None,
               verbose: bool = True) -> None:
    """
    Load model parameters from file.
    :param model: Model to load parameters into.
    :param param_path: Path to parameter file. Must contain {EPOCH} placeholder, and {MODEL} if model_str is None.
    :param epoch: Epoch to load.
    :param model_str: String to replace {MODEL} placeholder with. If None, ignored.
    :param device: Device to load the model to. If None, model is loaded to CPU.
    :param verbose: Print log messages.
    :return:
    """
    if '{EPOCH}' not in param_path or param_path[-4:] != '.pth':
        raise ValueError(
            "Parameter Path has to end in '.pth' and contain {EPOCH}. Provided: " +
            str(param_path)
        )
    if model_str is None and '{MODEL}' in param_path:
        raise ValueError(
            "Parameter Path must not contain {MODEL} if no model_str is provided. Provided: " +
            str(param_path)
        )
    map_location = torch.device('cpu') if device == "cpu" else device
    if model_str is None:
        model.load_state_dict(torch.load(param_path.format(EPOCH=epoch), map_location=map_location))
    else:
        model.load_state_dict(torch.load(param_path.format(EPOCH=epoch, MODEL=model_str), map_location=map_location))
    if verbose:
        log.info(f"Loaded model from '{param_path.format(EPOCH=epoch, MODEL=model_str)}'.")


def clip_discriminator_weights(dis: nn.Module, clip_value: float) -> None:
    for p in dis.parameters():
        p.data.clamp_(-clip_value, clip_value)


def contains_rnn(model: nn.Module) -> bool:
    return any(isinstance(module, torch.nn.RNNBase) for module in model.modules())


import torch


def split_data(data: torch.Tensor or list or tuple) -> (torch.Tensor, List[int] or None, list or None):
    """
    Splits the input data into samples, lengths, and labels based on its structure.
    Assumes that labels are always present if lengths are present.

    :param data: The input data which might be a tensor, a list, or a tuple.

    :returns: A tuple containing samples, lengths, and labels in this order.
    """
    # Check if data is a tensor
    if torch.is_tensor(data):
        samples = data
        labels = None
        lengths = None
    else:
        # Unpack data based on the length of the list or tuple
        if len(data) == 3 and not (isinstance(data[1], torch.Tensor) and data[1].dim() == 3):
            # The part behind the and captures the scenario that with have a list of three features.
            samples, lengths, labels = data
        elif len(data) == 2:
            # Assumes that labels are always present if lengths are present
            samples, labels = data
            lengths = None
        else:
            # Assume data is just samples
            samples = data
            labels = None
            lengths = None

    return samples, lengths, labels


def validate_loss_options(wgan: bool, gradient_penalty: bool, lp: bool, wdiv: bool = False) -> None:
    """
    Validates the combination of loss options for consistency.

    :param wgan: Indicates if Wasserstein GAN is used.
    :param gradient_penalty: Indicates if Gradient Penalty is used.
    :param lp: Indicates if Lipschitz Penalty is used.
    :param wdiv: Indicates if Wasserstein Divergence is used.
    :return: None
    :raises ValueError: If an invalid combination of options is provided.
    """
    if wdiv and gradient_penalty:
        raise ValueError("Either Wasserstein Divergence OR Gradient Penalty is possible, not both.")
    if gradient_penalty and not wgan:
        raise ValueError("Gradient Penalty only works with Wasserstein Loss.")
    if wdiv and not wgan:
        raise ValueError("Wasserstein Divergence only works with Wasserstein Loss.")
    if lp and not gradient_penalty:
        raise ValueError("Lipschitz Penalty only works with Gradient Penalty.")


def prepare_param_path(name: str, param_path: str = None) -> str:
    """
    Prepares the parameter path for saving model parameters.

    :param param_path: If not none overrides the default parameter path.
    :param name: The name to be appended to the base path.
    :return: The prepared parameter path with placeholders for epoch and model.
    """
    # Set default path if not provided
    param_path = param_path or os.path.join(config.PARAM_PATH, name)

    # Ensure the path ends with a '/'
    param_path = os.path.join(param_path, '')

    # Create the directory if it doesn't exist
    os.makedirs(param_path, exist_ok=True)

    # Prepare the final path with placeholders for epoch and model
    param_path = os.path.join(param_path, "{EPOCH}_{MODEL}.pth")

    return param_path


def verify_non_linearity(wgan: bool, last_layer: torch.nn.Module) -> None:
    """
    Verifies that the correct non-linearity is used in the discriminator's last layer.

    :param wgan: Indicates if Wasserstein GAN is used.
    :param last_layer: The last layer of the discriminator model.
    :raises ValueError: If an incorrect non-linearity is used.
    """
    if not wgan:
        if not isinstance(last_layer, torch.nn.Sigmoid):
            raise ValueError("For standard GAN, last layer has to be sigmoid!")
    else:
        if isinstance(last_layer, torch.nn.Sigmoid):
            raise ValueError("For WGAN, no sigmoid is used as last layer!")


class NullWriter:
    """
    A class to replace a Tensorboard SummaryWriter without code modifications.
    """

    def __getattr__(self, _):
        # Return a lambda function that does nothing
        return lambda *args, **kwargs: None


def visualize_trajectory_samples(
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        epoch: int,
        batch_i: int,
        real_lengths: List[int] = None,
        gen_lengths: List[int] = None,
        notebook: bool = False,
        tensorboard: bool = False,
        writer: Optional[SummaryWriter] = None
) -> None:
    """
    Visualizes trajectory samples by plotting an example trajectory and a point cloud of generated and real samples.

    :param gen_samples: Tensor of generated samples.
    :param real_samples: Tensor of real samples.
    :param gen_lengths: List of lengths of the generated samples if padded.
    :param epoch: Current epoch number for titling the plots.
    :param batch_i: Current batch index for titling the plots.
    :param real_lengths: List of lengths of the real samples if padded. Defaults to None.
    :param notebook: Specifies whether the visualization is in a Jupyter notebook. Defaults to False.
    :param tensorboard: Specifies whether to log the plots to TensorBoard. Defaults to False.
    :param writer: TensorBoard writer object. Required if tensorboard is True.

    :returns: None
    """

    # Convert to numpy
    gen_samples = gen_samples.detach().cpu().numpy()
    if not isinstance(real_samples, list):
        real_samples = real_samples.detach().cpu().numpy()

    if real_lengths is not None:
        # Slice to real length
        real_samples = [r[:real_lengths[i]] for i, r in enumerate(real_samples)]

    if gen_lengths is not None:
        # Slice to real length
        gen_samples = [g[:gen_lengths[i]] for i, g in enumerate(gen_samples)]

    # Randomly select one real and one generated sample for example plot
    idx = torch.randint(low=0, high=len(gen_samples), size=(1,))
    gen_traj = gen_samples[idx]
    real_traj = real_samples[idx]

    # Plot
    example_fig, _ = visualise.plot_trajectories(
        [real_traj, gen_traj],
        ['Real', 'Fake'],
        bbox=(-1, 1, -1, 1),  # Points are normalized to [-1, 1]
        title=f"Example Trajectory: Epoch {epoch} - Batch {batch_i}"
    )

    # Plot pointcloud
    gen_points = np.concatenate(gen_samples, axis=0)
    real_points = np.concatenate(real_samples, axis=0)

    heatmap, _ = visualise.plot_pointclouds(
        [gen_points, real_points],
        title=f"Point Cloud: Epoch {epoch} - Batch {batch_i}",
        labels=['Fake', 'Real'],
    )

    if notebook:
        # Display both figures by replacing the old one
        display.display(example_fig)
        display.display(heatmap)
        # Close old figures
        plt.close('all')
    if tensorboard:
        if writer is None:
            raise ValueError("TensorBoard writer must be provided if tensorboard logging is enabled.")
        writer: SummaryWriter
        writer.add_figure('Example', example_fig, global_step=batch_i)
        writer.add_figure('Heatmap', heatmap, global_step=batch_i)


def visualize_mnist_sequential(
        gen_samples: torch.Tensor,
        batches_done: int,
        notebook: bool = False,
        tensorboard: bool = False,
        writer: Optional[SummaryWriter] = None,
) -> None:
    gen_samples = gen_samples.view(-1, 1, 28, 28)
    if notebook:
        # Make a grid of images and convert it to numpy for plotting
        grid = make_grid(gen_samples[:25], nrow=5, normalize=True).permute(1, 2, 0).detach().cpu().numpy()
        # Display the grid of images
        fig, ax = plt.subplots()
        ax.imshow(grid)
        ax.axis('off')

        # Show new plot
        display.display(plt.gcf())
        # Close old figure
        plt.close('all')
    if tensorboard:
        # Write to TensorBoard
        writer.add_images("Example", gen_samples.data[:25], batches_done)


def compute_mask(samples: torch.Tensor, padding_value: float = 0.0) -> torch.Tensor:
    """
    Compute a mask for the given samples.

    :param samples: Padded samples with shape (batch_size, time_steps, features).
    :param padding_value: Value used for padding. Defaults to 0.0.
    :returns: Mask with shape (batch_size, time_steps), where each entry is 1 if the corresponding sample is not padding and 0 if it is padding.
    """
    # Assuming padding is done in the time dimension, and the padding value is constant across all features
    mask = (samples != padding_value).any(dim=-1).float()
    return mask


def compute_len_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the length of each sample from the given mask.

    :param mask: Mask with shape (batch_size, time_steps), where each entry is 1 if the corresponding sample is not padding and 0 if it is padding.
    :returns: Length of each sample with shape (batch_size,).
    """
    # Sum over time dimension
    lengths = mask.sum(dim=-1)
    return lengths


def compute_mask_from_lengths(x: torch.Tensor, lengths: List[int] or torch.Tensor) -> torch.Tensor:
    """
    Compute a mask for a given tensor x and a list of lengths.
    :param x: Expected shape = (N, L, C), for N = batch size, L = sequence length, C = feature channels
    :param lengths: List of lengths for each sequence in x
    :return: Binary mask of shape (N, L) with 1s for valid entries and 0s for invalid entries
    """
    # Validate input lengths
    max_length = x.size(1)
    if len(lengths) != x.size(0):
        raise ValueError("The number of lengths must match the batch size of x.")
    if max(lengths) > max_length:
        raise ValueError("Lengths exceed the maximum sequence length of x.")

    # Generate a range tensor that represents sequence positions
    sequence_positions = torch.arange(max_length, device=x.device)

    # Expand the sequence_positions tensor to match the batch size
    expanded_positions = sequence_positions.expand(len(lengths), max_length)

    # Convert lengths to a tensor and reshape for broadcasting
    lengths_tensor = torch.tensor(lengths, device=x.device).unsqueeze(-1)

    # Generate the mask by comparing each position to the corresponding length
    mask = expanded_positions.lt(lengths_tensor)

    return mask


class MergeMode(str, Enum):
    SUM = 'sum'
    AVERAGE = 'average'
    CONCAT = 'concat'
    MUL = 'mul'


def merge_bilstm_output(x: torch.Tensor, mode: MergeMode) -> torch.Tensor:
    """
    Merge the output of a bidirectional LSTM
    :param x: Output of the LSTM
    :param mode: How to merge the output (sum, mean, concat)
    :return:
    """
    if mode == MergeMode.SUM:
        x = torch.sum(x.view(x.size(0), x.size(1), 2, -1), dim=2)
    elif mode == MergeMode.AVERAGE:
        x = torch.mean(x.view(x.size(0), x.size(1), 2, -1), dim=2)
    elif mode == MergeMode.CONCAT:
        pass  # Default
    elif mode == MergeMode.MUL:
        x = torch.prod(x.view(x.size(0), x.size(1), 2, -1), dim=2)
    else:
        raise ValueError(f"Unknown merge mode: {mode}. Valid modes are 'sum', 'average', 'concat', and 'mul'.")
    return x


class LossCombination(str, Enum):
    MEAN = 'mean'
    SUM = 'sum'
    MAX = 'max'
    MIN = 'min'
    PROD = 'prod'
    NONE = 'none'
    ALTERNATE = 'alternate'


def combine_losses(losses: List[torch.Tensor], method: LossCombination = LossCombination.MEAN,
                   weights: List[float] = None, step: int = None):
    """
    Combine losses into a single loss value.

    :param losses: List of losses to combine
    :param method: Method to combine losses.
        'mean': Mean of losses
        'sum': Sum of losses
        'max': Maximum of losses
        'min': Minimum of losses
        'prod': Product of losses
        'none': Return list of losses
        'alternate': Alternate between losses
    :param weights: Weights for each loss
    :param step: Global step (Batch number)
    :return: Combined loss
    """
    if weights is None:
        weights = [1.0] * len(losses)
    assert len(losses) == len(weights), "Number of losses and weights must match."
    if method == 'mean':
        return sum([w * l for w, l in zip(weights, losses)]) / sum(weights)
    elif method == 'sum':
        return sum([w * l for w, l in zip(weights, losses)])
    elif method == 'max':
        return max(losses)
    elif method == 'min':
        return min(losses)
    elif method == 'prod':
        return torch.prod(torch.stack(losses))
    elif method == 'none':
        return losses
    elif method == 'alternate':
        if step is None:
            raise ValueError("Step must be provided for alternate loss combination.")
        return losses[step % len(losses)]
    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")
