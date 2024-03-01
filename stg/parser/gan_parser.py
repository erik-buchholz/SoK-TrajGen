#!/usr/bin/env python3
"""General GAN parser - Parent for specific GAN parsers."""
import argparse
import logging

from stg.datasets import DATASET_CLASSES
from stg.models import RNNModels
from stg.models.layers import NormOption, ActivationOption
from stg.models.utils import Optimizer, MergeMode


def clean_args(args: dict) -> dict:
    """Remove admin values from args dict."""
    admin_keys = ['config', 'save_config', 'summary']
    for key in admin_keys:
        if key in args:
            args.pop(key)
    return args


def get_base_parser(dp: bool = False, gan: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    general_group = parser.add_argument_group("General")
    general_group.add_argument("-g", "--gpu", type=int, required=True, help='GPU to use. -1 for CPU.')
    general_group.add_argument("-n", "--name", type=str, help="Model Name (determines parameter path)")

    # Training
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("-e", "--epochs", type=int, help="Number of epochs of training.")
    training_group.add_argument("-b", "--batch_size", type=int, help="Size of the batches.")
    training_group.add_argument("--save_freq", type=int, default=50,
                                help="Save Parameters after X EPOCHS. [-1 to deactivate]")

    # Optimizer
    adam_group = parser.add_argument_group("Adam Optimizer")
    adam_group.add_argument("--beta1", type=float, help="Optimizer beta_1.")
    adam_group.add_argument("--beta2", type=float, help="Optimizer beta_2.")

    # Logging and output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("-v", "--verbose", action='store_const', const=logging.DEBUG,
                              default=logging.INFO, dest='logging_lvl', help="Activate debug logging.")
    output_group.add_argument("-p", "--plot_freq", type=int, default=100,
                              help="Plot after X BATCHES. [-1 to deactivate]")
    output_group.add_argument("--tb", "--tensorboard", dest='tensorboard', action="store_false",
                              help="DEACTIVATE Tensorboard logging. [Default: ACTIVATED]")

    # Dataset
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("-d", "--dataset", choices=DATASET_CLASSES, help="Dataset to evaluate on.",
                               type=str.lower, required=True)
    dataset_group.add_argument("--shuffle", action="store_false",
                               help="DEACTIVATE shuffling data before each epoch. [Default: ACTIVATED]")

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("-c", "--config", type=str,
                              help="Load configuration from file in ./config/{CONFIG}.json. Overwrites other values!")
    config_group.add_argument("--save_config", action="store_true",
                              help="Write current setting to config file.")

    # Differential Privacy
    if dp:
        dp_group = parser.add_argument_group("Differential Private SGD (DP-SGD)")
        add_dp_arguments(dp_group)

    # Add GAN specific arguments
    if gan:
        parser = add_gan_arguments(parser)

    return parser


def add_dp_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dp", action='store_true', help="Use Differential Private-SGD for training.")
    parser.add_argument("--epsilon", type=float, help="Epsilon for DP.")
    parser.add_argument("--delta", type=float, help="Delta for DP.")
    return parser


def add_gan_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Optimizer Group
    opt_group = parser.add_argument_group("Optimizer")
    opt_group.add_argument("--opt_d", choices=[o.value for o in Optimizer], default=Optimizer.AUTO,
                           help="Discriminator Optimizer.")
    opt_group.add_argument("--opt_g", choices=[o.value for o in Optimizer], default=Optimizer.AUTO,
                           help="Generator Optimizer.")
    opt_group.add_argument("--lr_d", type=float, help="Learning rate (Discriminator).")
    opt_group.add_argument("--lr_g", type=float, help="Learning rate (Generator).")
    opt_group.add_argument("--n_critic", type=int, help="Discriminator runs per Generator run.", default=1)

    # (improved) Wasserstein GAN
    wgan_group = parser.add_argument_group("(improved) Wasserstein GAN")
    wgan_group.add_argument("--wgan", action='store_true',
                            help="Use Wasserstein Loss.")
    wgan_group.add_argument("--gp", action='store_true', dest="gradient_penalty",
                            help="Use Wasserstein Loss with Gradient Penalty.")
    wgan_group.add_argument("--lp", action='store_true',
                            help='Use Lipschitz penalty instead of gradient penalty.')
    wgan_group.add_argument("--clip_value", type=float,
                            help="WGAN clipping value for discriminator (if no GP used).")
    wgan_group.add_argument("--lambda", dest="lambda_gp", type=float, default=10,
                            help="Weight factor for gradient/lipschitz penalty.")

    return parser


def get_geotrajgan_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()

    defaults = {
        # General
        "epochs": 1000,
        "batch_size": 256,

        # Output
        "plot_freq": 100,
        "save_freq": 100,

        # Optimizer
        "opt_g": "auto",
        "lr_g": 1e-5,
        "opt_d": "auto",
        "lr_d": 1e-5,
        "beta1": 0.5,
        "beta2": 0.999,

        # Loss
        "n_critic": 1,
        "wgan": False,
        "gradient_penalty": False,
        "lp": False,
        "clip_value": 0.01,
        "lambda_gp": 10,

        # Architecture
        "sequential_mode": True,
        "n_dim": 2,
        "latent_dim_g": 256,
        "latent_dim_d": 256,
        "norm": "layer",
        "activation": "relu",
        "conv_only": False,
        "bi_lstm": True,
        "bi_lstm_merge_mode": "sum",
        "generator_lstm": True,
        "lstm_latent_dim": 64,
        "discriminator_lstm": False,
        "use_traj_discriminator": True,
        "use_stn_in_point_dis": False,
        "use_stn_in_traj_dis": False,
        "share_pointnet": False,


    }

    parser.set_defaults(**defaults)

    # ### GeoTrajGAN Architecture-specific Options ###
    geotrajgan_group = parser.add_argument_group("GeoTrajGAN Architecture")
    # geotrajgan_group.add_argument("--n_dim", type=int, help="Dimensionality of the data.")
    # n_dim defined via dataset
    geotrajgan_group.add_argument("--latent_dim_g", type=int, help="Latent dimension for the generator.")
    geotrajgan_group.add_argument("--latent_dim_d", type=int,
                                  help="Latent dimension for the discriminator.")
    geotrajgan_group.add_argument("--norm", choices=[n.value for n in NormOption],
                                  help="Normalization method to use.")
    geotrajgan_group.add_argument("--activation", choices=[a.value for a in ActivationOption],
                                  help="Activation function to use.")
    geotrajgan_group.add_argument("--conv_only", action='store_true',
                                  help="Replace all Linear Layers by Conv1D Layers with kernel size 1.")
    geotrajgan_group.add_argument("--bi_lstm", action='store_true',
                                  help="Use Bi-LSTM instead of LSTM (both models).")
    geotrajgan_group.add_argument("--bi_lstm_merge_mode", choices=[m.value for m in MergeMode],
                                  help="Merge mode for Bi-LSTM.")
    geotrajgan_group.add_argument("--sequential_mode", action='store_false',
                                  help="Use sequential mode instead of point mode.")
    geotrajgan_group.add_argument("--generator_lstm", action='store_true',
                                  help="Use LSTM in Generator.")
    geotrajgan_group.add_argument("--discriminator_lstm", action='store_true',
                                  help="Use LSTM in Discriminator.")
    geotrajgan_group.add_argument("--use_traj_discriminator", action='store_true',
                                  help="Use Trajectory Discriminator in addition to Point Discriminator.")
    geotrajgan_group.add_argument("--use_stn_in_point_dis", action='store_true', help="Use STN in Point Discriminator.")
    geotrajgan_group.add_argument("--use_stn_in_traj_dis", action='store_true',
                                  help="Use STN in Trajectory Discriminator.")
    geotrajgan_group.add_argument("--share_pointnet", action='store_true',
                                  help="Share the weights of the PointNet between per-point and per-trajectory.")
    geotrajgan_group.add_argument("--lstm_latent_dim", type=int,
                                  help="Latent dimensionality of LSTMs, affects both models.")

    return parser


def get_rgan_parser() -> argparse.ArgumentParser:
    defaults = {
        # General
        "epochs": 200,
        "batch_size": 28,

        # Optimizer
        "opt_g": "adam",
        "lr_g": 1e-3,
        "opt_d": "sgd",
        "lr_d": 0.1,
        "beta1": 0.5,
        "beta2": 0.999,

        # Architecture
        "latent_dim": 100,
        "noise_dim": 5,
        "rnn": 'lstm',
        "n_layers": 1,

        # Training
        "n_critic": 1,
        "clip_value": 0.01,
    }
    parser = get_base_parser(gan=True, dp=False)

    # ### Architecture-specific Options ###
    architecture_group = parser.add_argument_group("Architecture")
    architecture_group.add_argument("--latent_dim", type=int, help="Latent dimension")
    architecture_group.add_argument("--noise_dim", type=int, help="Noise dimension")
    architecture_group.add_argument("--output_dim", type=int, help="Output dimension")
    architecture_group.add_argument("--rnn", choices=[rnn.value for rnn in RNNModels], help="RNN Type.")
    architecture_group.add_argument("--n_layers", type=int, help="Number of recurrent layers")

    parser.set_defaults(**defaults)
    return parser


def get_trajgan_parser() -> argparse.ArgumentParser:
    defaults = {
        # General
        "epochs": 200,
        "batch_size": 256,
        # Optimizer
        "lr_d": 0.005,
        "lr_g": 0.005,
        "beta1": 0.5,
        "beta2": 0.999,
        # Training
        "shuffle": True,
        "clip_value": 0.01,
    }

    parser = get_base_parser(gan=True, dp=True)

    # Architecture-specific Options
    # Add a boolean flag for trajectory-shaped noise
    architecture_group = parser.add_argument_group("Architecture")
    architecture_group.add_argument("--traj_noise", action="store_true", dest="traj_shaped_noise",
                                    help="Use trajectory-shaped noise")
    # Add n_layer for number of recurrent layers
    architecture_group.add_argument("--n_layers", type=int, default=1,
                                    help="Number of recurrent layers")
    # Add latent_dim and noise_dim
    architecture_group.add_argument("--latent_dim", type=int, default=100,
                                    help="Latent dimension")
    architecture_group.add_argument("--noise_dim", type=int, default=100,
                                    help="Noise dimension")

    parser.set_defaults(**defaults)
    return parser


def get_rnn_parser() -> argparse.ArgumentParser:
    defaults = {
        # General
        "epochs": 300,
        "batch_size": 512,

        # Optimizer
        "opt": "adamw",
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,

        # Architecture
        "input_dim": 2,
        "hidden_size": 100,
        "rnn": "lstm",
        "n_layers": 1,

        # Training
        "clip_value": 0.,

        # Config
        "save_freq": 50,
        "plot_freq": 200,
        "tensorboard": True,
    }
    parser = get_base_parser(dp=False, gan=False)

    # ### Architecture-specific Options ###
    architecture_group = parser.add_argument_group("Architecture")
    architecture_group.add_argument("--input_dim", type=int,
                                    help="Input Dimension.")
    architecture_group.add_argument("--embedding_dim", type=int,
                                    help="Embedding Dimension. [Default: input_dim]")
    architecture_group.add_argument("--hidden_size", type=int,
                                    help="Hidden Dimension.")
    architecture_group.add_argument("--output_dim", type=int,
                                    help="Output Dimension. [Default: input_dim]")
    # RNN parameters
    architecture_group.add_argument("--rnn", choices=[rnn.value for rnn in RNNModels],
                                    help="RNN Type.")
    architecture_group.add_argument("--n_layers", type=int, help="Number of recurrent layers")

    # Add arguments to decide between models
    parser.add_argument("--model", choices=[m.value for m in RNNModels], required=True,
                        help="Model to train.")

    # Optimizer
    opt_group = parser.add_argument_group("Optimizer")
    opt_group.add_argument("--opt", choices=[o.value for o in Optimizer], help="Optimizer.")
    opt_group.add_argument("--lr", type=float, help="Learning rate.")

    parser.set_defaults(**defaults)
    return parser


def get_cnnGAN_parser() -> argparse.ArgumentParser:
    defaults = {
        "batch_size": 32,
        "plot_freq": 1000,
        "clip_value": 0.01,
        "beta1": 0.5,
        "beta2": 0.999,
        "lr_g": 1e-4,
        "lr_d": 3e-4,
        "n_critic": 1,
        "noise_dim": 100,
        "use_batch_norm": False,  # Should not be used in combination with WGAN-GP
    }
    parser = get_base_parser(dp=False, gan=True)

    # ### Architecture-specific Options ###
    architecture_group = parser.add_argument_group("Architecture")
    architecture_group.add_argument("--output_dim", type=int, help="Output dimension")
    architecture_group.add_argument("--noise_dim", type=int, help="Input Noise dimension")
    architecture_group.add_argument("--use_batch_norm", action='store_true',
                                    dest="use_batch_norm", help="Use Batch Normalization")

    parser.set_defaults(**defaults)
    return parser
