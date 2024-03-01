#!/usr/bin/env python3
"""Train the GeoTrajGAN model."""
import json

from torch.utils.data import DataLoader

from stg.datasets import mnist_sequential, DatasetModes
from stg.datasets.dataset_factory import Datasets, get_dataset
from stg.datasets.padding import ZeroPadding
from stg.models import GeoTrajGAN
from stg.parser import get_geotrajgan_parser, clean_args
from stg.utils import logger
from stg.utils.parser import load_config_file

if __name__ == '__main__':
    parser = get_geotrajgan_parser()
    opt = vars(parser.parse_args())
    if 'config' in opt and opt['config'] is not None:
        opt = load_config_file(opt)
    print("Arguments: ", json.dumps(opt, indent=4))

    log = logger.configure_root_loger(logging_level=opt.pop('logging_lvl'))

    # Prepare Data
    collate_fn = None
    dataset_name = opt.pop('dataset')
    if dataset_name == Datasets.MNIST_SEQUENTIAL:
        output_dim = 28
        dataset = mnist_sequential(output_dim)
    else:
        # Trajectory Dataset
        dataset = get_dataset(
            dataset_name=dataset_name,
            mode=DatasetModes.ALL,
            latlon_only=True,
            normalize=True,
            return_labels=True,
            keep_original=False
        )
        # Padding
        collate_fn = ZeroPadding(return_len=True, return_labels=True)
        output_dim = dataset[0][0].shape[-1]
    opt.pop('n_dim')

    # Create Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=opt.pop('batch_size'),
                            drop_last=True,
                            shuffle=opt.pop('shuffle'),
                            collate_fn=collate_fn,
                            pin_memory=True
                            )

    # Create GAN
    gtg = GeoTrajGAN(
        gpu=opt.pop('gpu'),
        n_dim=output_dim,
        latent_dim_g=opt.pop('latent_dim_g'),
        latent_dim_d=opt.pop('latent_dim_d'),
        norm=opt.pop('norm'),
        activation=opt.pop('activation'),
        conv_only=opt.pop('conv_only'),
        bi_lstm=opt.pop('bi_lstm'),
        bi_lstm_merge_mode=opt.pop('bi_lstm_merge_mode'),
        sequential_mode=opt.pop('sequential_mode'),
        generator_lstm=opt.pop('generator_lstm'),
        discriminator_lstm=opt.pop('discriminator_lstm'),
        use_traj_discriminator=opt.pop('use_traj_discriminator'),
        use_stn_in_point_dis=opt.pop('use_stn_in_point_dis'),
        use_stn_in_traj_dis=opt.pop('use_stn_in_traj_dis'),
        share_pointnet=opt.pop('share_pointnet'),
        lstm_latent_dim=opt.pop('lstm_latent_dim')
    )

    # Remove all used values from opt
    opt = clean_args(opt)

    # Training
    gtg.training_loop(
        dataloader=dataloader,
        dataset_name=dataset_name,
        **opt
    )
