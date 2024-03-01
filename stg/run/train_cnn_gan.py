#!/usr/bin/env python3
"""Train CNN GAN Model."""
import json
import logging

import torch
from torch.utils.data import DataLoader

from stg.datasets import mnist_sequential, DatasetModes
from stg.datasets.dataset_factory import Datasets, get_dataset
from stg.datasets.padding import ZeroPadding
from stg.models.cnn_gan import CNN_GAN
from stg.parser import get_cnnGAN_parser, clean_args
from stg.utils.parser import load_config_file
from stg.utils import logger

if __name__ == '__main__':
    parser = get_cnnGAN_parser()
    opt = vars(parser.parse_args())
    if 'config' in opt and opt['config'] is not None:
        opt = load_config_file(opt)
    print("Arguments: ", json.dumps(opt, indent=4))

    log = logger.configure_root_loger(logging_level=opt.pop('logging_lvl'))

    # Prepare Data
    collate_fn = None
    dataset_name = opt.pop('dataset')
    if dataset_name == Datasets.MNIST_SEQUENTIAL:
        output_dim = opt.pop('output_dim', 28)
        output_dim = 28 if output_dim is None else output_dim
        dataset = mnist_sequential(output_dim)
        max_length = 28 * 28 // output_dim
    else:
        # Trajectory Dataset
        RETURN_LABELS = True
        dataset = get_dataset(
            dataset_name=dataset_name,
            mode=DatasetModes.ALL,
            latlon_only=True,
            normalize=True,
            return_labels=RETURN_LABELS,
            keep_original=False
        )
        # Padding
        max_length = dataset.max_len
        collate_fn = ZeroPadding(return_len=True, return_labels=RETURN_LABELS, fixed_length=max_length)
        output_dim = dataset[0][0].shape[-1]
        opt.pop('output_dim')

    # Create Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=opt.pop('batch_size'),
                            drop_last=True,
                            shuffle=opt.pop('shuffle'),
                            collate_fn=collate_fn,
                            pin_memory=True
                            )

    # Create GAN
    cnn_gan = CNN_GAN(
        gpu=opt.pop('gpu'),
        output_length=max_length,
        output_dim=output_dim,
        noise_dim=opt.pop('noise_dim'),
        name=opt.pop('name'),
        use_batch_norm=opt.pop('use_batch_norm'),
    )

    # Remove all used values from opt
    opt = clean_args(opt)

    # Training
    cnn_gan.training_loop(
        dataloader=dataloader,
        dataset_name=dataset_name,
        notebook=False,
        **opt
    )
