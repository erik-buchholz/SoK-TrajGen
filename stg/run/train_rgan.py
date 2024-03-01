#!/usr/bin/env python3
"""Train RGAN Model."""
import json

from torch.utils.data import DataLoader

from stg.datasets import mnist_sequential, DatasetModes
from stg.datasets.dataset_factory import Datasets, get_dataset
from stg.datasets.padding import ZeroPadding
from stg.models import RGAN
from stg.parser import get_rgan_parser, clean_args
from stg.utils import logger
from stg.utils.parser import load_config_file

if __name__ == '__main__':
    parser = get_rgan_parser()
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
        collate_fn = ZeroPadding(return_len=True, return_labels=RETURN_LABELS)
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
    hidden_size, rnn_type, n_layers = opt.pop('latent_dim'), opt.pop('rnn'), opt.pop('n_layers')
    model = RGAN(
        gpu=opt.pop('gpu'),
        noise_dim=opt.pop('noise_dim'),
        output_dim=output_dim,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        num_layers=n_layers,
    )

    # Remove all used values from opt
    opt = clean_args(opt)

    # Training
    model.training_loop(
        dataloader=dataloader,
        dataset_name=dataset_name,
        **opt
    )
