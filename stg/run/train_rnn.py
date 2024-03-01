#!/usr/bin/env python3
"""Train RNN-based Model."""
import json

from torch.utils.data import DataLoader

from stg.datasets import mnist_sequential, DatasetModes
from stg.datasets.dataset_factory import Datasets, get_dataset
from stg.datasets.padding import ZeroPadding
from stg.models import RNN_MODEL_CLASSES
from stg.models.base_rnn import BaseRNN
from stg.parser import get_rnn_parser, clean_args
from stg.utils import logger
from stg.utils.parser import load_config_file

if __name__ == '__main__':
    parser = get_rnn_parser()
    opt = vars(parser.parse_args())
    if 'config' in opt and opt['config'] is not None:
        opt = load_config_file(opt)
    print("Arguments: ", json.dumps(opt, indent=4))

    log = logger.configure_root_loger(logging_level=opt.pop('logging_lvl'))

    # Device
    if opt['gpu'] > -1:
        DEVICE = f"cuda:{opt.pop('gpu')}"
    else:
        DEVICE = "cpu"
        opt.pop('gpu')

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

    # Create Model
    Generator: BaseRNN = RNN_MODEL_CLASSES[opt.pop('model')](
        input_dim=opt.pop('input_dim'),
        output_dim=output_dim,
        embedding_dim=opt.pop('embedding_dim'),
        hidden_size=opt.pop('hidden_size'),
        num_layers=opt.pop('n_layers'),
        dropout=0.0,
        rnn_type=opt.pop('rnn'),
        bidirectional=False,
    )

    # Clean up opt
    opt = clean_args(opt)
    print("Training Parameters:\t", opt)

    # Training
    Generator.training_loop(
        dataloader=dataloader,
        dataset_name=dataset_name,
        device=DEVICE,
        notebook=False,
        **opt
    )
