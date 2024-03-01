#!/usr/bin/env python3
"""Evaluation III: RAoPT vs LSTM"""
import argparse
import json
import logging
import random
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from stg import config
from stg.datasets import pad_feature_first, random_split, \
    DATASET_CLASSES, get_dataset
from stg.eval.shared import determine_epochs, fix_seeds
from stg.models.lstm_trajgan import LSTM_TrajGAN
from stg.models.traj_loss import TrajLoss
from stg.utils import logger

# General Parameters
N_FOLD = 5
OUTPUT_DIR = config.RESULT_DIR + 'raopt_vs_lstm/'

# LSTM-TrajGAN Parameters
# LT_EPOCHS = 250  # Computed below
# LT_MAX_LEN = 144  # Dataset Dependent -> defined below
LT_LATENT_DIM = 100
LT_BATCH_SIZE = 256
LT_LR = 0.001
LT_BETA = 0.5
LT_NAME = 'LSTM_TrajGAN_PT'

# RAoPT Parameters
RAoPT_EPOCHS = 1500
RAoPT_EARLY_STOP = 100
RAoPT_BATCH_SIZE = 256
# Add RAoPT to path
sys.path.append(config.BASE_DIR + 'RAoPT')

fix_seeds(42)

# Get logger
log = logging.getLogger()


def run_eval(opt: dict):
    start_time = timer()
    if opt['gpu_tf'] == opt['gpu_pt'] and opt['gpu_tf'] > -1:
        raise ValueError("GPU for PyTorch and TensorFlow must be different.")

    # Define TF GPU to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['gpu_tf'])  # This also affects PyTorch

    # noinspection PyUnresolvedReferences
    import tensorflow as tf

    # Fix tensorflow seed for reproducibility
    tf.random.set_seed(42)

    if opt['gpu_tf'] > -1:
        # Set TensorFlow GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[opt['gpu_tf']], 'GPU')

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    # Output file uses current date and time - so we won't overwrite anything and can concatenate later
    OUTPUT_FILE = OUTPUT_DIR + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{opt['dataset']}_{opt['batch_size']}.csv"
    METADATA_FILE = OUTPUT_DIR + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{opt['dataset']}_{opt['batch_size']}.json"
    print(f"Saving results to:\t", OUTPUT_FILE)

    # Load dataset
    all_data = get_dataset(opt['dataset'], latlon_only=opt['latlon_only'])

    # Print Dataset Information
    print("Total Samples:\t\t", len(all_data))
    print("Reference Point:\t", all_data.reference_point)
    print("Scaling Factor:\t\t", all_data.scale_factor)

    for run in range(1, opt['runs'] + 1):
        print("#" * 15, "Starting Run #", run, "#" * 15)
        start_time_run = timer()

        # Randomly split into train and test with 50/50 ratio
        # We use such a large ratio because we need to split the predicted data into train and test later for
        # RAoPT evaluation
        train_data, generation_data = random_split(all_data, [0.5, 0.5])

        print("LSTM-TrajGAN Train:\t", len(train_data))
        # As we fixed the seed, we can print the first 5 TIDs to verify that the split is reproducible
        print("First 5 TIDs:\t\t", sorted(train_data.tids)[:5])
        print("LSTM-TrajGAN Gen:\t", len(generation_data))
        print("First 5 TIDs:\t\t", sorted(generation_data.tids)[:5])
        assert len(train_data) + len(generation_data) == len(all_data), "Splitting failed."
        # Make sure every TID is only in one of the two datasets
        assert len(set(train_data.tids).intersection(set(generation_data.tids))) == 0, "Splitting failed."

        # Create LSTM-TrajGAN model with parameters from paper
        gan = LSTM_TrajGAN(
            reference_point=all_data.reference_point,
            features=all_data.features,
            scale_factor=all_data.scale_factor,
            gpu=opt['gpu_pt'],
            # Have to use 0 in case of TF usage because the device is set up via environment variable for TF,
            learning_rate=LT_LR,
            beta=LT_BETA,
            latent_dim=LT_LATENT_DIM,
            model_name=LT_NAME + "_" + str(opt['dataset']).upper(),
        )

        if opt['dataset'] == 'geolife':
            # Overwrite loss function, because we only consider spatial information
            gan.gen_loss = TrajLoss(p_hour=0, p_cat=0, p_dow=0)

        # Create DataLoaders
        train_dl = DataLoader(train_data, batch_size=LT_BATCH_SIZE, shuffle=True, drop_last=True,
                              collate_fn=pad_feature_first)

        # Compute number of epochs equal to 2000 batches as used in paper
        LT_EPOCHS = int(2000 / len(train_dl))
        print("LSTM-TrajGAN Epochs:\t", LT_EPOCHS)

        # Train LSTM-TrajGAN
        if opt['load_lt']:
            log.warning("Loading LSTM-TrajGAN parameters from file. Should be used for debugging only!")
            gan.load_parameters(epoch=LT_EPOCHS)
        else:
            gan.training_loop(
                train_dl,
                epochs=LT_EPOCHS,
                save_freq=LT_EPOCHS,  # Save only the last model state
                print_training=False
            )

        # Predict and Convert all data in case FS-NYC is used because of small dataset size
        # log.warning("Predicting and converting all data. Should be used for debugging only!")
        # generation_data = all_data

        # Predict and Convert generation set
        synthetic_data: pd.DataFrame = gan.predict_and_convert(generation_data, )

        # Free up GPU memory used by PyTorch
        del gan
        torch.cuda.empty_cache()

        # Split generation data into train and test set for RAoPT, proportion 2:1
        train_original, test_original = random_split(generation_data, [2 / 3, 1 / 3])

        # Create pandas DataFrames for Original Values
        train_original_df = pd.concat(train_original.originals.values())
        test_original_df = pd.concat(test_original.originals.values())

        # Split the generated data into the same sets
        train_mask = synthetic_data['tid'].isin(train_original.tids)
        train_synthetic = synthetic_data[train_mask]
        test_synthetic = synthetic_data[~train_mask]
        assert len(train_synthetic) + len(test_synthetic) == len(synthetic_data), \
            f"Splitting failed: {len(train_synthetic)} + {len(test_synthetic)} != {len(synthetic_data)}"
        assert len(set(train_synthetic['tid']).intersection(set(test_synthetic['tid']))) == 0, \
            f"Splitting failed: {len(set(train_synthetic['tid']).intersection(set(test_synthetic['tid'])))}"
        # Verify that test_synthetic contains all TIDs from test_original
        assert len(set(test_original.tids).difference(set(test_synthetic['tid']))) == 0, \
            f"Splitting failed: {len(set(test_original.tids).difference(set(test_synthetic['tid'])))}"
        assert len(train_original_df) == len(train_synthetic), \
            f"Splitting failed: {len(train_original_df)} != {len(train_synthetic)}"
        assert len(test_original_df) == len(test_synthetic), \
            f"Splitting failed: {len(test_original_df)} != {len(test_synthetic)}"

        # Column names required for RAoPT
        column_renaming = {
            'lat': 'latitude',
            'lon': 'longitude',
            'label': 'uid',
            'tid': 'trajectory_id',
        }

        # Save the original & generated datasets for RAoPT
        data_transfer_dir = config.BASE_DIR + 'data/raopt/'
        Path(data_transfer_dir).mkdir(parents=True, exist_ok=True)
        file_train_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file_test_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file_train_syn = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file_test_syn = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tempfiles = [file_train_orig, file_test_orig, file_train_syn, file_test_syn]

        # Store originals
        train_original_df.rename(columns=column_renaming).to_csv(file_train_orig.name, index=False)
        test_original_df.rename(columns=column_renaming).to_csv(file_test_orig.name, index=False)
        # Store predicted
        train_synthetic.rename(columns=column_renaming).to_csv(file_train_syn.name, index=False)
        test_synthetic.rename(columns=column_renaming).to_csv(file_test_syn.name, index=False)

        # Import RAoPT
        # noinspection PyUnresolvedReferences
        from raopt.ml.model import AttackModel as RAoPT
        # noinspection PyUnresolvedReferences
        from raopt.ml import encoder
        # noinspection PyUnresolvedReferences
        from raopt.utils import helpers as raopt_helpers
        # noinspection PyUnresolvedReferences
        from raopt.eval import main as eval_main

        # Load datasets in correct format
        train_orig = raopt_helpers.read_trajectories_from_csv(file_train_orig.name)
        train_syn = raopt_helpers.read_trajectories_from_csv(file_train_syn.name)

        # Encode (ignore time)
        encoded_originals = encoder.encode_trajectory_dict(train_orig, ignore_time=True)
        encoded_protected = encoder.encode_trajectory_dict(train_syn, ignore_time=True)
        keys = list(encoded_protected.keys())
        assert len(keys) == len(train_original)
        assert set(keys) == set(train_synthetic['tid']), "Encoding failed."
        trainX = [encoded_protected[i] for i in keys]
        trainY = [encoded_originals[i] for i in keys]
        print("RAoPT Train Samples:\t", len(trainX))

        if all_data.columns[0] == 'lon':
            # Swap ref and sf because RAoPT expects latitude first
            ref = (all_data.reference_point[1], all_data.reference_point[0])
            sf = (all_data.scale_factor[1], all_data.scale_factor[0])
        else:
            ref = all_data.reference_point
            sf = all_data.scale_factor

        # Create and train model
        raopt_features = ['latlon']  # We ignore time for RAoPT for simplicity
        raopt = RAoPT(
            max_length=all_data.max_len,
            features=raopt_features,
            scale_factor=sf,
            reference_point=ref,
            parameter_file=config.BASE_DIR + f'parameters/RAoPT/{str(opt["dataset"])}/parameters_{run}.hdf5',
        )

        # Debug
        # print('Model Summary:')
        # print(raopt.model.summary())

        epochs, num_batches = determine_epochs(opt['epochs'], opt['num_batches'], len(trainX))
        print("RAoPT Epochs:\t\t", epochs)
        print("RAoPT Num Batches:\t", num_batches)

        # Use default parameters
        h = raopt.train(
            trainX,
            trainY,
            use_val_loss=False,  # Save as many samples as possible
            early_stopping=opt['early_stop'],
            epochs=epochs,
            batch_size=opt['batch_size'],
        )

        # Print stopping epoch
        print(f"Training Completed After {len(h.history['loss'])} Epochs.")

        # Evaluation

        # Load test data
        test_syn = raopt_helpers.read_trajectories_from_csv(file_test_syn.name)
        print("RAoPT Test Samples:\t", len(test_syn))

        # Reconstruct
        reconstructed = raopt.predict(x=list(test_syn.values()))

        # Clear GPU Memory used by TensorFlow
        del raopt
        tf.keras.backend.clear_session()

        # Measure
        reconstructed: Dict[str, pd.DataFrame] = {str(p['trajectory_id'][0]): p for p in reconstructed}
        originals = raopt_helpers.read_trajectories_from_csv(file_test_orig.name)
        distances = eval_main.parallelized_distance_computation(
            test_orig=originals, reconstructed=reconstructed, test_p=test_syn, fold=run)

        # Store results
        df = pd.DataFrame(distances)
        # Rename columns fold to 'run'
        df.rename(columns={'fold': 'run'}, inplace=True)
        # Append results
        df.to_csv(OUTPUT_FILE, mode='a', header=(run == 1), index=False)
        log.info(f"Wrote distances to: {OUTPUT_FILE}")

        # Store metadata
        metadata = {
            'dataset': opt['dataset'],
            'n_lt_train': len(train_data),
            'n_lt_gen': len(generation_data),
            'n_runs': opt['runs'],
            'lt_features': all_data.features,
            'lt_lr': LT_LR,
            'lt_beta': LT_BETA,
            'lt_latent_dim': LT_LATENT_DIM,
            'lt_epochs': LT_EPOCHS,
            'lt_batch_size': LT_BATCH_SIZE,
            'n_raopt_train': len(train_original),
            'n_raopt_test': len(test_original),
            'raopt_features': ['latlon'],
            'raopt_max_len': all_data.max_len,
            'raopt_epochs': epochs,
            'raopt_batch_size': opt['batch_size'],
            'raopt_early_stop': opt['early_stop'],
            'raopt_num_batches': num_batches,
            'runtime_per_run': f"{(timer() - start_time) / run:.2f}s",
            'runtime_total': f"{timer() - start_time:.2f}s",
        }
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)

        eval_main.print_results_detailed(df)
        print(f"Run {run} took:\t{timedelta(seconds=int(timer() - start_time_run))}.")

        # Close the temporary files
        for f in tempfiles:
            f.close()

    # Print Final Results over all runs
    print("#" * 15, "Final Results", "#" * 15)
    df = pd.read_csv(OUTPUT_FILE)
    eval_main.print_results_detailed(df)
    # Print total time in format hh:mm:ss
    print(f"Total Time:\t\t{timedelta(seconds=int(timer() - start_time))}")
    print("#" * 15, "End of Evaluation", "#" * 15)


if __name__ == '__main__':
    logger.configure_root_loger(logging_level=logging.WARNING)  # Configure root logger
    # Argument parser via argparse
    parser = argparse.ArgumentParser(description='RAoPT vs LSTM-TrajGAN.')
    parser.add_argument('-g', '--gpu_tf', type=int, default=0, help='GPU to use for TensorFlow.')
    parser.add_argument('-p', '--gpu_pt', type=int, default=1, help='GPU to use for PyTorch.')
    parser.add_argument('-r', '--runs', type=int, default=N_FOLD, help='Number of runs to perform.')
    parser.add_argument('--load_lt', action='store_true', help='Load LSTM-TrajGAN parameters from file.')
    parser.add_argument('--print', type=str, default=None, metavar='FILE',
                        help='Print summary of the results from FILE.')
    parser.add_argument('-d', '--dataset', type=str, default='fs', choices=DATASET_CLASSES.keys(),
                        help='Dataset to use for evaluation.')
    # RAoPT batch size
    parser.add_argument('--batch_size', type=int, default=RAoPT_BATCH_SIZE, help='RAoPT batch size.')
    # RAoPT early stopping
    parser.add_argument('--early_stop', type=int, default=RAoPT_EARLY_STOP, help='RAoPT early stopping.')
    # RAoPT epochs
    parser.add_argument('--epochs', type=int, default=RAoPT_EPOCHS, help='RAoPT epochs.')
    # RAoPT num_batches (takes precedence)
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Number of batches to train RAoPT for. Takes precedence over number of epochs.')
    # Whether to use latlon only for LSTM-TrajGAN
    parser.add_argument('--latlon_only', action='store_true', help='Use only latlon features for LSTM-TrajGAN.')

    opt = vars(parser.parse_args())
    if opt['print'] is not None:
        # noinspection PyUnresolvedReferences
        from raopt.eval import main as eval_main

        df = pd.read_csv(OUTPUT_DIR + opt['print'])
        eval_main.print_results_detailed(df)
    else:
        run_eval(opt)
