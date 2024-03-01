#!/usr/bin/env python3
"""Implements a TensorFlow v2 version of LSTM-TrajGAN as baseline.
Refer to https://github.com/GeoDS/LSTM-TrajGAN/ for the original
implementation.
"""
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import os

import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from stg import config
from stg.utils import logger
from stg.utils.helpers import get_ref_point, get_scaling_factor

# CONSTANTS ####################################################################
FEATURES = ['latlon', 'hour', 'dow', 'category']
VOCAB_SIZE = {
    'latlon': 2,
    'hour': 24,
    'dow': 7,
    'category': 10
}
EMBEDDING_SIZE = {
    'latlon': 64,
    'hour': 24,
    'dow': 7,
    'category': 10
}
log = logging.getLogger()
# Constants from Paper
MAX_LEN = 144
LATENT_DIM = 100
EPOCHS = 250
LEARNING_RATE = 0.001
BETA = 0.5
BATCH_SIZE = 256


def to_trajGAN_input(
        trajectories: List[pd.DataFrame] or pd.DataFrame,
        ref_point: (float, float),
        num_cat: int = 10,
        lat_label: str = 'lat',
        lon_label: str = 'lon',
        tid_label: str = 'tid',
        # uid_label: str = 'uid',
        hour_label: str = 'hour',
        dow_label: str = 'day',
        cat_label: str = 'category'
) -> List[np.ndarray]:
    # If list is passed, concatenate
    if type(trajectories) is list or type(trajectories) is tuple:
        trajectories = pd.concat(trajectories)
    elif type(trajectories) is dict:
        trajectories = pd.concat(trajectories.values())
    elif type(trajectories) is not pd.DataFrame:
        raise ValueError(f"Unsupported input type: {type(trajectories)}")

    result = [
        [],  # lat_lon
        [],  # HoW
        [],  # DoW
    ]

    # Input Validation
    if cat_label in trajectories:
        result.append([])  # category
        assert num_cat == trajectories[cat_label].nunique(), "Number of categories does not match"
    assert trajectories[hour_label].nunique() <= 24, "More than 24 values for Hour of Day"
    assert trajectories[dow_label].nunique() <= 7, "More than 7 values for Day of Week"

    for tid, df in trajectories.groupby(tid_label):
        result[0].append(df[[lat_label, lon_label]].to_numpy() - ref_point)
        # Note: Reference point is subtracted, but data is not scaled to [-1;1]
        result[1].append(np.eye(24)[df[hour_label].to_numpy()])
        result[2].append(np.eye(7)[df[dow_label].to_numpy()])
        if cat_label in trajectories:
            result[3].append(np.eye(num_cat)[df[cat_label].to_numpy()])
        # Mask: 1 for each unpadded time
        # result[-1].append(np.array([[1.] for _ in range(len(df))]))
    return np.array(result, dtype=object)


class LSTM_TrajGAN:
    def __init__(
            self,
            reference_point: (float, float),
            scale_factor: (float, float),
            max_length: int,
            features: List[str] = FEATURES,
            vocab_size: Dict[str, int] = VOCAB_SIZE,
            embedding_size: Dict[str, int] = EMBEDDING_SIZE,
            parameter_file: str = None,
            latent_dim: int = 100,
            masking: bool = False,
            model_name: str = 'LSTM_TrajGAN_TF',
            learning_rate: float = LEARNING_RATE,
            beta: float = BETA,
            use_regularizer: bool = True,
            scale_in_model: bool = True
    ):
        """
        Initialize the model
        :param max_length: Maximal length of any trajectory (for padding)
        :param features: List of features used by the model. First feature has to be latlon!
        :param vocab_size: Dict stating the number of values for each feature
                           (e.g., 7 for an onehot encoded day of week)
        :param scale_factor: Scale factor for latitude and longitude (lat: float, lon: float)
        :param parameter_file: File to save parameters of the model
        :param reference_point: (lat, lon)
        :param use_regularizer: Whether to use regularization in the loss function and during recurrent layers.
                                Reason: For DP training, all recurrent layers have to be deactivated
        :param scale_in_model: Whether to scale the generator's output from [-1;1] to the original range within the
                                generator or outside (in the predict method). The default is to scale within the model.
        """
        from keras.optimizers import Adam
        from keras import Input, Model
        from stg.ml_tf.loss import trajLoss
        import tensorflow as tf

        self.latent_dim = latent_dim
        self.max_length = max_length

        self.features = features
        self.vocab_size = vocab_size
        self.num_features = sum(self.vocab_size[k] for k in features)
        self.embedding_size = embedding_size

        self.lat0, self.lon0 = reference_point
        self.scale_factor = scale_factor
        self.scale_in_model = scale_in_model
        if not self.scale_in_model:
            log.warning("Scaling is deactivated. Make sure to normalize the data before passing it to the model.")

        self.param_file = parameter_file
        self.masking = masking
        self.model_name = f'masked_{model_name}' if masking else model_name
        self.param_path = config.PARAM_PATH + self.model_name + '/'
        Path(self.param_path).mkdir(exist_ok=True, parents=True)

        self.learning_rate = learning_rate
        self.beta = beta

        # For DP training to work, we need to disable regularization
        # See: https://github.com/tensorflow/privacy/issues/180
        self.use_regularizer = use_regularizer

        self.optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta)

        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dim,), name='input_noise')
        inputs = []
        for idx, feature in enumerate(self.features):
            i = Input(shape=(self.max_length, self.vocab_size[feature]), name='Input_' + feature)
            inputs.append(i)
        inputs.append(noise)

        # gen_trajs has one component for each feature in self.features (default: 4)
        gen_trajs = self.generator(inputs)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=tf.keras.losses.binary_crossentropy,
                                   optimizer=self.optimizer, metrics=['accuracy'])

        # The combined model only trains the trajectory generator
        self.discriminator.trainable = False

        # The discriminator takes generated trajectories as input and makes predictions
        pred = self.discriminator(gen_trajs)

        # Loss function in TF2 does not work as in TF1
        y_true = Input(shape=(1,), name="True Labels")
        mod_inputs = inputs + [y_true]
        # The combined model (combining the generator and the discriminator)
        self.combined = Model(mod_inputs, pred, name=f"{self.model_name}_COM")
        self.combined.add_loss(trajLoss(y_true=y_true, y_pred=pred, real_traj=inputs, gen_traj=gen_trajs,
                                        use_regularizer=self.use_regularizer))

        # We cannot us the same optimizer object in TF2 for discriminator and combined
        # and the loss function has to be modified.
        # self.combined.compile(loss=trajLoss(inputs, gen_trajs), optimizer=self.optimizer)
        self.combined.compile(loss=None, optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta))
        # Save Models----------------------------------------------------------
        # *Composed model cannot be saved easily since TFv2*
        # Create Directories for parameters
        self.combined_param_path = self.param_path + 'C_Model/'
        self.generator_param_path = self.param_path + 'G_Model/'
        self.discriminator_param_path = self.param_path + 'D_Model/'
        for d in [
            self.combined_param_path,
            self.generator_param_path,
            self.discriminator_param_path
        ]:
            Path(d).mkdir(exist_ok=True)
        # ----------------------------------------------------------------------

    def build_generator(self):
        """
        Return an LSTM-based model that reconstructs original trajectories from protected versions.

        :return: The tensorflow.keras.models.Model.
        """
        from keras import Model
        from keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate, LSTM, Lambda
        from keras.regularizers import l1
        import tensorflow as tf

        # Input layer-----------------------------------------------------------
        inputs = []
        for i, feature in enumerate(self.features):
            input_ = Input(shape=(self.max_length, self.vocab_size[feature]),
                           name=f"Input_{feature}")
            inputs.append(input_)
        # Noise
        noise = Input(shape=(self.latent_dim,), name='Input_noise')

        # Embedding Layer-------------------------------------------------------
        embeddings = []
        for i, feature in enumerate(self.features):
            emb = Dense(units=self.embedding_size[feature],
                        activation='relu',
                        name=f'Embedding_{feature}_dense')
            if self.masking:
                # Masking Layer! (Inputs are padded)------------------------------------
                # For some reason, the masking layer produces a warning that can be prevented
                # by commenting the following line: https://github.com/tensorflow/tensorflow/issues/57052
                embedding = TimeDistributed(emb, name=f'Embedding_{feature}')(
                    Masking(name=f"Mask_{feature}", mask_value=0.0)(inputs[i])
                )
            else:
                embedding = TimeDistributed(emb, name=f'Embedding_{feature}')(inputs[i])
            embeddings.append(embedding)
        # ----------------------------------------------------------------------
        inputs.append(noise)

        # Feature Fusion -------------------------------------------------------
        # Some datasets only contain latitude and longitude. In that case, we do
        # not need to fuse any features
        if len(embeddings) > 1:
            concatenation = Concatenate(axis=-1, name="Join_Features")(embeddings)
        else:
            concatenation = embeddings[0]

        # Now we only have concatenated the locations. However, we still need to
        # add the noise to each timestamp
        repeated_noise = tf.repeat(tf.reshape(noise, shape=(-1, 1, 100)), self.max_length, axis=1)
        full_concat = Concatenate(axis=-1, name="Concat_Noise")([concatenation, repeated_noise])
        # full_concat = concatenation

        feature_fusion = TimeDistributed(Dense(
            units=100,
            activation='relu'),
            name='Feature_Fusion'
        )(full_concat)
        # ----------------------------------------------------------------------

        # Bidirectional LSTM layer ---------------------------------------------
        rec_regularizer = l1(0.02) if self.use_regularizer else None
        bidirectional_lstm = LSTM(units=100,
                                  return_sequences=True,
                                  recurrent_regularizer=rec_regularizer
                                  )(feature_fusion)
        # ----------------------------------------------------------------------

        # Output Layer ---------------------------------------------------------
        output_lat = TimeDistributed(
            Dense(1, activation='tanh'), name='Output_lat')(bidirectional_lstm)
        output_lon = TimeDistributed(
            Dense(1, activation='tanh'), name='Output_lon')(bidirectional_lstm)
        # We need to scale up because tanh outputs are between -1 an 1
        if self.scale_in_model:
            rescaling1 = Lambda(lambda x: x * self.scale_factor[0])
            rescaling2 = Lambda(lambda x: x * self.scale_factor[1])
            lat_scaled = TimeDistributed(rescaling1,
                                         name='Output_lat_scaled')(output_lat)
            lon_scaled = TimeDistributed(rescaling2,
                                         name='Output_lon_scaled')(output_lon)
        else:
            # Output range is [-1;1]: Data needs to be scaled accordingly
            lat_scaled = output_lat
            lon_scaled = output_lon
        latlon_scaled = Concatenate(axis=-1)([lat_scaled, lon_scaled])
        outputs = [latlon_scaled, ]

        for feature in self.features[1:]:
            out = TimeDistributed(Dense(
                self.vocab_size[f'{feature}'], activation='softmax'), name=f'Output_{feature}')(bidirectional_lstm)
            outputs.append(out)

        model = Model(inputs=inputs, outputs=outputs,
                      name=f"{self.model_name}_GEN")

        return model

    def build_discriminator(self):
        from keras import Model
        from keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate, LSTM
        from keras.regularizers import l1

        # Input layer-----------------------------------------------------------
        inputs = []
        for i, feature in enumerate(self.features):
            input_ = Input(shape=(self.max_length, self.vocab_size[feature]),
                           name=f"Input_{feature}")
            inputs.append(input_)

        # Masking Layer! (Inputs are padded)------------------------------------
        masked = [Masking(name=f"Mask_{self.features[i]}")(x) for i, x in enumerate(inputs)]

        # Embedding Layer-------------------------------------------------------
        embeddings = []
        for i, feature in enumerate(self.features):
            emb = Dense(units=self.embedding_size[feature],
                        activation='relu',
                        name=f'Embedding_{feature}_dense')
            embedding = TimeDistributed(emb, name=f'Embedding_{feature}')(masked[i])
            embeddings.append(embedding)
        # ----------------------------------------------------------------------

        # Feature Fusion -------------------------------------------------------
        # Some datasets only contain latitude and longitude. In that case, we do
        # not need to fuse any features
        if len(embeddings) > 1:
            concatenation = Concatenate(axis=-1, name="Join_Features")(embeddings)
        else:
            concatenation = embeddings[0]
        feature_fusion = TimeDistributed(
            Dense(
                units=100,
                activation='relu'),
            name='Feature_Fusion'
        )(concatenation)
        # ----------------------------------------------------------------------

        # LSTM Modeling Layer (many-to-one)
        rec_regularizer = l1(0.02) if self.use_regularizer else None
        lstm_cell = LSTM(units=100,
                         recurrent_regularizer=rec_regularizer
                         )(feature_fusion)

        # Output
        sigmoid = Dense(1, activation='sigmoid')(lstm_cell)

        return Model(inputs=inputs, outputs=sigmoid, name=f"{self.model_name}_DIS")

    def train(self, trainX: np.ndarray, epochs: int = 250,
              batch_size: int = 256, save_freq: int = 50,
              batch_equals_epoch: bool = False,
              print_training: bool = True
              ):
        """

        :param print_training:
        :param trainX:
        :param epochs:
        :param batch_size:
        :param save_freq:
        :param batch_equals_epoch: If True, one batch = one epoch.
        :return:
        """
        from keras.utils import pad_sequences
        # Padding: Generates a copy, original data unchanged!
        trainX = [pad_sequences(x, self.max_length, padding='pre', dtype='float64') for x in trainX]

        num_batches = 1 if batch_equals_epoch else math.ceil(len(trainX[0]) / batch_size)

        iterator = range(1, epochs + 1) if print_training else tqdm(range(1, epochs + 1), desc="Training")
        for epoch in iterator:

            random_indices = np.random.permutation(len(trainX[0])) if not batch_equals_epoch else None

            for batch in range(num_batches):

                if batch_equals_epoch:
                    # Select Random Batch indices
                    idx = np.random.randint(0, len(trainX[0]), batch_size)
                else:
                    idx = random_indices[batch_size * batch:batch_size * (batch + 1)]

                # Discriminator Ground Truth
                real_labels = np.ones((idx.shape[0], 1))
                syn_labels = np.zeros((idx.shape[0], 1))

                # Build set of real trajectories
                real_trajs = [
                    trainX[i][idx] for i in range(len(trainX))
                ]
                noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))
                real_trajs.append(noise)

                # Generate a batch of synthetic trajectories
                log.debug(f"Generate {idx.shape[0]} synthetic trajectories...")
                gen_trajs_bc = self.generator.predict(real_trajs, verbose=0)
                log.debug("...generation done.")

                # Train the discriminator
                # No noise is used, hence slicing with length of features
                log.debug(f"Train discriminator on real samples...")
                d_loss_real = self.discriminator.train_on_batch(real_trajs[:len(self.features)], real_labels)
                log.debug(f"Train discriminator on generated samples...")
                d_loss_syn = self.discriminator.train_on_batch(gen_trajs_bc[:len(self.features)], syn_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

                # Train the generator
                # Mask and noise are used
                noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))
                real_trajs[-1] = noise
                log.debug(f"Train generator...")
                inputs = real_trajs
                inputs.append(real_labels)  # Append labels so that y_true has access
                g_loss = self.combined.train_on_batch(inputs)
                log.debug(f"...Training Completed.")

                if print_training:
                    if not batch_equals_epoch:
                        msg = f"[Epoch {epoch:03d}/{epochs}] [Batch {batch + 1:03d}/{num_batches}]" \
                              f"D Loss: {d_loss[0]} | G Loss: {g_loss}"
                    else:
                        msg = f"[Epoch {epoch}/{epochs}] D Loss: {d_loss[0]} | G Loss: {g_loss}"
                    log.info(msg)

            # Print and save the losses/params
            if save_freq != 0 and epoch % save_freq == 0:
                self.save_checkpoint(epoch)
                log.info('Model params saved to the disk.')

    def save_checkpoint(self, epoch):
        self.combined.save_weights(f"{self.combined_param_path}{epoch}.hdf5")
        self.discriminator.save_weights(f"{self.discriminator_param_path}{epoch}.hdf5")
        self.generator.save_weights(f"{self.generator_param_path}{epoch}.hdf5")
        log.info(f"Training Params saved to {self.param_path}.")

    def load_checkpoint(self, epoch):
        self.combined.load_weights(f"{self.combined_param_path}{epoch}.hdf5")
        self.discriminator.load_weights(f"{self.discriminator_param_path}{epoch}.hdf5")
        self.generator.load_weights(f"{self.generator_param_path}{epoch}.hdf5")
        log.info(f"Training Params loaded from {self.param_path}.")

    def predict(self, testX: np.ndarray):
        from keras.utils import pad_sequences
        # Padding: Generates a copy, original data unchanged!
        testX = [pad_sequences(x, self.max_length, padding='pre', dtype='float64') for x in testX]
        noise = np.random.normal(0, 1, (len(testX[0]), self.latent_dim))
        testX.append(noise)
        return self.generator.predict(testX)

    def predict_and_convert(self, trajectories: List[pd.DataFrame] or pd.DataFrame) -> pd.DataFrame:

        testX = to_trajGAN_input(trajectories, (self.lat0, self.lon0), self.vocab_size['category'])

        prediction = self.predict(testX=testX)

        # Label names
        tid_label: str = ts.tid_label
        uid_label: str = ts.uid_label
        lat_label: str = ts.lat_label
        lon_label: str = ts.lon_label

        result = {
            tid_label: [],
            uid_label: [],
            lat_label: [],
            lon_label: []
        }
        for i, feature in enumerate(self.features):
            if feature != 'latlon':
                result[feature] = []
            for j, trajectory in enumerate(prediction[i]):
                original_length = len(testX[i][j])
                # Remove the padding (pre-padding was used)
                unpadded = trajectory[self.max_length - original_length:]
                if feature == 'latlon':
                    result[lat_label].extend(unpadded[:, 0])
                    result[lon_label].extend(unpadded[:, 1])
                else:
                    # We assume all other features are softmax outputs
                    result[feature].extend(np.argmax(unpadded, axis=1))
        # Add trajectory and user IDs
        # We know that the order has been preserved
        result[tid_label] = ts.get_tid_array()
        result[uid_label] = ts.get_uid_array()

        if not self.scale_in_model:
            result[lat_label] *= self.scale_factor[0]
            result[lon_label] *= self.scale_factor[1]

        # Add reference point to latitude and longitude
        result[lat_label] += self.lat0
        result[lon_label] += self.lon0

        # Create DataFrame
        df = pd.DataFrame(result)

        return df


if __name__ == '__main__':
    logger.configure_root_loger(logging.INFO, file=config.LOG_DIR + 'lstm_trajGAN_tf.log')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs of training")
    parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE, help="size of the batches")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--beta", type=float, default=BETA, help="Optimizer beta")
    parser.add_argument('--name', type=str, default='LSTM_TrajGAN_TF',
                        help="Model Name (determines parameter path)")
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM,
                        help="dimensionality of the noise space")
    parser.add_argument("--dp", action='store_true')
    parser.add_argument("--summary", action='store_true')
    parser.add_argument("-g", "--gpu", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--no_reg", action='store_false', help="Deactivate L1 regularizer for LSTM")
    parser.add_argument("-s", "--save_freq", type=int, default=50, help="Save frequency")
    parser.add_argument("-p", "--silent", action='store_true', help="Show progress bar only.")
    opt = parser.parse_args()
    log.info(f"CMD line arguments: {opt}")
    if opt.summary:
        # Deactivate GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = f"-1"
        gan = LSTM_TrajGAN((0, 0), (1, 1), 100)
        print("#" * 80)
        print("Generator:")
        print("#" * 80)
        print(gan.generator.summary())
        print("#" * 80)
        print(gan.discriminator.summary())
        print("#" * 80)
        print("Combined:")
        print(gan.combined.summary())
    else:
        # Only use one GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{opt.gpu}"
        log.info(f"Using GPU {opt.gpu}.")

        # Load training data
        path = config.BASE_DIR + 'data/fs_nyc/all_latlon.csv'
        ts: pd.DataFrame = pd.read_csv(path, dtype={'tid': str, 'label': 'int32'})

        # Determine Reference point
        ref = get_ref_point(ts[['lat', 'lon']])
        sf = get_scaling_factor(ts[['lat', 'lon']], ref)

        trainX = to_trajGAN_input(ts, ref, 10)
        log.info(f"Reference Point:\t{ref}")
        log.info(f"Scale Factor:\t{sf}")

        # Load Test set
        # tes = Trajectories.from_csv('data/test_latlon.csv', uid_label='label')

        gan = LSTM_TrajGAN(
            ref,
            sf,
            MAX_LEN,
            latent_dim=opt.latent_dim,
            model_name=opt.name,
            learning_rate=opt.lr,
            beta=opt.beta,
            use_regularizer=opt.no_reg,
        )
        gan.train(trainX,
                  epochs=opt.epochs,
                  save_freq=opt.save_freq,
                  batch_size=opt.batch_size
                  )
