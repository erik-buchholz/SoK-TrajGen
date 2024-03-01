#!/usr/bin/env python3
"""Implements the loss functions."""
import tensorflow as tf


def compute_mask_trajLoss(real_traj):
    mask_value = tf.constant([0.0, 0.0], dtype=tf.float32)
    mask = tf.reduce_all(tf.math.not_equal(real_traj[0], mask_value, name="Mask Value"), axis=-1)
    mask = tf.cast(mask, dtype=tf.float32, name="Mask")
    traj_length = tf.math.reduce_sum(mask, name="trajLength", axis=-1, keepdims=True)
    return mask, traj_length


def compute_bce(y_true, y_pred, use_regularizer):
    if use_regularizer:
        bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    else:
        # For DP training to work, we need to disable regularization
        # See: https://github.com/tensorflow/privacy/issues/180
        bce_loss = tf.keras.losses.BinaryCrossentropy(reduction='none')(y_true, y_pred)
    return bce_loss


def compute_latlon_loss(real_traj, gen_traj, mask, traj_length):
    masked_latlon_full = tf.math.reduce_sum(
        tf.math.reduce_sum(
            tf.math.multiply(
                tf.math.multiply(
                    (gen_traj[0] - real_traj[0]),
                    (gen_traj[0] - real_traj[0])
                ),
                tf.repeat(tf.reshape(mask, shape=(-1, mask.shape[-1], 1)), 2, axis=-1)  # Ignore the masked values
            ), axis=1),
        axis=1,
        keepdims=True, name="masked_latlon"
    )
    masked_latlon_mse = tf.math.reduce_sum(tf.math.divide(masked_latlon_full, traj_length), name="latlon_mse")
    return masked_latlon_mse


def compute_hour_loss(real_traj, gen_traj, mask, traj_length):
    ce_hour = tf.nn.softmax_cross_entropy_with_logits(gen_traj[1], real_traj[1], name="ce_hour")

    ce_hour_masked = tf.math.multiply(ce_hour, mask)

    ce_hour_mean = tf.math.reduce_sum(tf.math.divide(ce_hour_masked, traj_length))
    return ce_hour_mean


def compute_dow_loss(real_traj, gen_traj, mask, traj_length):
    ce_dow = tf.nn.softmax_cross_entropy_with_logits(gen_traj[2], real_traj[2], name="ce_dow")

    ce_dow_masked = tf.math.multiply(ce_dow, mask)

    ce_dow_mean = tf.math.reduce_sum(tf.math.divide(ce_dow_masked, traj_length))
    return ce_dow_mean


def compute_cat_loss(real_traj, gen_traj, mask, traj_length):
    ce_category = tf.nn.softmax_cross_entropy_with_logits(gen_traj[3], real_traj[3], name="ce_cat")

    ce_category_masked = tf.math.multiply(ce_category, mask)

    ce_category_mean = tf.math.reduce_sum(tf.math.divide(ce_category_masked, traj_length))
    return ce_category_mean


# trajLoss for the generator
def trajLoss(y_true, y_pred, real_traj, gen_traj, use_regularizer: bool = True):
    # Find out the length of the real_traj by counting the number
    # points if (lat, lon) == (0, 0)
    mask, traj_length = compute_mask_trajLoss(real_traj)

    bce_loss = compute_bce(y_true, y_pred, use_regularizer)

    masked_latlon_mse = compute_latlon_loss(real_traj, gen_traj, mask, traj_length)

    ce_category_mean = compute_cat_loss(real_traj, gen_traj, mask, traj_length)
    ce_dow_mean = compute_dow_loss(real_traj, gen_traj, mask, traj_length)
    ce_hour_mean = compute_hour_loss(real_traj, gen_traj, mask, traj_length)

    p_bce = 1
    p_latlon = 10
    p_cat = 1
    p_dow = 1
    p_hour = 1

    bce_loss = tf.cast(bce_loss, dtype=masked_latlon_mse.dtype)

    return bce_loss * p_bce + masked_latlon_mse * p_latlon + \
        ce_category_mean * p_cat + ce_dow_mean * p_dow + ce_hour_mean * p_hour
