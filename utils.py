import math

import argparse
import tensorflow as tf
from tqdm import tqdm

def kl_weight(step):
    return (math.tanh((step - 3500)/1000) + 1)/2

def temperature(step):
    return 1 - kl_weight(step) + 1e-6

def tf_kl_weight(step):
    return (tf.math.tanh((step - 3500)/1000) + 1)/2

def tf_temperature(step):
    return tf.cast(1 - tf_kl_weight(step) + 1e-6, tf.float32)

# TODO Ensure that summing it over the batch has the expected effect
class KL_Loss(tf.keras.losses.Loss):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO):

        """

        See https://arxiv.org/pdf/1606.05908.pdf equation 7 for
        reference.

        Parameters
        ----------
        reduction
        """

        super().__init__(reduction=reduction)

    def call(self, mean, logvar):
        return 0.5 * tf.reduce_sum(tf.exp(logvar) + mean ** 2 - 1 - logvar, axis=-1)

class CrossEntropyWithLogitsLoss(tf.keras.losses.Loss):

    def __init__(self, probs_sum_to_one=True, reduction=tf.keras.losses.Reduction.AUTO):

        """
        Calculate cross entropy loss for each individual sample and reduce them.
        Whilst the dimensionality of `y_true` is expected to be the same as that of
        `y_pred`, the logits fed into cross entropy can either be normalised via
        softmax to represent a single probability distribution (`probs_sum_to_one`
        == True) of one class being the true class or via sigmoid to represent
        separate probabilities for each class (`probs_sum_to_one` == False).

        How losses of individual samples are combined is determined by `reduction`.
        See https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction for
        details.

        Parameters
        ----------
        probs_sum_to_one
        reduction
        """

        super().__init__(reduction=reduction)
        if probs_sum_to_one:
            self.function = tf.nn.softmax_cross_entropy_with_logits
        else:
            self.function = tf.nn.sigmoid_cross_entropy_with_logits

    def call(self, y_true, y_pred):
        return self.function(y_true, y_pred)

def get_sequential_mask(x, padding_token):
    return tf.cast(x != padding_token, tf.bool)