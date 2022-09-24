import tensorflow as tf
from typing import Union

ks = tf.keras


class MATDistanceMatrix(ks.layers.Layer):

    def __init__(self, **kwargs):
        super(MATDistanceMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MATDistanceMatrix, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # Shape of inputs (batch, N, 3)
        # Shape of mask (batch, N, 3)
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(inputs, axis=2)
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        # Mask
        if mask is not None:
            diff_mask = tf.expand_dims(inputs, axis=1) * tf.expand_dims(inputs, axis=2)
            dist_mask = tf.reduce_prod(diff_mask, axis=-1)
            dist = dist * dist_mask
            return dist, dist_mask
        else:
            return dist


class MATAttentionHead(ks.layers.Layer):

    def __init__(self, units=64, lambda_a=1, lambda_g=0.5, lambda_d=0.5, **kwargs):
        super(MATAttentionHead, self).__init__(**kwargs)
        self.units = int(units)
        self.lambda_a = lambda_a
        self.lambda_g = lambda_g
        self.lambda_d = lambda_d
        self.scale = self.units ** -0.5

    def build(self, input_shape):
        super(MATAttentionHead, self).build(input_shape)

    def call(self, inputs, mask, **kwargs):
        h, a_d, a_a = inputs
        h_mask, a_d_mask, a_a_mask = mask
        q =

        return out

    def get_config(self):
        config = super(MATAttentionHead, self).get_config()
        config.update({"units": self.units, "lambda_a": self.lambda_a,
                       "lambda_g": self.lambda_g, "lambda_d": self.lambda_d})
        return config
