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
        self.dense_q = ks.layers.Dense(units=units)
        self.dense_k = ks.layers.Dense(units=units)
        self.dense_v = ks.layers.Dense(units=units)

    def build(self, input_shape):
        super(MATAttentionHead, self).build(input_shape)

    def call(self, inputs, mask, **kwargs):
        h, a_d, a_a = inputs
        h_mask, a_d_mask, a_a_mask = mask
        q = tf.expand_dims(self.dense_q(h), axis=2)
        k = tf.expand_dims(self.dense_k(h), axis=1)
        v = self.dense_v(h)
        qk = tf.einsum('bij...,bjk...->bik...', q, k) / self.scale
        qk = tf.nn.softmax(qk, axis=2)
        # Apply mask on self-attention
        qk_mask = tf.expand_dims(h_mask, axis=1) * tf.expand_dims(h_mask, axis=2)  # (b, 1, n, ...) * (b, n, 1, ...)
        qk *= qk_mask
        att = self.lambda_g * qk + self.lambda_d * tf.cast(a_d, dtype=h.dtype) + self.lambda_a * tf.cast(a_a,
                                                                                                         dtype=h.dtype)
        hp = tf.einsum('bij...,bjk...->bik...', att, tf.expand_dims(h, axis=2))
        hp = tf.squeeze(hp, axis=2)
        hp *= h_mask
        return

    def get_config(self):
        config = super(MATAttentionHead, self).get_config()
        config.update({"units": self.units, "lambda_a": self.lambda_a,
                       "lambda_g": self.lambda_g, "lambda_d": self.lambda_d})
        return config
