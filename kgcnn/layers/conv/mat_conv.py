import tensorflow as tf
from typing import Union

ks = tf.keras


class MATGlobalPool(ks.layers.Layer):

    def __init__(self, pooling_method: str = "sum", **kwargs):
        super(MATGlobalPool, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        # TODO: Add mean with mask.

        if self.pooling_method not in ["sum"]:
            raise ValueError("`pooling_method` must be in ['sum']")

    def build(self, input_shape):
        super(MATGlobalPool, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.pooling_method == "sum":
            return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        config = super(MATGlobalPool, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


class MATDistanceMatrix(ks.layers.Layer):

    def __init__(self, trafo: Union[str, None] = "exp", **kwargs):
        super(MATDistanceMatrix, self).__init__(**kwargs)
        self.trafo = trafo
        if self.trafo not in [None, "exp", "softmax"]:
            raise ValueError("`trafo` must be in [None, 'exp', 'softmax']")

    def build(self, input_shape):
        super(MATDistanceMatrix, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # Shape of inputs (batch, N, 3)
        # Shape of mask (batch, N, 3)
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(inputs, axis=2)
        dist = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)
        if self.trafo == "exp":
            dist = tf.exp(-dist)
        if self.trafo == "softmax":
            dist = tf.nn.softmax(dist, axis=-1)
        # If no mask.
        if mask is None:
            return dist
        diff_mask = tf.expand_dims(inputs, axis=1) * tf.expand_dims(inputs, axis=2)
        dist_mask = tf.reduce_prod(diff_mask, axis=-1, keepdims=True)
        dist = dist * dist_mask
        return dist, dist_mask

    def get_config(self):
        config = super(MATDistanceMatrix, self).get_config()
        config.update({"trafo": self.trafo})
        return config


class MATReduceMask(ks.layers.Layer):

    def __init__(self, axis: int, keepdims: bool, **kwargs):
        super(MATReduceMask, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def build(self, input_shape):
        super(MATReduceMask, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.reduce_prod(inputs, keepdims=self.keepdims, axis=self.axis)

    def get_config(self):
        config = super(MATReduceMask, self).get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config


class MATAttentionHead(ks.layers.Layer):

    def __init__(self, units: int = 64, lambda_a: float = 1.0, lambda_g: float = 0.5, lambda_d: float = 0.5, **kwargs):
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
        h, a_d, a_g = inputs
        h_mask, a_d_mask, a_g_mask = mask
        q = tf.expand_dims(self.dense_q(h)*h_mask, axis=2)
        k = tf.expand_dims(self.dense_k(h)*h_mask, axis=1)
        v = self.dense_v(h)*h_mask
        qk = tf.einsum('bij...,bjk...->bik...', q, k) / self.scale
        qk = tf.nn.softmax(qk, axis=2)
        # Apply mask on self-attention
        qk_mask = tf.expand_dims(h_mask, axis=1) * tf.expand_dims(h_mask, axis=2)  # (b, 1, n, ...) * (b, n, 1, ...)
        qk *= qk_mask
        # Weights
        qk = self.lambda_a * qk
        a_d = self.lambda_d * tf.cast(a_d, dtype=h.dtype)
        a_g = self.lambda_g * tf.cast(a_g, dtype=h.dtype)
        att = qk + a_d + a_g
        hp = tf.einsum('bij...,bjk...->bik...', att, tf.expand_dims(v, axis=2))
        hp = tf.squeeze(hp, axis=2)
        hp *= h_mask
        return hp

    def get_config(self):
        config = super(MATAttentionHead, self).get_config()
        config.update({"units": self.units, "lambda_a": self.lambda_a,
                       "lambda_g": self.lambda_g, "lambda_d": self.lambda_d})
        return config
