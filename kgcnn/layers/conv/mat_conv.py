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

    def call(self, inputs, mask, **kwargs):
        # Shape of inputs (batch, N, 3)
        # Shape of mask (batch, N, 3)
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(inputs, axis=2)
        dist = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)
        # shape of dist (batch, N, N, 1)
        diff_mask = tf.expand_dims(mask, axis=1) * tf.expand_dims(mask, axis=2)
        dist_mask = tf.reduce_prod(diff_mask, axis=-1, keepdims=True)

        if self.trafo == "exp":
            dist += tf.where(
                tf.cast(dist_mask, dtype="bool"), tf.zeros_like(dist), tf.ones_like(dist) / ks.backend.epsilon())
            dist = tf.exp(-dist)
        elif self.trafo == "softmax":
            dist += tf.where(
                tf.cast(dist_mask, dtype="bool"), tf.zeros_like(dist), -tf.ones_like(dist) / ks.backend.epsilon())
            dist = tf.nn.softmax(dist, axis=2)

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


class MATExpandMask(ks.layers.Layer):

    def __init__(self, axis: int, **kwargs):
        super(MATExpandMask, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super(MATExpandMask, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(MATExpandMask, self).get_config()
        config.update({"axis": self.axis})
        return config


class MATAttentionHead(ks.layers.Layer):

    def __init__(self, units: int = 64,
                 lambda_distance: float = 0.3, lambda_attention: float = 0.3,
                 lambda_adjacency: Union[float, None] = None,
                 dropout: Union[float, None] = None,
                 **kwargs):
        super(MATAttentionHead, self).__init__(**kwargs)
        self.units = int(units)
        self.lambda_distance = lambda_distance
        self.lambda_attention = lambda_attention
        if lambda_adjacency is not None:
            self.lambda_adjacency = lambda_adjacency
        else:
            self.lambda_adjacency = 1.0 - self.lambda_attention - self.lambda_distance
        self.scale = self.units ** -0.5
        self.dense_q = ks.layers.Dense(units=units)
        self.dense_k = ks.layers.Dense(units=units)
        self.dense_v = ks.layers.Dense(units=units)
        self._dropout = dropout
        if self._dropout is not None:
            self.layer_dropout = ks.layers.Dropout(self._dropout)

    def build(self, input_shape):
        super(MATAttentionHead, self).build(input_shape)

    def call(self, inputs, mask, **kwargs):
        h, a_d, a_g = inputs
        h_mask, a_d_mask, a_g_mask = mask
        q = tf.expand_dims(self.dense_q(h), axis=2)
        k = tf.expand_dims(self.dense_k(h), axis=1)
        v = self.dense_v(h) * h_mask
        qk = q * k / self.scale
        # Apply mask on self-attention
        qk_mask = tf.expand_dims(h_mask, axis=1) * tf.expand_dims(h_mask, axis=2)  # (b, 1, n, ...) * (b, n, 1, ...)
        qk += tf.where(tf.cast(qk_mask, dtype="bool"), tf.zeros_like(qk), -tf.ones_like(qk) / ks.backend.epsilon())
        qk = tf.nn.softmax(qk, axis=2)
        qk *= qk_mask
        # Weights
        qk = self.lambda_attention * qk
        a_d = self.lambda_distance * tf.cast(a_d, dtype=h.dtype)
        a_g = self.lambda_adjacency * tf.cast(a_g, dtype=h.dtype)
        # print(qk.shape, a_d.shape, a_g.shape)
        att = qk + a_d + a_g
        # v has shape (b, N, F)
        # att has shape (b, N, N, F)
        if self._dropout is not None:
            att = self.layer_dropout(att)

        # Or permute feature dimension to batch and apply on last axis via and permute back again
        v = tf.transpose(v, perm=[0, 2, 1])
        att = tf.transpose(att, perm=[0, 3, 1, 2])
        hp = tf.einsum('...ij,...jk->...ik', att, tf.expand_dims(v, axis=3))  # From example in tf docs
        hp = tf.squeeze(hp, axis=3)
        hp = tf.transpose(hp, perm=[0, 2, 1])

        # Same as above but may be slower.
        # hp = tf.einsum('bij...,bjk...->bik...', att, tf.expand_dims(v, axis=2))
        # hp = tf.squeeze(hp, axis=2)

        hp *= h_mask
        return hp

    def get_config(self):
        config = super(MATAttentionHead, self).get_config()
        config.update({"units": self.units, "lambda_adjacency": self.lambda_adjacency,
                       "lambda_attention": self.lambda_attention, "lambda_distance": self.lambda_distance,
                       "dropout": self._dropout})
        return config
