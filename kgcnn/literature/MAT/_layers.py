import keras as ks
from keras import ops
from typing import Union


class MATGlobalPool(ks.layers.Layer):

    def __init__(self, pooling_method: str = "sum", **kwargs):
        super(MATGlobalPool, self).__init__(**kwargs)
        self.pooling_method = pooling_method

        # Add mean with mask if required.
        if self.pooling_method not in ["sum"]:
            raise ValueError("`pooling_method` must be in ['sum']")

    def build(self, input_shape):
        super(MATGlobalPool, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs (Tensor): Node or edge features of shape `(batch, N, ...)` .
            mask (Tensor): Not used.

        Returns:
            Tensor: Pooled features, e.g. summed over first axis.
        """
        if self.pooling_method == "sum":
            return ops.sum(inputs, axis=1)

    def get_config(self):
        config = super(MATGlobalPool, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


class MATDistanceMatrix(ks.layers.Layer):

    def __init__(self, trafo: Union[str, None] = "exp", **kwargs):
        super(MATDistanceMatrix, self).__init__(**kwargs)
        self.trafo = trafo
        # self._softmax = ks.layers.Softmax(axis=2)
        if self.trafo not in [None, "exp", "softmax"]:
            raise ValueError("`trafo` must be in [None, 'exp', 'softmax']")

    def build(self, input_shape):
        super(MATDistanceMatrix, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass

        Args:
            inputs (Tensor): Padded Coordinates of shape `(batch, N, 3)` .
            mask (Tensor): Mask of coordinates of similar shape.

        Returns:
            tuple: Distance matrix of shape `(batch, N, N, 1)` plus mask.
        """
        # Shape of inputs (batch, N, 3)
        # Shape of mask (batch, N, 3)
        diff = ops.expand_dims(inputs, axis=1) - ops.expand_dims(inputs, axis=2)
        dist = ops.sum(ops.square(diff), axis=-1, keepdims=True)
        # shape of dist (batch, N, N, 1)
        mask = ops.cast(mask, dtype="float32")
        diff_mask = ops.expand_dims(mask, axis=1) * ops.expand_dims(mask, axis=2)
        dist_mask = ops.prod(diff_mask, axis=-1, keepdims=True)

        if self.trafo == "exp":
            dist += ops.where(
                ops.cast(dist_mask, dtype="bool"), ops.zeros_like(dist), ops.ones_like(dist) / ks.backend.epsilon())
            dist = ops.exp(-dist)
        elif self.trafo == "softmax":
            dist += ops.where(
                ops.cast(dist_mask, dtype="bool"), ops.zeros_like(dist), -ops.ones_like(dist) / ks.backend.epsilon())
            dist = ops.nn.softmax(dist, axis=2)

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
        r"""Forward Pass.

        Args:
            inputs (Tensor): Any (mask) Tensor of sufficient rank to reduce for given axis.

        Returns:
            Tensor: Product of inputs along axis.
        """
        out = ops.prod(inputs, keepdims=self.keepdims, axis=self.axis)
        return out

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
        r"""Forward Pass.

        Args:
            inputs (Tensor): Any (mask) Tensor to expand given axis.

        Returns:
            Tensor: Input tensor with expanded axis.
        """
        out = ops.expand_dims(inputs, axis=self.axis)
        return out

    def get_config(self):
        config = super(MATExpandMask, self).get_config()
        config.update({"axis": self.axis})
        return config


class MATAttentionHead(ks.layers.Layer):

    def __init__(self, units: int = 64,
                 lambda_distance: float = 0.3, lambda_attention: float = 0.3,
                 lambda_adjacency: Union[float, None] = None, add_identity: bool = False,
                 dropout: Union[float, None] = None,
                 **kwargs):
        super(MATAttentionHead, self).__init__(**kwargs)
        self.units = int(units)
        self.add_identity = bool(add_identity)
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

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): List of [h_n, A_d, A_g] represented by padded :obj:`Tensor` .
                These are node features and adjacency matrix from distances and bonds or bond order.
            mask (list): Mask tensors matching inputs, i.e. a mask tensor for each padded input.

        Returns:
            Tensor: Padded node features of :math:`h_n` .
        """
        h, a_d, a_g = inputs
        h_mask, a_d_mask, a_g_mask = mask
        h_mask = ops.cast(h_mask, dtype=h.dtype)
        q = ops.expand_dims(self.dense_q(h), axis=2)
        k = ops.expand_dims(self.dense_k(h), axis=1)
        v = self.dense_v(h) * h_mask
        qk = q * k / self.scale
        # Apply mask on self-attention
        qk_mask = ops.expand_dims(h_mask, axis=1) * ops.expand_dims(h_mask, axis=2)  # (b, 1, n, ...) * (b, n, 1, ...)
        qk += ops.where(ops.cast(qk_mask, dtype="bool"), ops.zeros_like(qk), -ops.ones_like(qk) / ks.backend.epsilon())
        qk = ops.nn.softmax(qk, axis=2)
        qk *= qk_mask
        # Add diagonal to graph adjacency (optional).
        if self.add_identity:
            a_g_eye = ops.eye(ops.shape(a_g)[1], dtype=a_g.dtype)
            a_g_eye = ops.repeat(ops.expand_dims(a_g_eye, axis=0), ops.shape(a_g)[:1], axis=0)
            if a_g.shape.rank > 3:
                a_g_eye = ops.expand_dims(a_g_eye, axis=-1)
            a_g += a_g_eye
        # Weights
        qk = self.lambda_attention * qk
        a_d = self.lambda_distance * ops.cast(a_d, dtype=h.dtype)
        a_g = self.lambda_adjacency * ops.cast(a_g, dtype=h.dtype)
        # print(qk.shape, a_d.shape, a_g.shape)
        att = qk + a_d + a_g
        # v has shape (b, N, F)
        # att has shape (b, N, N, F)
        if self._dropout is not None:
            att = self.layer_dropout(att)

        # Or permute feature dimension to batch and apply on last axis via and permute back again
        v = ops.transpose(v, axes=[0, 2, 1])
        att = ops.transpose(att, axes=[0, 3, 1, 2])
        hp = ops.einsum('...ij,...jk->...ik', att, ops.expand_dims(v, axis=3))  # From example in tf docs
        hp = ops.squeeze(hp, axis=3)
        hp = ops.transpose(hp, axes=[0, 2, 1])

        # Same as above but may be slower.
        # hp = tf.einsum('bij...,bjk...->bik...', att, tf.expand_dims(v, axis=2))
        # hp = tf.squeeze(hp, axis=2)

        hp *= h_mask
        return hp

    def get_config(self):
        config = super(MATAttentionHead, self).get_config()
        config.update({"units": self.units, "lambda_adjacency": self.lambda_adjacency,
                       "lambda_attention": self.lambda_attention, "lambda_distance": self.lambda_distance,
                       "dropout": self._dropout, "add_identity": self.add_identity})
        return config
