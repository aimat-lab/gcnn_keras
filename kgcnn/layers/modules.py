import keras as ks
from keras import ops


class Embedding(ks.layers.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embeddings_initializer = ks.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = ks.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = ks.regularizers.get(activity_regularizer)
        self.embeddings_constraint = ks.constraints.get(embeddings_constraint)

        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(input_dim, output_dim),
            dtype=self.dtype,
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            trainable=True,
        )

    def build(self, input_shape):
        super(Embedding, self).build(input_shape)

    def call(self, inputs):
        return ops.take(self.embeddings, inputs, axis=0)

    def get_config(self):
        return super(Embedding, self).get_config()


class ExpandDims(ks.layers.Layer):

    def __init__(self, axis, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return ops.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDims, self).get_config()
        config.update({"axis": self.axis})
        return config


class SqueezeDims(ks.layers.Layer):

    def __init__(self, axis, **kwargs):
        super(SqueezeDims, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return ops.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super(SqueezeDims, self).get_config()
        config.update({"axis": self.axis})
        return config


def Input(
        shape=None,
        batch_size=None,
        dtype=None,
        sparse=None,
        batch_shape=None,
        name=None,
        tensor=None,
        ragged=None
    ):

    layer = ks.layers.InputLayer(
        shape=shape,
        batch_size=batch_size,
        dtype=dtype,
        sparse=sparse,
        batch_shape=batch_shape,
        name=name,
        input_tensor=tensor,
    )
    return layer.output


class ZerosLike(ks.layers.Layer):
    r"""Layer to make a zero tensor"""

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(ZerosLike, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(ZerosLike, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (Tensor): Tensor of node or edge embeddings of shape ([N], F, ...)

        Returns:
            Tensor: Zero-like tensor of input.
        """
        return ops.zeros_like(inputs)