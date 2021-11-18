import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ExpandDims')
class ExpandDims(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis  # We do not change the axis here

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        if isinstance(inputs, tf.RaggedTensor):
            axis = get_positive_axis(self.axis, inputs.shape.rank + 1)
            if axis > 1 and inputs.ragged_rank == 1:
                value_tensor = inputs.values  # will be Tensor
                out_tensor = tf.expand_dims(value_tensor, axis=axis - 1)
                return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting keras call... ")
        # Try normal operation
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(GraphBaseLayer, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ZerosLike')
class ZerosLike(GraphBaseLayer):
    """Make a zero-like graph tensor. Calls tf.zeros_like()."""

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(ZerosLike, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(ZerosLike, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Tensor of node or edge embeddings of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Zero-like tensor of input.
        """
        if isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                zero_tensor = tf.zeros_like(inputs.values)  # will be Tensor
                return tf.RaggedTensor.from_row_splits(zero_tensor, inputs.row_splits, validate=self.ragged_validate)
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1.")
        # Try normal tf call
        return tf.zeros_like(inputs)