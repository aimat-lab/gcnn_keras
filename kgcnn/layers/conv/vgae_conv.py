import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='VGAEZerosLike')
class VGAEZerosLike(GraphBaseLayer):
    """Make a zero-like graph tensor. Calls tf.zeros_like()."""

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(VGAEZerosLike, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(VGAEZerosLike, self).build(input_shape)

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
