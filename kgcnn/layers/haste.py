import tensorflow as tf
try:
    import haste_tf as haste
except ModuleNotFoundError:
    print("WARNING: Could not load haste implementation of GRU. Please check https://github.com/lmnt-com/haste.")

from kgcnn.layers.base import GraphBaseLayer


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='HasteGRUupdate')
class HasteGRUupdate(GraphBaseLayer):
    """Gated recurrent unit update with hast GRU.

    Args:
        units (int): Units for GRU cell.
        trainable (bool): If GRU is trainable. Defaults to True.
    """

    def __init__(self, units, trainable=True, **kwargs):
        """Initialize layer."""
        super(HasteGRUupdate, self).__init__(trainable=trainable, **kwargs)
        self.units = units

        self.gru_cell = haste.GRUCell(units=units, trainable=trainable)

    def build(self, input_shape):
        """Build layer."""
        super(HasteGRUupdate, self).build(input_shape)
        assert len(input_shape) == 2

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, updates]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - updates (tf.RaggedTensor): Matching node updates of shape (batch, [N], F)

        Returns:
           tf.RaggedTensor: Updated nodes of shape (batch, [N], F)
        """
        dyn_inputs = inputs
        # We cast to values here
        n, npart = dyn_inputs[0].values, dyn_inputs[0].row_splits
        eu, _ = dyn_inputs[1].values, dyn_inputs[1].row_splits

        out, _ = self.gru_cell(eu, n, **kwargs)

        out = tf.RaggedTensor.from_row_splits(out, npart, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(HasteGRUupdate, self).get_config()
        config.update({"units": self.units})
        return config


