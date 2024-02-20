import keras as ks
from keras import ops


class CorrectPartialCharges(ks.layers.Layer):
    """Layer to compute average charge corrections for partial charges from total charge."""

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CorrectPartialCharges, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        # Nothing to build here.
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Computes charge correction.

        Args:
            inputs (list): [predict_tc, tot_charge, count_nodes, batch_id_node]

                - predict_tc (Tensor): Predicted total charge of shape `(batch, 1)`.
                - tot_charge (Tensor): True total charge of shape `(batch, 1)`.
                - count_nodes (Tensor): Number of nodes per sample of shape `(batch, )`.
                - batch_id_node (Tensor): Batch ID of nodes of shape `([N], )`.

        Returns:
            list: Corrections of partial charges of shape `([N], 1)`.
        """
        predict_tc, tot_charge, count_nodes, batch_id_node = inputs
        charge_diff = tot_charge-predict_tc
        avg_charge_diff = charge_diff/ops.expand_dims(count_nodes, axis=-1)
        avg_charge_diff_per_node = ops.take(avg_charge_diff, batch_id_node, axis=0)
        return avg_charge_diff_per_node
