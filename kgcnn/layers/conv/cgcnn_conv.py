import tensorflow as tf
from kgcnn.layers.conv.message import MessagePassingBase
from kgcnn.layers.norm import GraphBatchNormalization
from kgcnn.layers.modules import ActivationEmbedding, LazyMultiply, LazyConcatenate, LazyAdd
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='CGCNNLayer')
class CGCNNLayer(MessagePassingBase):
    r"""Message Passing Layer used in the Crystal Graph Convolutional Neural Network:
    `CGCNN <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_.

    Based on the original code in pytorch (<https://github.com/txie-93/cgcnn>).

    Args:
        units (int): Units for Dense layer.
        activation_s (str): Tensorflow activation applied before gating the message.
        activation_out (str): Tensorflow activation applied the very end of the layer (after gating).
        batch_normalization (bool): Whether to use batch normalization (GraphBatchNormalization) or not.
    """

    def __init__(self, units=64,
                 activation_s='relu',
                 activation_out='relu',
                 batch_normalization=False,
                 **kwargs):
        super(CGCNNLayer, self).__init__(**kwargs)
        self.units = units
        self.batch_normalization = batch_normalization
        self.activation_f_layer = ActivationEmbedding(activation="sigmoid")
        self.activation_s_layer = ActivationEmbedding(activation_s)
        self.activation_out_layer = ActivationEmbedding(activation_out)
        self.batch_norm_f = GraphBatchNormalization()
        self.batch_norm_s = GraphBatchNormalization()
        self.batch_norm_out = GraphBatchNormalization()
        self.f = ks.layers.Dense(self.units)
        self.s = ks.layers.Dense(self.units)
        self.lazy_mult = LazyMultiply()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate()

    def message_function(self, inputs, **kwargs):
        r"""Prepare messages.

        Args:
            inputs: [nodes_in, nodes_out, edges]

                - nodes_in (tf.RaggedTensor): Embedding of sending nodes of shape `(batch, [M], F)`
                - nodes_out (tf.RaggedTensor): Embedding of sending nodes of shape `(batch, [M], F)`
                - edges (tf.RaggedTensor): Embedding of edges of shape `(batch, [M], E)`

        Returns:
            tf.RaggedTensor: Messages for updates of shape `(batch, [M], units)`.
        """

        nodes_in = inputs[0]  # shape: (batch_size, M, F)
        nodes_out = inputs[1]  # shape: (batch_size, M, F)
        edge_features = inputs[2]  # shape: (batch_size, M, E)

        x = self.lazy_concat([nodes_in, nodes_out, edge_features], axis=2, **kwargs)
        x_s, x_f = self.s(x, **kwargs), self.f(x, **kwargs)
        if self.batch_normalization:
            x_s, x_f = self.batch_norm_s(x_s, **kwargs), self.batch_norm_f(x_f, **kwargs)
        x_s, x_f = self.activation_s_layer(x_s, **kwargs), self.activation_f_layer(x_f, **kwargs)
        x_out = self.lazy_mult([x_s, x_f], **kwargs)  # shape: (batch_size, M, self.units)
        if self.batch_normalization:
            x_out = self.batch_norm_out(x_out, **kwargs)
        x_out = self.activation_out_layer(x_out, **kwargs)
        return x_out

    def update_nodes(self, inputs, **kwargs):
        """Update node embeddings.

        Args:
            inputs: [nodes, nodes_updates]

                - nodes (tf.RaggedTensor): Embedding of nodes of previous layer of shape `(batch, [M], F)`
                - nodes_updates (tf.RaggedTensor): Node updates of shape `(batch, [M], F)`

        Returns:
            tf.RaggedTensor: Updated nodes of shape `(batch, [N], F)`.
        """
        nodes = inputs[0]
        nodes_update = inputs[1]
        return self.lazy_add([nodes, nodes_update], **kwargs)

    def get_config(self):
        """Update layer config."""
        config = super(CGCNNLayer, self).get_config()
        config.update({
            "units": self.units,
            "batch_normalization": self.batch_normalization})
        conf_s = self.activation_s_layer.get_config()
        conf_out = self.activation_out_layer.get_config()
        config.update({"activation_s": conf_s["activation"]})
        config.update({"activation_out": conf_out["activation"]})
        return config
