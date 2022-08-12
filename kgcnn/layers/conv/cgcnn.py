from networkx.algorithms.assortativity.pairs import node_attribute_xy
from kgcnn.layers.conv.message import MessagePassingBase
from kgcnn.layers.norm import GraphBatchNormalization

import tensorflow.keras as ks
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='kgcnn', name='CGCNNLayer')
class CGCNNLayer(MessagePassingBase):
    r"""Message Passing Layer used in the Crystal Graph Convolutional Neural Network (CGCNN).

    For more information look at the paper (<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>)
    or the original code base in pytorch (<https://github.com/txie-93/cgcnn>).

    Args:
        units (int): Units for Dense layer.
        activation_s (str): Tensorflow activation applied before gating the message.
        activation_out (str): Tensorflow activation applied the very end of the layer (after gating).
        batch_normalization (bool): Whether to use batch normalization (GraphBatchNormalization) or not.
    """

    def __init__(self, units=64, activation_s='softplus', activation_out='softplus', batch_normalization=False, **kwargs):
        super(CGCNNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_s = activation_s
        self.activation_out = activation_out
        self.batch_normalization = batch_normalization
        self.activation_f_layer = ks.activations.sigmoid
        self.activation_s_layer = ks.layers.Activation(activation_s)
        self.activation_out_layer = ks.layers.Activation(activation_out)
        self.batch_norm_f = GraphBatchNormalization()
        self.batch_norm_s = GraphBatchNormalization()
        self.batch_norm_out = GraphBatchNormalization()
        self.f = ks.layers.Dense(self.units)
        self.s = ks.layers.Dense(self.units)

    def message_function(self, inputs, **kwargs):

        nodes_in = inputs[0] # shape: (batch_size, M, F)
        nodes_out = inputs[1] # shape: (batch_size, M, F)
        edge_features = inputs[2] # shape: (batch_size, M, E)

        x = tf.concat([nodes_in, nodes_out, edge_features], axis=2)
        x_s, x_f = self.s(x), self.f(x)
        if self.batch_normalization:
            x_s, x_f = self.batch_norm_s(x_s), self.batch_norm_f(x_f)
        x_s, x_f = self.activation_s_layer(x_s), self.activation_f_layer(x_f)
        x_out = x_s * x_f # shape: (batch_size, M, self.units)
        return x_out

    def update_nodes(self, inputs, **kwargs):
        nodes, nodes_update = inputs
        nodes_updated = nodes + nodes_update
        if self.batch_normalization:
            nodes_updated = self.batch_norm_out(nodes_updated)
        nodes_updated = self.activation_out_layer(nodes_updated)
        return nodes_updated

    def get_config(self):
        """Update layer config."""
        config = super(CGCNNLayer, self).get_config()
        config.update({"units": self.units, "activation_s": self.activation_s, "activation_out": self.activation_out,
            "batch_normalization": self.batch_normalization})
        return config
