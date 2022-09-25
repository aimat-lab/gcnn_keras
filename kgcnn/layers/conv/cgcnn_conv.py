import tensorflow as tf
from kgcnn.layers.message import MessagePassingBase
from kgcnn.layers.norm import GraphBatchNormalization
from kgcnn.layers.modules import ActivationEmbedding, LazyMultiply, LazyConcatenate, LazyAdd, DenseEmbedding
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
        batch_normalization (bool): Whether to use batch normalization (:obj:`GraphBatchNormalization`) or not.
        use_bias (bool): Boolean, whether the layer uses a bias vector. Default is True.
        kernel_initializer: Initializer for the `kernel` weights matrix. Default is "glorot_uniform".
        bias_initializer: Initializer for the bias vector. Default is "zeros".
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix. Default is None.
        bias_regularizer: Regularizer function applied to the bias vector. Default is None.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation"). Default is None.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix. Default is None.
        bias_constraint: Constraint function applied to the bias vector. Default is None.
    """

    def __init__(self, units: int = 64,
                 activation_s="softplus",
                 activation_out="softplus",
                 batch_normalization: bool = False,
                 use_bias: bool = True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CGCNNLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.batch_normalization = batch_normalization
        kernel_args = {"kernel_regularizer": kernel_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.activation_f_layer = ActivationEmbedding(activation="sigmoid", activity_regularizer=activity_regularizer)
        self.activation_s_layer = ActivationEmbedding(activation_s, activity_regularizer=activity_regularizer)
        self.activation_out_layer = ActivationEmbedding(activation_out, activity_regularizer=activity_regularizer)
        if batch_normalization:
            self.batch_norm_f = GraphBatchNormalization()
            self.batch_norm_s = GraphBatchNormalization()
            self.batch_norm_out = GraphBatchNormalization()
        self.f = DenseEmbedding(self.units, activation="linear", use_bias=use_bias, **kernel_args)
        self.s = DenseEmbedding(self.units, activation="linear", use_bias=use_bias, **kernel_args)
        self.lazy_mult = LazyMultiply()
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate(axis=2)

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

        x = self.lazy_concat([nodes_in, nodes_out, edge_features], **kwargs)
        x_s, x_f = self.s(x, **kwargs), self.f(x, **kwargs)
        if self.batch_normalization:
            x_s, x_f = self.batch_norm_s(x_s, **kwargs), self.batch_norm_f(x_f, **kwargs)
        x_s, x_f = self.activation_s_layer(x_s, **kwargs), self.activation_f_layer(x_f, **kwargs)
        x_out = self.lazy_mult([x_s, x_f], **kwargs)  # shape: (batch_size, M, self.units)
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
        if self.batch_normalization:
            nodes_update = self.batch_norm_out(nodes_update, **kwargs)
        nodes_updated = self.lazy_add([nodes, nodes_update], **kwargs)
        nodes_updated = self.activation_out_layer(nodes_updated, **kwargs)
        return nodes_updated

    def get_config(self):
        """Update layer config."""
        config = super(CGCNNLayer, self).get_config()
        config.update({
            "units": self.units, "use_bias": self.use_bias,
            "batch_normalization": self.batch_normalization})
        conf_s = self.activation_s_layer.get_config()
        conf_out = self.activation_out_layer.get_config()
        config.update({"activation_s": conf_s["activation"]})
        config.update({"activation_out": conf_out["activation"],
                       "activity_regularizer": conf_out["activity_regularizer"]})
        conf_f = self.f.get_config()
        for x in ["kernel_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer"]:
            config.update({x: conf_f[x]})
        return config
