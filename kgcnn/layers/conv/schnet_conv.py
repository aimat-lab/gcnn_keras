import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import LazyMultiply, DenseEmbedding, LazyAdd
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SchNetCFconv')
class SchNetCFconv(GraphBaseLayer):
    r"""Continuous filter convolution of `SchNet <https://aip.scitation.org/doi/pdf/10.1063/1.5019779>`_ .

    Edges are processed by 2 :obj:`Dense` layers, multiplied on outgoing node features and pooled for receiving node.

    Args:
        units (int): Units for Dense layer.
        cfconv_pool (str): Pooling method. Default is 'segment_sum'.
        use_bias (bool): Use bias. Default is True.
        activation (str): Activation function. Default is 'kgcnn>shifted_softplus'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, units,
                 cfconv_pool='segment_sum',
                 use_bias=True,
                 activation='kgcnn>shifted_softplus',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.units = units
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = DenseEmbedding(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_dense2 = DenseEmbedding(units=self.units, activation='linear', use_bias=self.use_bias, **kernel_args)
        self.lay_sum = PoolingLocalEdges(pooling_method=cfconv_pool)
        self.gather_n = GatherNodesOutgoing()
        self.lay_mult = LazyMultiply()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass. Calculate edge update.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [N], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [N], 2)

        Returns:
            tf.RaggedTensor: Updated node features.
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(edge, **kwargs)
        x = self.lay_dense2(x, **kwargs)
        node2exp = self.gather_n([node, indexlist], **kwargs)
        x = self.lay_mult([node2exp, x], **kwargs)
        x = self.lay_sum([node, x, indexlist], **kwargs)
        return x

    def get_config(self):
        """Update layer config."""
        config = super(SchNetCFconv, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SchNetInteraction')
class SchNetInteraction(GraphBaseLayer):
    r"""`SchNet <https://aip.scitation.org/doi/pdf/10.1063/1.5019779>`_ interaction block,
    which uses the continuous filter convolution from :obj:`SchNetCFconv`.

    Args:
        units (int): Dimension of node embedding. Default is 128.
        cfconv_pool (str): Pooling method information for SchNetCFconv layer. Default is'segment_sum'.
        use_bias (bool): Use bias in last layers. Default is True.
        activation (str): Activation function. Default is 'kgcnn>shifted_softplus'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units=128,
                 cfconv_pool='sum',
                 use_bias=True,
                 activation='kgcnn>shifted_softplus',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.use_bias = use_bias
        self.units = units
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        conv_args = {"units": self.units, "use_bias": use_bias, "activation": activation, "cfconv_pool": cfconv_pool}

        # Layers
        self.lay_cfconv = SchNetCFconv(**conv_args, **kernel_args)
        self.lay_dense1 = DenseEmbedding(units=self.units, activation='linear', use_bias=False, **kernel_args)
        self.lay_dense2 = DenseEmbedding(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_dense3 = DenseEmbedding(units=self.units, activation='linear', use_bias=self.use_bias, **kernel_args)
        self.lay_add = LazyAdd()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass. Calculate node update.

        Args:
            inputs: [nodes, edges, tensor_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [N], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [N], 2)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F).
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(node, **kwargs)
        x = self.lay_cfconv([x, edge, indexlist], **kwargs)
        x = self.lay_dense2(x, **kwargs)
        x = self.lay_dense3(x, **kwargs)
        out = self.lay_add([node, x], **kwargs)
        return out

    def get_config(self):
        config = super(SchNetInteraction, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units, "use_bias": self.use_bias})
        conf_dense = self.lay_dense2.get_config()
        for x in ["activation", "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                  "kernel_constraint", "bias_constraint", "kernel_initializer", "bias_initializer"]:
            config.update({x: conf_dense[x]})
        return config