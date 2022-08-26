import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodes
from kgcnn.layers.pooling import PoolingLocalMessages, PoolingLocalEdgesLSTM


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GraphSageNodeLayer')
class GraphSageNodeLayer(GraphBaseLayer):
    r"""This is a convolutional layer for `GraphSAGE <http://arxiv.org/abs/1706.02216>`_  model as proposed
    by Hamilton et al. (2018). It is not used in the :obj:``kgcnn.literature.GraphSAGE`` model implementation
    but meant as a simplified module for other networks.

    Args:
        units: Dimensionality of embedding for each layer in the MLP.
        use_edge_features: Whether to use edge-features in addition to node features for convolution. Default is False.
        pooling_method: Pooling method to apply to node attributes. Default is "sum".
        activation: Activation function to use. Default is "relu".
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    """

    def __init__(self,
                 units,
                 use_edge_features=False,
                 pooling_method='sum',
                 activation='relu',
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize layer."""
        super(GraphSageNodeLayer, self).__init__(**kwargs)  # Sets additional kwargs for base GraphBaseLayer
        self.units = units
        self.pooling_method = pooling_method
        self.use_edge_features = use_edge_features
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}

        self.gather_nodes_outgoing = GatherNodesOutgoing()
        self.concatenate = LazyConcatenate()
        self.update_node_from_neighbors_mlp = GraphMLP(units=units, activation=activation, **kernel_args)
        self.update_node_from_self_mlp = GraphMLP(units=units, activation=activation, **kernel_args)
        if self.pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            # We do not allow full access to all parameters for the LSTM here for simplification.
            self.pooling = PoolingLocalEdgesLSTM(pooling_method=pooling_method, units=units)
        else:
            self.pooling = PoolingLocalMessages(pooling_method=pooling_method)
        self.normalize_nodes = GraphLayerNormalization(axis=-1)

    def build(self, input_shape):
        """Build layer."""
        super(GraphSageNodeLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edge_index] or [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Node embeddings of shape (batch, [N], F)
        """
        if self.use_edge_features:
            n, ed, edi = inputs
        else:
            n, edi = inputs
            ed = None

        neighboring_node_features = self.gather_nodes_outgoing([n, edi], **kwargs)
        if self.use_edge_features:
            neighboring_node_features = self.concatenate([neighboring_node_features, ed], **kwargs)

        neighboring_node_features = self.update_node_from_neighbors_mlp(neighboring_node_features, **kwargs)

        # Pool message
        nu = self.pooling([n, neighboring_node_features, edi], **kwargs)
        nu = self.concatenate([n, nu], **kwargs)  # LazyConcatenate node features with new edge updates

        n = self.update_node_from_self_mlp(nu, **kwargs)
        n = self.normalize_nodes(n, **kwargs)  # Normalize
        return n

    def get_config(self):
        """Update config."""
        config = super(GraphSageNodeLayer, self).get_config()
        config.update({"pooling_method": self.pooling_method, "units": self.units,
                       "use_edge_features": self.use_edge_features})
        conf_mlp = self.update_node_from_neighbors_mlp.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "use_bias", "activation"]:
            config.update({x: conf_mlp[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GraphSageEdgeUpdateLayer')
class GraphSageEdgeUpdateLayer(GraphBaseLayer):
    r"""An extension for `GraphSAGE <http://arxiv.org/abs/1706.02216>`_ model to have edge updates.

    It is a direct extension and should fit the GraphSAGE idea of message passing.

    Args:
        units: Dimensionality of embedding for each layer in the MLP.
        use_normalization: Whether to use GraphLayerNormalization at the output of the update. Default is True.
        activation: Activation function to use. Default is "relu".
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_normalization=True,
                 **kwargs):
        super(GraphSageEdgeUpdateLayer, self).__init__(**kwargs)  # Sets additional kwargs for base GraphBaseLayer
        self.units = units
        self.use_normalization = use_normalization
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}
        # non-stateful layers
        self.gather_nodes = GatherNodes()
        self.concatenate = LazyConcatenate()
        self.update_edge_mlp = GraphMLP(units=units, activation=activation, **kernel_args)

        # normalization layer for edge features
        self.normalize_edges = GraphLayerNormalization(axis=-1)

    def build(self, input_shape):
        """Build layer."""
        super(GraphSageEdgeUpdateLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Edge embeddings of shape (batch, [M], F)
        """
        n, ed, edi = inputs
        node_features = self.gather_nodes([n, edi], **kwargs)
        ed_new_input = self.concatenate([ed, node_features], **kwargs)
        ed = self.update_edge_mlp(ed_new_input, **kwargs)
        if self.use_normalization:
            ed = self.normalize_edges(ed, **kwargs)
        return ed

    def get_config(self):
        """Update config."""
        config = super(GraphSageEdgeUpdateLayer, self).get_config()
        config.update({"units": self.units, "use_normalization": self.use_normalization})
        conf_mlp = self.update_edge_mlp.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "use_bias", "activation"]:
            config.update({x: conf_mlp[x]})
        return config
