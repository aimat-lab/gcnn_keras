import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.embedding import SplitEmbedding
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.keras import Dense, Activation, Add, Multiply, Concatenate
from kgcnn.layers.mlp import MLP, BatchNormMLP
from kgcnn.layers.update import MultiplyEquivariant
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingWeightedLocalEdges, PoolingGlobalEdges, \
    PoolingNodes
import kgcnn.ops.activ


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GIN')
class GIN(GraphBaseLayer):
    r"""Graph Isomorphism Network from: How Powerful are Graph Neural Networks?

    Computes graph convolution as
    :math:`h_\nu^{(k)} = \phi^{(k)} ((1+\epsilon^{(k)}) h_\nu^{k-1} + \sum\limits_{u\in N(\nu)}) h_u^{k-1}`.
    with optional learnable :math:`\epsilon^{(k)}`
    Note: The non-linear mapping :math:`\phi^{(k)}`, usually an MLP, is not included in this layer.

    Args:
        epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
    """

    def __init__(self,
                 pooling_method='sum',
                 epsilon_learnable=False,
                 **kwargs):
        """Initialize layer."""
        super(GIN, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable

        pool_args = {"pooling_method": pooling_method}

        # Layers
        self.lay_gather = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_pool = PoolingLocalEdges(**pool_args, **self._kgcnn_info)
        self.lay_add = Add(**self._kgcnn_info)

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(None, trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(GIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Node embeddings of shape (batch, [N], F)
        """
        node, edge_index = inputs
        ed = self.lay_gather([node, edge_index])
        nu = self.lay_pool([node, ed, edge_index])  # Summing for each node connection
        no = (1+self.eps_k)*node
        # no = node
        out = self.lay_add([no, nu])
        return out

    def get_config(self):
        """Update config."""
        config = super(GIN, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GCN')
class GCN(GraphBaseLayer):
    r"""Graph convolution according to Kipf et al.
    
    Computes graph convolution as :math:`\sigma(A_s(WX+b))` where :math:`A_s` is the precomputed and scaled adjacency
    matrix. The scaled adjacency matrix is defined by :math:`A_s = D^{-0.5} (A + I) D^{-0.5}` with the degree
    matrix :math:`D`. In place of :math:`A_s`, this layers uses edge features (that are the entries of :math:`A_s`) and
    edge indices. :math:`A_s` is considered pre-scaled, this is not done by this layer.
    If no scaled edge features are available, you could consider use e.g. "segment_mean", or normalize_by_weights to
    obtain a similar behaviour that is expected by a pre-scaled adjacency matrix input.
    Edge features must be possible to broadcast to node features. Ideally they have shape (..., 1).
    
    Args:
        units (int): Output dimension/ units of dense layer.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
            In this case the edge features are considered weights of dimension (...,1) and are summed for each node.
        activation (str): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}}.
        use_bias (bool): Use bias. Default is True.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units,
                 pooling_method='sum',
                 normalize_by_weights=False,
                 activation='kgcnn>leaky_relu',
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
        super(GCN, self).__init__(**kwargs)
        self.normalize_by_weights = normalize_by_weights
        self.pooling_method = pooling_method
        self.units = units

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}
        pool_args = {"pooling_method": pooling_method, "normalize_by_weights": normalize_by_weights}

        # Layers
        self.lay_gather = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_dense = Dense(units=self.units, activation='linear',
                               input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                               **kernel_args)
        self.lay_pool = PoolingWeightedLocalEdges(**pool_args, **self._kgcnn_info)
        self.lay_act = Activation(activation, ragged_validate=self.ragged_validate,
                                  input_tensor_type=self.input_tensor_type)

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Node embeddings of shape (batch, [N], F)
        """
        node, edges, edge_index = inputs
        no = self.lay_dense(node)
        no = self.lay_gather([no, edge_index])
        nu = self.lay_pool([node, no, edge_index, edges])  # Summing for each node connection
        out = self.lay_act(nu)
        return out

    def get_config(self):
        """Update config."""
        config = super(GCN, self).get_config()
        config.update({"normalize_by_weights": self.normalize_by_weights,
                       "pooling_method": self.pooling_method, "units": self.units})
        conf_dense = self.lay_dense.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "use_bias"]:
            config.update({x: conf_dense[x]})
        conf_act = self.lay_act.get_config()
        config.update({"activation": conf_act["activation"]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SchNetCFconv')
class SchNetCFconv(GraphBaseLayer):
    """Continuous filter convolution of SchNet.
    
    Edges are processed by 2 Dense layers, multiplied on outgoing node features and pooled for ingoing node.
    
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
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)
        self.lay_dense2 = Dense(units=self.units, activation='linear', use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)
        self.lay_sum = PoolingLocalEdges(pooling_method=cfconv_pool, **self._kgcnn_info)
        self.gather_n = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_mult = Multiply(**self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [N], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [N], 2)
        
        Returns:
            tf.RaggedTensor: Updated node features.
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2exp = self.gather_n([node, indexlist])
        x = self.lay_mult([node2exp, x])
        x = self.lay_sum([node, x, indexlist])
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


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='PAiNNconv')
class PAiNNconv(GraphBaseLayer):
    """Continuous filter convolution of PAiNN.

    Args:
        units (int): Units for Dense layer.
        conv_pool (str): Pooling method. Default is 'sum'.
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
                 conv_pool='sum',
                 use_bias=True,
                 activation='swish',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(PAiNNconv, self).__init__(**kwargs)
        self.conv_pool = conv_pool
        self.units = units
        self.use_bias = use_bias

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)
        self.lay_phi = Dense(units=self.units*3, activation='linear', use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)
        self.lay_w = Dense(units=self.units*3, activation='linear', use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)

        self.lay_split = SplitEmbedding(3, axis=-1)

        self.lay_sum = PoolingLocalEdges(pooling_method=conv_pool, **self._kgcnn_info)
        self.lay_sum_v = PoolingLocalEdges(pooling_method=conv_pool, **self._kgcnn_info)

        self.gather_n = GatherNodesOutgoing(**self._kgcnn_info)
        self.gather_v = GatherNodesOutgoing(**self._kgcnn_info)

        self.lay_mult = Multiply(**self._kgcnn_info)
        self.lay_mult_vv = MultiplyEquivariant(**self._kgcnn_info)
        self.lay_mult_vw = MultiplyEquivariant(**self._kgcnn_info)

        self.lay_add = Add(**self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(PAiNNconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: [nodes, equivariant, rbf, r_ij, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - equivariant (tf.RaggedTensor): Equivariant node embedding of shape (batch, [N], F, 3)
                - rdf (tf.RaggedTensor): Radial basis expansion pair-wise distance of shape (batch, [M], #Basis)
                - r_ij (tf.RaggedTensor): Normalized pair-wise distance of shape (batch, [M], 3)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tuple: [ds, dv]

                - ds (tf.RaggedTensor) Updated node features of shape (batch, [N], F)
                - dv (tf.RaggedTensor) Updated equivariant features of shape (batch, [N], F, 3)
        """
        node, equivariant, rbf, r_ij, indexlist = inputs
        s = self.lay_dense1(node)
        s = self.lay_phi(s)
        s = self.gather_n([s, indexlist])
        w = self.lay_w(rbf)
        # Cos-cutoff apply here TODO
        sw = self.lay_mult([s, w])
        sw1, sw2, sw3 = self.lay_split(sw)
        ds = self.lay_sum([node, sw1, indexlist])
        vj = self.gather_v([equivariant, indexlist])
        dv1 = self.lay_mult_vv([sw2, vj])
        dv2 = self.lay_mult_vw([sw3, r_ij])
        dv = self.lay_add([dv1, dv2])
        dv = self.lay_sum_v([node, dv, indexlist])
        return ds, dv

    def get_config(self):
        """Update layer config."""
        config = super(PAiNNconv, self).get_config()
        config.update({"conv_pool": self.conv_pool, "units": self.units})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config
