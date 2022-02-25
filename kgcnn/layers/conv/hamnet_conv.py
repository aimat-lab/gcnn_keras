import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.pooling import PoolingLocalEdgesAttention, PoolingNodes, PoolingNodesAttention
from kgcnn.layers.modules import LazySubtract, DenseEmbedding, DropoutEmbedding, LazyConcatenate, ActivationEmbedding


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='HamNetGlobalReadoutAttend')
class HamNetGlobalReadoutAttend(GraphBaseLayer):
    def __init__(self,
                 units,
                 activation="kgcnn>leaky_relu",
                 activation_last="elu",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_dropout=False,
                 rate=None, noise_shape=None, seed=None,
                 **kwargs):
        """Initialize layer."""
        super(HamNetGlobalReadoutAttend, self).__init__(**kwargs)
        self.units = int(units)
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        self.gather_state = GatherState()
        if self.use_dropout:
            self.dropout_layer = DropoutEmbedding(rate=rate, noise_shape=noise_shape, seed=seed)
        self.dense_attend = DenseEmbedding(units=units, activation=activation, use_bias=use_bias, **kernel_args)
        self.dense_align = DenseEmbedding(1, activation="linear", use_bias=use_bias, **kernel_args)
        self.lay_concat = LazyConcatenate(axis=-1)
        self.pool_attention = PoolingNodesAttention()
        self.final_activ = ActivationEmbedding(activation=activation_last,
                                               activity_regularizer=activity_regularizer)

    def build(self, input_shape):
        """Build layer."""
        super(HamNetGlobalReadoutAttend, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [state, nodes]

                - state (tf.Tensor): Molecular embedding of shape (batch, F)
                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled node attentions of shape (batch, F)
        """
        hm_ftr, hv_ftr = inputs
        hm_v_ftr = self.gather_state([hm_ftr, hv_ftr], **kwargs)

        attend_ftr = hv_ftr
        if self.use_dropout:
            attend_ftr = self.dropout_layer(attend_ftr, **kwargs)
        attend_ftr = self.dense_attend(attend_ftr, **kwargs)
        align_ftr = self.lay_concat([hm_v_ftr, hv_ftr], **kwargs)
        if self.use_dropout:
            align_ftr = self.dropout_layer(align_ftr, **kwargs)
        align_ftr = self.dense_align(align_ftr, **kwargs)
        mm_ftr = self.pool_attention([attend_ftr, align_ftr], **kwargs)
        mm_ftr = self.final_activ(mm_ftr, **kwargs)
        return mm_ftr, align_ftr

    def get_config(self):
        """Update layer config."""
        config = super(HamNetGlobalReadoutAttend, self).get_config()
        config.update({"use_bias": self.use_bias, "units": self.units, "use_dropout": self.use_dropout})
        conf_sub = self.dense_attend.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        if self.use_dropout:
            conf_drop = self.dropout_layer.get_config()
            for x in ["rate", "noise_shape", "seed"]:
                config.update({x: conf_drop[x]})
        conf_last = self.final_activ.get_config()
        config.update({"activation_last": conf_last["activation"]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='HamNetFingerprintGenerator')
class HamNetFingerprintGenerator(GraphBaseLayer):
    r"""Computes readout or fingerprint generation according to `HamNet <https://arxiv.org/abs/2105.03688>`_ .

    Args:
        units (int): Units for the linear trafo of node features before attention.
        units_attend (int): Units fot attention attributes.
        activation (str, dict): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}},
        use_bias (bool): Use bias. Default is True.
        depth (int): Number of iterations. Default is 4.
        pooling_method(str): Initial pooling before iteration. Default is "mean".
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units: int,
                 units_attend: int,
                 activation="kgcnn>leaky_relu",
                 use_bias: bool = True,
                 depth=4,
                 pooling_method="mean",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 recurrent_activation='sigmoid',
                 recurrent_initializer='orthogonal',
                 recurrent_regularizer=None,
                 recurrent_constraint=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 reset_after=True,
                 use_dropout=False,
                 rate=None, noise_shape=None, seed=None,
                 **kwargs):
        """Initialize layer."""
        super(HamNetFingerprintGenerator, self).__init__(**kwargs)
        self.units = int(units)
        self.units_attend = int(units_attend)
        self.use_bias = bool(use_bias)
        self.use_dropout = use_dropout
        self.depth = int(depth)
        self.pooling_method = pooling_method
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        gru_args = {"recurrent_activation": recurrent_activation,
                    "use_bias": use_bias, "kernel_initializer": kernel_initializer,
                    "recurrent_initializer": recurrent_initializer, "bias_initializer": bias_initializer,
                    "kernel_regularizer": kernel_regularizer, "recurrent_regularizer": recurrent_regularizer,
                    "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                    "recurrent_constraint": recurrent_constraint, "bias_constraint": bias_constraint,
                    "dropout": dropout, "recurrent_dropout": recurrent_dropout, "reset_after": reset_after}
        self.pool_nodes = PoolingNodes(pooling_method=self.pooling_method)
        self.vertex2mol = DenseEmbedding(
            units=units, activation=activation, use_bias=use_bias, **kernel_args)

        self.readouts = [HamNetGlobalReadoutAttend(
            units=units_attend, activation=activation, activation_last="elu", use_bias=use_bias,
            use_dropout=use_dropout, rate=rate, noise_shape=noise_shape, seed=seed,
            **kernel_args) for _ in range(self.depth)]

        self.unions = [tf.keras.layers.GRUCell(
            units=units, activation="tanh", **gru_args) for _ in range(self.depth)]
        self.final_activ = ActivationEmbedding(activation=activation,
                                               activity_regularizer=activity_regularizer)

    def build(self, input_shape):
        """Build layer."""
        super(HamNetFingerprintGenerator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled node attentions of shape (batch, F)
        """
        self.assert_ragged_input_rank(inputs)
        hv_ftr = inputs
        hm_ftr = self.vertex2mol(hv_ftr, **kwargs)
        hm_ftr = self.pool_nodes(hm_ftr, **kwargs)
        alignments = []
        for i in range(self.depth):
            mm_ftr, align = self.readouts[i]([hm_ftr, hv_ftr], **kwargs)
            # alignments.append(align)
            hm_ftr, _ = self.unions[i](mm_ftr, hm_ftr, **kwargs)
            hm_ftr = self.final_activ(hm_ftr, **kwargs)
        return hm_ftr

    def get_config(self):
        """Update layer config."""
        config = super(HamNetFingerprintGenerator, self).get_config()
        config.update({"use_bias": self.use_bias, "units": self.units, "units_attend": self.units_attend,
                       "use_dropout": self.use_dropout, "depth": self.depth, "pooling_method": self.pooling_method})
        conf_sub = self.vertex2mol.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        if len(self.unions) > 0:
            conf_gru = self.unions[0].get_config()
            for x in ["recurrent_activation", "recurrent_initializer", "recurrent_regularizer", "recurrent_constraint",
                      "dropout", "recurrent_dropout", "reset_after"]:
                config.update({x: conf_gru[x]})
        if len(self.readouts) > 0:
            conf_read = self.readouts[0].get_config()
            for x in ["use_dropout", "seed", "rate", "noise_shape", ]:
                config.update({x: conf_read[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='HamNaiveDynMessage')
class HamNaiveDynMessage(GraphBaseLayer):
    def __init__(self,
                 units,
                 units_edge,
                 activation="kgcnn>leaky_relu",
                 activation_last="elu",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_dropout=False,
                 rate=None, noise_shape=None, seed=None,
                 **kwargs):
        """Initialize layer."""
        super(HamNaiveDynMessage, self).__init__(**kwargs)
        self.units = int(units)
        self.units_edge= int(units_edge)
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        self.gather_v = GatherNodes(concat_axis=None, split_axis=2)
        self.gather_p = GatherNodes(concat_axis=None, split_axis=2)
        self.gather_q = GatherNodes(concat_axis=None, split_axis=2)
        self.lazy_sub_p = LazySubtract()
        self.lazy_sub_q = LazySubtract()
        self.lay_concat = LazyConcatenate(axis=-1)
        self.lay_concat_align = LazyConcatenate(axis=-1)
        self.lay_concat_edge = LazyConcatenate(axis=-1)
        if self.use_dropout:
            self.dropout_layer = DropoutEmbedding(rate=rate, noise_shape=noise_shape, seed=seed)
        self.dense_attend = DenseEmbedding(units=units, use_bias=use_bias, activation=activation, **kernel_args)
        self.dense_align = DenseEmbedding(1, activation="linear", use_bias=use_bias, **kernel_args)
        self.dense_e = DenseEmbedding(units=units_edge, activation=activation, use_bias=use_bias, **kernel_args)
        self.lay_concat = LazyConcatenate(axis=-1)
        self.pool_attention = PoolingLocalEdgesAttention()
        self.final_activ = ActivationEmbedding(activation=activation_last,
                                               activity_regularizer=activity_regularizer)

    def build(self, input_shape):
        """Build layer."""
        super(HamNaiveDynMessage, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [hv_ftr, he_ftr, p_ftr, q_ftr, edge_index]

                - state (tf.Tensor): Molecular embedding of shape (batch, F)
                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled node attentions of shape (batch, F)
        """
        hv_ftr, he_ftr, p_ftr, q_ftr, edi = inputs
        if self.use_dropout:
            hv_ftr = self.dropout_layer(hv_ftr, **kwargs)
            he_ftr = self.dropout_layer(he_ftr, **kwargs)
        hv_u_ftr, hv_v_ftr = self.gather_v([hv_ftr, edi], **kwargs)
        q_u_ftr, q_v_ftr = self.gather_p([q_ftr, edi], **kwargs)
        p_u_ftr, p_v_ftr = self.gather_q([p_ftr, edi], **kwargs)
        p_uv_ftr = self.lazy_sub_p([p_v_ftr, p_u_ftr], **kwargs)
        q_uv_ftr = self.lazy_sub_p([q_v_ftr, q_u_ftr], **kwargs)
        he2_ftr = self.lay_concat([he_ftr, he_ftr], **kwargs)

        attend_ftr = self.dense_attend(hv_v_ftr, **kwargs)

        align_ftr = self.lay_concat_align([p_uv_ftr, q_uv_ftr, he2_ftr], **kwargs)
        align_ftr = self.dense_align(align_ftr, **kwargs)
        mv_ftr = self.pool_attention([hv_ftr, attend_ftr, align_ftr, edi], **kwargs)
        mv_ftr = self.final_activ(mv_ftr , **kwargs)

        me_ftr = self.lay_concat_edge([hv_u_ftr, p_uv_ftr, q_uv_ftr, hv_v_ftr], **kwargs)
        me_ftr = self.dense_e(me_ftr)

        return mv_ftr, me_ftr

    def get_config(self):
        """Update layer config."""
        config = super(HamNaiveDynMessage, self).get_config()
        config.update({"use_bias": self.use_bias, "units": self.units, "use_dropout": self.use_dropout})
        conf_sub = self.dense_attend.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        if self.use_dropout:
            conf_drop = self.dropout_layer.get_config()
            for x in ["rate", "noise_shape", "seed"]:
                config.update({x: conf_drop[x]})
        conf_last = self.final_activ.get_config()
        config.update({"activation_last": conf_last["activation"]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='HamiltonEngine')
class HamiltonEngine(GraphBaseLayer):
    def __init__(self,
                 units,
                 activation="kgcnn>leaky_relu",
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
        super(HamiltonEngine, self).__init__(**kwargs)
        self.units = int(units)
        self.use_bias = bool(use_bias)
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        self.dense_atom = DenseEmbedding(units=units, activation="tanh", use_bias=use_bias, **kernel_args)
        self.dense_edge = DenseEmbedding(units=units, activation="tanh", use_bias=use_bias, **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(HamiltonEngine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edge, edge_indices]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled node attentions of shape (batch, F)
        """
        atom_ftr, bond_ftr, edge_idx = inputs
        hv_ftr = self.dense_atom(atom_ftr, **kwargs)
        he_ftr = self.dense_edge(bond_ftr, **kwargs)
        return

    def get_config(self):
        """Update layer config."""
        config = super(HamiltonEngine, self).get_config()
        config.update({"use_bias": self.use_bias, "units": self.units,})
        return config
