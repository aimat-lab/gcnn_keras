from keras.layers import Layer, Subtract, Dense, Dropout, Concatenate, Activation, GRUCell
from kgcnn.layers.gather import GatherState, GatherNodes
from kgcnn.layers.aggr import AggregateLocalEdgesAttention
from kgcnn.layers.pooling import PoolingNodes, PoolingNodesAttention
from kgcnn.layers.update import GRUUpdate
import kgcnn.ops.activ

# Gated recurrent unit update. See kgcnn.layers.conv.mpnn_conv for details.
HamNetGRUUnion = GRUUpdate


class HamNetNaiveUnion(Layer):
    r"""Simple union that concatenates a feature tensor :math:`\mathbf{x}` and its updates :math:`\mathbf{x}_u`
    and applies a fully connected dense layer,
    i.e. a linear transformation with weights :math:`\mathbf{W}^{\top}`, :math:`\mathbf{b}` plus activation
    :math:`\sigma`.

    .. math::

        \mathbf{x}^{\prime} = \sigma \left[ \left( \mathbf{x} \; || \; \mathbf{x}_u \right) \mathbf{W}^{\top} +
        \mathbf{b} \right]

    """
    def __init__(self,
                 units: int,
                 activation="kgcnn>leaky_relu2",
                 use_bias: bool = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        r"""Initialize layer with arguments of :obj:`ks.layers.Dense`.

        Args:
            units (int): Positive integer, dimensionality of the output space.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`). Default is "kgcnn>leaky_relu2".
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
        super(HamNetNaiveUnion, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.units = int(units)
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        self.lay_dense = Dense(units=units, activation=activation, use_bias=use_bias, **kernel_args)
        self.lay_concat = Concatenate()

    def build(self, input_shape):
        """Build layer."""
        super(HamNetNaiveUnion, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, node_updates]

                - nodes (Tensor): Node features of shape `([N], F)`
                - node_updates (Tensor): Node features of shape `([N], F)`

        Returns:
            Tensor: Embedding tensor of updated node features of shape `([N], F)`.
        """
        n, nu = inputs
        nnu = self.lay_concat([n, nu], **kwargs)
        n_out = self.lay_dense(nnu, **kwargs)
        return n_out

    def get_config(self):
        """Update layer config."""
        config = super(HamNetNaiveUnion, self).get_config()
        config.update({"units": self.units, "use_bias": self.use_bias})
        conf_dense = self.lay_dense.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "use_bias", "activation"]:
            if x in conf_dense.keys():
                config.update({x: conf_dense[x]})
        return config


class HamNetGlobalReadoutAttend(Layer):
    r"""Computes attentive updates for fingerprint generation according to `HamNet <https://arxiv.org/abs/2105.03688>`_.
    The naming convention follows the authors `implementation <https://github.com/PKUterran/MoleculeClub>`_.
    The layer is used in :obj:`HamNetFingerprintGenerator` and computes the attentive state updates.
    The node features are first transformed by a :obj:`Dense` layer:
    :math:`\mathbf{h}' = \sigma\;(\mathbf{h} \mathbf{W}^T)` which yields the attention coefficients from state
    :math:`\mathbf{s}`:

    .. math::
        a_i = w^T [\mathbf{h}_i' \; || \; \mathbf{s}]

    with :math:`\alpha_i = \text{softmax}({a_i \; | \; i \in V})` the final state update :math:`\mathbf{m}`:

    .. math::
        \mathbf{m} = \sigma \; \sum_i \alpha_i \mathbf{h}'_i

    Update :math:`\mathbf{m}` is returned by the layer. Here, :math:`\sigma` denotes an activation function.

    """

    def __init__(self,
                 units,
                 activation="kgcnn>leaky_relu2",
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
        """Initialize layer.

        Args:
            units (int): Units for the linear transformation of node features before attention.
            activation (str, dict): Activation. Default is "kgcnn>leaky_relu2".
            activation_last (str, dict): Last activation for messages. Default is "elu".
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
            use_dropout (bool): Whether to use dropout on input features. Default is False.
            rate (float): Float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
            seed (int): A Python integer to use as random seed.
        """
        super(HamNetGlobalReadoutAttend, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.units = int(units)
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        self.gather_state = GatherState()
        if self.use_dropout:
            self.dropout_layer = Dropout(rate=rate, noise_shape=noise_shape, seed=seed)
        self.dense_attend = Dense(units=units, activation=activation, use_bias=use_bias, **kernel_args)
        self.dense_align = Dense(1, activation="linear", use_bias=use_bias, **kernel_args)
        self.lay_concat = Concatenate(axis=-1)
        self.pool_attention = PoolingNodesAttention()
        self.final_activ = Activation(activation=activation_last,
                                      activity_regularizer=activity_regularizer)

    def build(self, input_shape):
        """Build layer."""
        super(HamNetGlobalReadoutAttend, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [state, nodes, batch_id_nodes]

                - state (Tensor): Molecular embedding of shape `(batch, F)`
                - nodes (Tensor): Node features of shape `([N], F)`

        Returns:
            Tensor: Embedding tensor of pooled node attentions of shape (batch, F)
        """
        hm_ftr, hv_ftr, batch_id_nodes = inputs
        hm_v_ftr = self.gather_state([hm_ftr, batch_id_nodes], **kwargs)

        attend_ftr = hv_ftr
        if self.use_dropout:
            attend_ftr = self.dropout_layer(attend_ftr, **kwargs)
        attend_ftr = self.dense_attend(attend_ftr, **kwargs)
        align_ftr = self.lay_concat([hm_v_ftr, hv_ftr], **kwargs)
        if self.use_dropout:
            align_ftr = self.dropout_layer(align_ftr, **kwargs)
        align_ftr = self.dense_align(align_ftr, **kwargs)
        mm_ftr = self.pool_attention([hm_ftr, attend_ftr, align_ftr, batch_id_nodes], **kwargs)
        mm_ftr = self.final_activ(mm_ftr, **kwargs)
        return mm_ftr, align_ftr

    def get_config(self):
        """Update layer config."""
        config = super(HamNetGlobalReadoutAttend, self).get_config()
        config.update({"use_bias": self.use_bias, "units": self.units, "use_dropout": self.use_dropout})
        conf_sub = self.dense_attend.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            if x in conf_sub:
                config.update({x: conf_sub[x]})
        if self.use_dropout:
            conf_drop = self.dropout_layer.get_config()
            for x in ["rate", "noise_shape", "seed"]:
                if x in conf_drop.keys():
                    config.update({x: conf_drop[x]})
        conf_last = self.final_activ.get_config()
        config.update({"activation_last": conf_last["activation"]})
        return config


class HamNetFingerprintGenerator(Layer):
    r"""Computes readout or fingerprint generation according to `HamNet <https://arxiv.org/abs/2105.03688>`__ .
    The naming convention follows the authors `implementation <https://github.com/PKUterran/MoleculeClub>`__ .
    The layer generates a molecular or global message by iteratively updating from node embeddings. Initial state
    :math:`\mathbf{s}^0 = \frac{1}{n} \sum_i \sigma (\mathbf{h} W^T)` is updated :math:`l=1\dots L` times from
    messages :math:`\mathbf{m}^l` via a gated recurrent unit and subsequent activation :math:`\sigma`:

    .. math::
        \mathbf{s}^{l+1} = \sigma \left[\; \text{GRU}(\mathbf{s}^l, \mathbf{m}^{l}) \;\right]

    The message is obtained from an attentive readout function :math:`f` which is implemented here in
    :obj:`HamNetGlobalReadoutAttend`:

    .. math::
        \mathbf{m}^{l+1} = f(\mathbf{h}, \mathbf{m}^l)

    The final embedding :math:`\mathbf{s}^L` is used as output or molecular state.

    """

    def __init__(self,
                 units: int,
                 units_attend: int,
                 activation="kgcnn>leaky_relu2",
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
        """Initialize layer.

        Args:
            units (int): Units for the linear transformation of node features before attention.
            units_attend (int): Units for attention attributes.
            activation (str, dict): Activation. Default is "kgcnn>leaky_relu2".
            use_bias (bool): Boolean, whether the layer uses a bias vector. Default is True.
            depth (int): Number of iterations. Default is 4.
            pooling_method(str): Initial pooling before iteration. Default is "mean".
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
            recurrent_activation: Activation function to use for the recurrent step.
                Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
                applied (ie. "linear" activation: `a(x) = x`).
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix, used for the linear transformation of the recurrent state.
                Default: `orthogonal`.
            recurrent_regularizer: Regularizer function applied to the
                `recurrent_kernel` weights matrix. Default: `None`.
            recurrent_constraint: Constraint function applied to the `recurrent_kernel`
                weights matrix. Default: `None`.
            dropout: Float between 0 and 1. Fraction of the units to drop for the
                linear transformation of the inputs. Default: 0.
            recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
                the linear transformation of the recurrent state. Default: 0.
            reset_after: GRU convention (whether to apply reset gate after or
                before matrix multiplication). False = "before",
                True = "after" (default and cuDNN compatible).
            use_dropout (bool): Whether to use dropout on input features. Default is False.
            rate (float): Float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
            seed (int): A Python integer to use as random seed.
        """
        super(HamNetFingerprintGenerator, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
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
        self.vertex2mol = Dense(
            units=units, activation=activation, use_bias=use_bias, **kernel_args)

        self.readouts = [HamNetGlobalReadoutAttend(
            units=units_attend, activation=activation, activation_last="elu", use_bias=use_bias,
            use_dropout=use_dropout, rate=rate, noise_shape=noise_shape, seed=seed,
            **kernel_args) for _ in range(self.depth)]

        self.unions = [GRUCell(units=units, activation="tanh", **gru_args) for _ in range(self.depth)]
        self.final_activ = Activation(activation=activation,
                                      activity_regularizer=activity_regularizer)

    def build(self, input_shape):
        """Build layer."""
        super(HamNetFingerprintGenerator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [reference, nodes, batch_id]

                - state (Tensor): reference of shape `(batch, )`
                - nodes (Tensor): Node features of shape `([N], F)`
                - batch_id (Tensor): Batch ID of nodes of shape `([N], )`

        Returns:
            Tensor: Embedding tensor of pooled node attentions of shape `(batch, F)`
        """
        ref, hv_ftr, batch_id = inputs
        hm_ftr = self.vertex2mol(hv_ftr, **kwargs)
        hm_ftr = self.pool_nodes([ref, hm_ftr, batch_id], **kwargs)
        alignments = []
        for i in range(self.depth):
            mm_ftr, align = self.readouts[i]([hm_ftr, hv_ftr, batch_id], **kwargs)
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
            if x in conf_sub.keys():
                config.update({x: conf_sub[x]})
        if len(self.unions) > 0:
            conf_gru = self.unions[0].get_config()
            for x in ["recurrent_activation", "recurrent_initializer", "recurrent_regularizer", "recurrent_constraint",
                      "dropout", "recurrent_dropout", "reset_after"]:
                if x in conf_gru.keys():
                    config.update({x: conf_gru[x]})
        if len(self.readouts) > 0:
            conf_read = self.readouts[0].get_config()
            for x in ["use_dropout", "seed", "rate", "noise_shape"]:
                if x in conf_read.keys():
                    config.update({x: conf_read[x]})
        return config


class HamNaiveDynMessage(Layer):
    r"""Message passing block from `HamNet <https://arxiv.org/abs/2105.03688>`__ which makes use of attention.
    The naming convention follows the authors `implementation <https://github.com/PKUterran/MoleculeClub>`__ .
    The layer computes the following, let :math:`\mathbf{h}`, :math:`\mathbf{\epsilon}_{ij}` be node, edge features
    and :math:`\mathbf{q}`, :math:`\mathbf{p}` be (generalized) node coordinates and momentum. With
    :math:`\mathbf{p}_{ij} = \mathbf{p}_{j} - \mathbf{p}_{i}` and
    :math:`\mathbf{q}_{ij} = \mathbf{q}_{j} - \mathbf{q}_{i}` the attention coefficients read:

    .. math::
        \mathbf{a}_{ij} = \mathbf{w}^T \left(\mathbf{p}_{ij} \; || \; \mathbf{q}_{ij} \; ||
        \mathbf{\epsilon}_{ij} \right)

    and the new node update or message, using the attention coefficients
    :math:`\alpha_{ij} = \, \text{softmax}(\{\mathbf{a}_{ij} \; | \; j \in \mathcal{N}(i)\})`:

    .. math::
        \mathbf{m}_{v} = \sigma \; \sum_{j \in \mathcal{N}(i)} \; \alpha_{ij} \;  \sigma
        \left[ \; \mathbf{h}_j \; \mathbf{W}^T \;  \right]

    and edge updates:

    .. math::
        \mathbf{m}_{e} = \sigma \left[ \left(\; \mathbf{h}_i \; || \; \mathbf{q}_{ij} \; || \;
        \mathbf{h}_j \; \right) \mathbf{W}^T \right]

    the layer returns :math:`\mathbf{m}_{v}` and :math:`\mathbf{m}_{e}`.

    """
    def __init__(self,
                 units,
                 units_edge,
                 activation="kgcnn>leaky_relu2",
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
        r"""Initialize layer.

        Args:
            units (int): Units for the linear transformation of node features before attention.
            units_edge (int): Units for :obj:`Dense` layer for edge updates.
            activation (str, dict): Activation. Default is "kgcnn>leaky_relu2".
            activation_last (str, dict): Last activation for messages. Default is "elu".
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
            use_dropout (bool): Whether to use dropout on input features. Default is False.
            rate (float): Float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
            seed (int): A Python integer to use as random seed.
        """
        super(HamNaiveDynMessage, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.units = int(units)
        self.units_edge = int(units_edge)
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        self.gather_v = GatherNodes(split_indices=[0, 1], concat_axis=None)
        self.gather_p = GatherNodes(split_indices=[0, 1], concat_axis=None)
        self.gather_q = GatherNodes(split_indices=[0, 1], concat_axis=None)
        self.lazy_sub_p = Subtract()
        self.lazy_sub_q = Subtract()
        # self.lay_concat = LazyConcatenate(axis=-1)
        self.lay_concat_align = Concatenate(axis=-1)
        self.lay_concat_edge = Concatenate(axis=-1)
        if self.use_dropout:
            self.dropout_layer = Dropout(rate=rate, noise_shape=noise_shape, seed=seed)
        self.dense_attend = Dense(units=units, use_bias=use_bias, activation=activation, **kernel_args)
        self.dense_align = Dense(1, activation="linear", use_bias=use_bias, **kernel_args)
        self.dense_e = Dense(units=units_edge, activation=activation, use_bias=use_bias, **kernel_args)
        self.pool_attention = AggregateLocalEdgesAttention()
        self.final_activ = Activation(activation=activation_last,
                                      activity_regularizer=activity_regularizer)

    def build(self, input_shape):
        """Build layer."""
        super(HamNaiveDynMessage, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [hv_ftr, he_ftr, p_ftr, q_ftr, edge_index]

                - hv_ftr (Tensor): Node features of shape `([N], F)`
                - he_ftr (Tensor): Edge features of shape `([M], F)`
                - p_ftr (Tensor): Momentum node features of shape `([N], F)`
                - q_ftr (Tensor): Positional node features of shape `([N], F)`
                - edge_index (Tensor): Edge connection index list of shape `(2, [M])`

        Returns:
            list: [mv_ftr, me_ftr]

                - mv_ftr (Tensor): Node feature updates of shape `([N], F)`
                - me_ftr (Tensor): Edge feature updates of shape `([M], F)`
        """
        hv_ftr, he_ftr, p_ftr, q_ftr, edi = inputs
        if self.use_dropout:
            hv_ftr = self.dropout_layer(hv_ftr, **kwargs)
            he_ftr = self.dropout_layer(he_ftr, **kwargs)
        hv_u_ftr, hv_v_ftr = self.gather_v([hv_ftr, edi], **kwargs)
        q_u_ftr, q_v_ftr = self.gather_p([q_ftr, edi], **kwargs)
        p_u_ftr, p_v_ftr = self.gather_q([p_ftr, edi], **kwargs)
        p_uv_ftr = self.lazy_sub_p([p_v_ftr, p_u_ftr], **kwargs)
        q_uv_ftr = self.lazy_sub_q([q_v_ftr, q_u_ftr], **kwargs)

        attend_ftr = self.dense_attend(hv_v_ftr, **kwargs)

        align_ftr = self.lay_concat_align([p_uv_ftr, q_uv_ftr, he_ftr], **kwargs)
        align_ftr = self.dense_align(align_ftr, **kwargs)
        mv_ftr = self.pool_attention([hv_ftr, attend_ftr, align_ftr, edi], **kwargs)
        mv_ftr = self.final_activ(mv_ftr, **kwargs)

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
            if x in conf_sub.keys():
                config.update({x: conf_sub[x]})
        if self.use_dropout:
            conf_drop = self.dropout_layer.get_config()
            for x in ["rate", "noise_shape", "seed"]:
                if x in conf_drop.keys():
                    config.update({x: conf_drop[x]})
        conf_last = self.final_activ.get_config()
        config.update({"activation_last": conf_last["activation"]})
        return config
