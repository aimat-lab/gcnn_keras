import tensorflow as tf
from kgcnn.ops.axis import get_positive_axis
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import LazyAdd, LazyMultiply, Dense, LazyConcatenate, ExpandDims
from kgcnn.layers.geom import EuclideanNorm, ScalarProduct
from kgcnn.layers.gather import GatherNodesOutgoing
ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='PAiNNconv')
class PAiNNconv(GraphBaseLayer):
    """Continuous filter convolution block of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ .

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
                 cutoff=None,
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
        self.cutoff = cutoff

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_phi = Dense(units=self.units * 3, activation='linear', use_bias=self.use_bias, **kernel_args)
        self.lay_w = Dense(units=self.units * 3, activation='linear', use_bias=self.use_bias, **kernel_args)

        self.lay_split = SplitEmbedding(3, axis=-1)
        self.lay_sum = AggregateLocalEdges(pooling_method=conv_pool)
        self.lay_sum_v = AggregateLocalEdges(pooling_method=conv_pool)
        self.gather_n = GatherNodesOutgoing()
        self.gather_v = GatherNodesOutgoing()
        self.lay_mult = LazyMultiply()
        if self.cutoff is not None:
            self.lay_mult_cutoff = LazyMultiply()
        self.lay_exp_vv = ExpandDims(axis=-2)
        self.lay_exp_vw = ExpandDims(axis=-2)
        self.lay_exp_r = ExpandDims(axis=-1)
        self.lay_mult_vv = LazyMultiply()
        self.lay_mult_vw = LazyMultiply()

        self.lay_add = LazyAdd()

    def build(self, input_shape):
        """Build layer."""
        super(PAiNNconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: [nodes, equivariant, rbf, envelope, r_ij, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - equivariant (tf.RaggedTensor): Equivariant node embedding of shape (batch, [N], 3, F)
                - rdf (tf.RaggedTensor): Radial basis expansion pair-wise distance of shape (batch, [M], #Basis)
                - envelope (tf.RaggedTensor): Distance envelope of shape (batch, [N], 1)
                - r_ij (tf.RaggedTensor): Normalized pair-wise distance of shape (batch, [M], 3)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tuple: [ds, dv]

                - ds (tf.RaggedTensor) Updated node features of shape (batch, [N], F)
                - dv (tf.RaggedTensor) Updated equivariant features of shape (batch, [N], F, 3)
        """
        node, equivariant, rbf, envelope, r_ij, indexlist = inputs
        s = self.lay_dense1(node)
        s = self.lay_phi(s)
        s = self.gather_n([s, indexlist])
        w = self.lay_w(rbf)
        if self.cutoff is not None:
            w = self.lay_mult_cutoff([w, envelope])
        sw = self.lay_mult([s, w])
        sw1, sw2, sw3 = self.lay_split(sw)
        ds = self.lay_sum([node, sw1, indexlist])
        vj = self.gather_v([equivariant, indexlist])
        sw2 = self.lay_exp_vv(sw2)
        dv1 = self.lay_mult_vv([sw2, vj])
        sw3 = self.lay_exp_vw(sw3)
        r_ij = self.lay_exp_r(r_ij)
        dv2 = self.lay_mult_vw([sw3, r_ij])
        dv = self.lay_add([dv1, dv2])
        dv = self.lay_sum_v([node, dv, indexlist])
        return ds, dv

    def get_config(self):
        """Update layer config."""
        config = super(PAiNNconv, self).get_config()
        config.update({"conv_pool": self.conv_pool, "units": self.units, "cutoff": self.cutoff})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='PAiNNUpdate')
class PAiNNUpdate(GraphBaseLayer):
    """Continuous filter convolution of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ .

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
        add_eps: Whether to add eps in the norm.
    """

    def __init__(self, units,
                 use_bias=True,
                 activation='swish',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 add_eps: bool = False,
                 **kwargs):
        """Initialize Layer."""
        super(PAiNNUpdate, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.add_eps = add_eps

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_lin_u = Dense(self.units, activation='linear', use_bias=False, **kernel_args)
        self.lay_lin_v = Dense(self.units, activation='linear', use_bias=False, **kernel_args)
        self.lay_a = Dense(units=self.units * 3, activation='linear', use_bias=self.use_bias, **kernel_args)

        self.lay_scalar_prod = ScalarProduct(axis=2)
        self.lay_norm = EuclideanNorm(axis=2, add_eps=self.add_eps)
        self.lay_concat = LazyConcatenate(axis=-1)
        self.lay_split = SplitEmbedding(3, axis=-1)

        self.lay_mult = LazyMultiply()
        self.lay_exp_v = ExpandDims(axis=-2)
        self.lay_mult_vv = LazyMultiply()
        self.lay_add = LazyAdd()

    def build(self, input_shape):
        """Build layer."""
        super(PAiNNUpdate, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: [nodes, equivariant]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - equivariant (tf.RaggedTensor): Equivariant node embedding of shape (batch, [N], 3, F)

        Returns:
            tuple: [ds, dv]

                - ds (tf.RaggedTensor) Updated node features of shape (batch, [N], F)
                - dv (tf.RaggedTensor) Updated equivariant features of shape (batch, [N], 3, F)
        """
        node, equivariant = inputs
        v_v = self.lay_lin_v(equivariant, **kwargs)
        v_u = self.lay_lin_u(equivariant, **kwargs)
        v_prod = self.lay_scalar_prod([v_u, v_v], **kwargs)
        v_norm = self.lay_norm(v_v, **kwargs)
        a = self.lay_concat([node, v_norm], **kwargs)
        a = self.lay_dense1(a, **kwargs)
        a = self.lay_a(a, **kwargs)
        a_vv, a_sv, a_ss = self.lay_split(a, **kwargs)
        a_vv = self.lay_exp_v(a_vv, **kwargs)
        dv = self.lay_mult_vv([a_vv, v_u], **kwargs)
        ds = self.lay_mult([v_prod, a_sv], **kwargs)
        ds = self.lay_add([ds, a_ss], **kwargs)
        return ds, dv

    def get_config(self):
        """Update layer config."""
        config = super(PAiNNUpdate, self).get_config()
        config.update({"units": self.units, "add_eps": self.add_eps})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EquivariantInitialize')
class EquivariantInitialize(GraphBaseLayer):
    """Equivariant initializer of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ .

    Args:
        dim (int): Dimension of equivariant features. Default is 3.
        method (str): How to initialize equivariant tensor. Default is "zeros".
    """

    def __init__(self, dim=3, method: str = "zeros", value: float = 1.0, stddev: float = 1.0, **kwargs):
        """Initialize Layer."""
        super(EquivariantInitialize, self).__init__(**kwargs)
        self.dim = int(dim)
        self.method = str(method)
        self.value = float(value)
        self.stddev = float(stddev)

    def build(self, input_shape):
        """Build layer."""
        super(EquivariantInitialize, self).build(input_shape)
        assert len(input_shape) >= 3, "ERROR:kgcnn: Need input shape of form (batch, None, F_dim)."

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: nodes

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Equivariant tensor of shape (batch, [N], dim, F)
        """
        inputs = self.assert_ragged_input_rank(inputs)
        if self.method == "zeros":
            out = tf.zeros_like(inputs.values)
            out = tf.expand_dims(out, axis=1)
            out = tf.repeat(out, self.dim, axis=1)
        elif self.method == "eps":
            out = tf.zeros_like(inputs.values) + ks.backend.epsilon()
            out = tf.expand_dims(out, axis=1)
            out = tf.repeat(out, self.dim, axis=1)
        elif self.method == "ones":
            out = tf.ones_like(inputs.values)
            out = tf.expand_dims(out, axis=1)
            out = tf.repeat(out, self.dim, axis=1)
        elif self.method == "eye":
            values = inputs.values
            out = tf.eye(self.dim, num_columns=values.shape[1], batch_shape=tf.shape(values)[:1], dtype=values.dtype)
        elif self.method == "normal":
            values = inputs.values
            out = tf.expand_dims(tf.random.normal([self.dim, values.shape[1]], stddev=self.stddev), axis=0)
            out = tf.repeat(out, tf.shape(values)[0], axis=0)
        elif self.method == "const":
            out = tf.ones_like(inputs.values)*self.value
            out = tf.expand_dims(out, axis=1)
            out = tf.repeat(out, self.dim, axis=1)
        elif self.method == "node":
            out = tf.expand_dims(inputs.values, axis=1)
            out = tf.repeat(out, self.dim, axis=1)
        else:
            raise ValueError("Unknown initialization method %s" % self.method)
        # Static shape expansion for dim, tf.repeat would be possible too.
        out = tf.RaggedTensor.from_row_splits(out, inputs.row_splits)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(EquivariantInitialize, self).get_config()
        config.update({"dim": self.dim, "method": self.method, "value": self.value, "stddev": self.stddev})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SplitEmbedding')
class SplitEmbedding(GraphBaseLayer):
    """Split embeddings of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ .

    Args:
        num_or_size_splits: Number or size of splits.
        axis (int): Axis to split.
        num (int): Number of output splits.
    """

    def __init__(self,
                 num_or_size_splits,
                 axis=-1,
                 num=None,
                 **kwargs):
        super(SplitEmbedding, self).__init__(**kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.out_num = num

    def build(self, input_shape):
        super(SplitEmbedding, self).build(input_shape)
        # If rank is not defined can't call on values if axis does not happen to be positive.
        self.axis = get_positive_axis(self.axis, len(input_shape))
        if self.axis <= 1:
            raise ValueError("Can not split tensor at axis <= 1.")

    def call(self, inputs, **kwargs):
        r"""Forward pass: Split embeddings across feature dimension e.g. `axis=-1` .

        Args:
            inputs (tf.RaggedTensor): Embeddings of shape (batch, [N], F)

        Returns:
            list: List of tensor splits of shape (batch, [N], F/num)
        """
        self.assert_ragged_input_rank(inputs, ragged_rank=1)
        # Axis will be positive and >=1 from built!
        # Axis for values is axis-1.
        out_tensors = tf.split(inputs.values, self.num_or_size_splits, axis=self.axis-1, num=self.out_num)
        return [tf.RaggedTensor.from_row_splits(x, inputs.row_splits, validate=self.ragged_validate) for x in
                out_tensors]

    def get_config(self):
        config = super(SplitEmbedding, self).get_config()
        config.update({"num_or_size_splits": self.num_or_size_splits, "axis": self.axis, "num": self.out_num})
        return config
