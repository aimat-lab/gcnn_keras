import keras as ks
from keras import ops
from kgcnn.layers.aggr import AggregateLocalEdges
from keras.layers import Add, Multiply, Dense, Concatenate
from kgcnn.layers.geom import EuclideanNorm, ScalarProduct
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.modules import ExpandDims


class PAiNNconv(ks.layers.Layer):
    """Continuous filter convolution block of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`__ .

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
                 conv_pool='scatter_sum',
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
        self.lay_mult = Multiply()
        if self.cutoff is not None:
            self.lay_mult_cutoff = Multiply()
        self.lay_exp_vv = ExpandDims(axis=-2)
        self.lay_exp_vw = ExpandDims(axis=-2)
        self.lay_exp_r = ExpandDims(axis=-1)
        self.lay_mult_vv = Multiply()
        self.lay_mult_vw = Multiply()

        self.lay_add = Add()

    def build(self, input_shape):
        """Build layer."""
        super(PAiNNconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: [nodes, equivariant, rbf, envelope, r_ij, edge_index]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - equivariant (Tensor): Equivariant node embedding of shape ([N], 3, F)
                - rdf (Tensor): Radial basis expansion pair-wise distance of shape ([M], #Basis)
                - envelope (Tensor): Distance envelope of shape ([N], 1)
                - r_ij (Tensor): Normalized pair-wise distance of shape ([M], 3)
                - edge_index (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            tuple: [ds, dv]

                - ds (Tensor) Updated node features of shape ([N], F)
                - dv (Tensor) Updated equivariant features of shape ([N], F, 3)
        """
        node, equivariant, rbf, envelope, r_ij, indexlist = inputs
        s = self.lay_dense1(node)
        s = self.lay_phi(s)
        s = self.gather_n([s, indexlist])
        w = self.lay_w(rbf)
        if self.cutoff is not None:
            w = self.lay_mult_cutoff([w, envelope])
        sw = self.lay_mult([s, w])
        sw1, sw2, sw3 = self.lay_split(sw, **kwargs)
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
            if x in config_dense:
                config.update({x: config_dense[x]})
        return config


class PAiNNUpdate(ks.layers.Layer):
    """Continuous filter convolution of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`__ .

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

        self.lay_scalar_prod = ScalarProduct(axis=1)
        self.lay_norm = EuclideanNorm(axis=1, add_eps=self.add_eps)
        self.lay_concat = Concatenate(axis=-1)
        self.lay_split = SplitEmbedding(3, axis=-1)

        self.lay_mult = Multiply()
        self.lay_exp_v = ExpandDims(axis=-2)
        self.lay_mult_vv = Multiply()
        self.lay_add = Add()

    def build(self, input_shape):
        """Build layer."""
        super(PAiNNUpdate, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: [nodes, equivariant]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - equivariant (Tensor): Equivariant node embedding of shape ([N], 3, F)

        Returns:
            tuple: [ds, dv]

                - ds (Tensor) Updated node features of shape ([N], F)
                - dv (Tensor) Updated equivariant features of shape ([N], 3, F)
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
            if x in config_dense.keys():
                config.update({x: config_dense[x]})
        return config


class EquivariantInitialize(ks.layers.Layer):
    """Equivariant initializer of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`__ .

    Args:
        dim (int): Dimension of equivariant features. Default is 3.
        method (str): How to initialize equivariant tensor. Default is "zeros".
    """

    def __init__(self, dim=3, units=128, method: str = "zeros", value: float = 1.0, stddev: float = 1.0,
                 seed: int = 42, **kwargs):
        """Initialize Layer."""
        super(EquivariantInitialize, self).__init__(**kwargs)
        self.dim = int(dim)
        self.units = units
        self.method = str(method)
        self.value = float(value)
        self.stddev = float(stddev)
        self.seed = seed

    def build(self, input_shape):
        """Build layer."""
        super(EquivariantInitialize, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: nodes

                - nodes (Tensor): Node embeddings of shape ([N], F) or ([N], ).

        Returns:
            Tensor: Equivariant tensor of shape ([N], dim, F) or ([N], dim, units).
        """
        if len(ops.shape(inputs)) < 2:
            inputs = ops.expand_dims(inputs, axis=-1)
            inputs = ops.repeat(inputs, self.units, axis=-1)

        if self.method == "zeros":
            out = ops.zeros_like(inputs, dtype=self.dtype)
            out = ops.expand_dims(out, axis=1)
            out = ops.repeat(out, self.dim, axis=1)
        elif self.method == "eps":
            out = ops.zeros_like(inputs, dtype=self.dtype) + ks.backend.epsilon()
            out = ops.expand_dims(out, axis=1)
            out = ops.repeat(out, self.dim, axis=1)
        elif self.method == "normal":
            out = ks.random.normal((self.dim, ops.shape(inputs)[-1]), seed=self.seed)
            out = ops.expand_dims(out, axis=0)
            out = ops.repeat(out, ops.shape(inputs)[0], axis=0)
        elif self.method == "ones":
            out = ops.ones_like(inputs, dtype=self.dtype)
            out = ops.expand_dims(out, axis=1)
            out = ops.repeat(out, self.dim, axis=1)
        elif self.method == "eye":
            out = ops.eye(self.dim, ops.shape(inputs)[1], dtype=self.dtype)
            out = ops.expand_dims(out, axis=0)
            out = ops.repeat(out, ops.shape(inputs)[0], axis=0)
        elif self.method == "const":
            out = ops.ones_like(inputs, dtype=self.dtype)*self.value
            out = ops.expand_dims(out, axis=1)
            out = ops.repeat(out, self.dim, axis=1)
        elif self.method == "node":
            out = ops.expand_dims(inputs, axis=1)
            out = ops.repeat(out, self.dim, axis=1)
            out = ops.cast(out, dtype=self.dtype)
        else:
            raise ValueError("Unknown initialization method %s" % self.method)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(EquivariantInitialize, self).get_config()
        config.update({"dim": self.dim, "method": self.method, "value": self.value, "stddev": self.stddev,
                       "units": self.units, "seed": self.seed})
        return config


class SplitEmbedding(ks.layers.Layer):
    """Split embeddings of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`__ .

    Args:
        num_or_size_splits: Number or size of splits.
        axis (int): Axis to split.
        num (int): Number of output splits.
    """

    def __init__(self,
                 indices_or_sections,
                 axis=-1,
                 **kwargs):
        super(SplitEmbedding, self).__init__(**kwargs)
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def build(self, input_shape):
        super(SplitEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass: Split embeddings across feature dimension e.g. `axis=-1` .

        Args:
            inputs (Tensor): Embeddings of shape ([N], F)

        Returns:
            list: List of tensor splits of shape ([N], F/num)
        """
        outs = ops.split(inputs, self.indices_or_sections, axis=self.axis)
        return outs

    def get_config(self):
        config = super(SplitEmbedding, self).get_config()
        config.update({"indices_or_sections": self.indices_or_sections, "axis": self.axis})
        return config
