import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.axis import get_positive_axis

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.pool.pooling import PoolingLocalEdges
from kgcnn.layers.keras import Add, Multiply, Dense, ExpandDims, Concatenate
from kgcnn.layers.geom import CosCutOff, EuclideanNorm, ScalarProduct
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.embedding import SplitEmbedding


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
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)
        self.lay_phi = Dense(units=self.units*3, activation='linear', use_bias=self.use_bias, **kernel_args,
                             **self._kgcnn_info)
        self.lay_w = Dense(units=self.units*3, activation='linear', use_bias=self.use_bias, **kernel_args,
                           **self._kgcnn_info)

        self.lay_split = SplitEmbedding(3, axis=-1)
        if self.cutoff is not None:
            self.lay_cos_cut = CosCutOff(cutoff=self.cutoff, **self._kgcnn_info)

        self.lay_sum = PoolingLocalEdges(pooling_method=conv_pool, **self._kgcnn_info)
        self.lay_sum_v = PoolingLocalEdges(pooling_method=conv_pool, **self._kgcnn_info)

        self.gather_n = GatherNodesOutgoing(**self._kgcnn_info)
        self.gather_v = GatherNodesOutgoing(**self._kgcnn_info)

        self.lay_mult = Multiply(**self._kgcnn_info)
        self.lay_exp_vv = ExpandDims(axis=-1, **self._kgcnn_info)
        self.lay_exp_vw = ExpandDims(axis=-1, **self._kgcnn_info)
        self.lay_exp_r = ExpandDims(axis=-2, **self._kgcnn_info)
        self.lay_mult_vv = Multiply(**self._kgcnn_info)
        self.lay_mult_vw = Multiply(**self._kgcnn_info)

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
        if self.cutoff is not None:
            w = self.lay_cos_cut(w)  # Cos-cutoff
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


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='TrafoEquivariant')
class TrafoEquivariant(GraphBaseLayer):
    """Linear Combination of equivariant features.
    Used by PAiNN.

    """

    def __init__(self,
                 units,
                 axis=-2,
                 activation="linear",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize layer same as tf.keras.Multiply."""
        super(TrafoEquivariant, self).__init__(**kwargs)
        self.axis = axis
        self._kgcnn_wrapper_args = ["units", "activation", "use_bias", "kernel_initializer", "bias_initializer",
                                    "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                                    "kernel_constraint", "bias_constraint"]
        self._kgcnn_wrapper_layer = ks.layers.Dense(units=units, activation=activation,
                                                    use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                    bias_initializer=bias_initializer,
                                                    kernel_regularizer=kernel_regularizer,
                                                    bias_regularizer=bias_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    kernel_constraint=kernel_constraint,
                                                    bias_constraint=bias_constraint)

    def build(self, input_shape):
        super(TrafoEquivariant, self).build(input_shape)
        self.axis = get_positive_axis(self.axis, len(input_shape))
        mul_dim = input_shape[self.axis]

        if mul_dim is None:
            raise ValueError(
                'The axis %s of the inputs to `TrafoEquivariant` should be defined. Found `None`.' % self.axis)

        if self.axis <= 1:
            raise ValueError('The axis argument to `TrafoEquivariant` must be >1. Found `%s`.' % self.axis)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: equivariant

                - equivariant (tf.RaggedTensor): Equivariant embedding of shape (batch, [N], F, D)

        Returns:
           tf.RaggedTensor: Linear transformation of shape (batch, [N], F', D)
        """
        # Require RaggedTensor with ragged_rank=1 as inputs.
        # tf.transpose with perm argument does not allow ragged input in tf.__version__='2.5.0'.
        assert isinstance(inputs, tf.RaggedTensor) and inputs.ragged_rank == 1
        values = inputs.values

        # Find axis to apply linear transformation.
        axis = self.axis - 1  # is positive axis from build.
        if axis == values.shape.rank - 1:
            values = self._kgcnn_wrapper_layer(values)
        else:
            # Permute axis to last dimension for dense.
            perm_order = [i for i in range(values.shape.rank)]
            perm_order[axis] = values.shape.rank - 1
            perm_order[-1] = axis

            values = tf.transpose(values, perm=perm_order)
            values = self._kgcnn_wrapper_layer(values)
            # Swap axes back.
            values = tf.transpose(values, perm=perm_order)

        out = tf.RaggedTensor.from_row_splits(values, inputs.row_splits, validate=self.ragged_validate)
        return out


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='PAiNNUpdate')
class PAiNNUpdate(GraphBaseLayer):
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
        super(PAiNNUpdate, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)
        self.lay_lin_u = TrafoEquivariant(self.units, axis=-2, activation='linear', use_bias=False, **kernel_args,
                                          **self._kgcnn_info)
        self.lay_lin_v = TrafoEquivariant(self.units, axis=-2, activation='linear', use_bias=False, **kernel_args,
                                          **self._kgcnn_info)
        self.lay_a = Dense(units=self.units * 3, activation='linear', use_bias=self.use_bias, **kernel_args,
                           **self._kgcnn_info)

        self.lay_scalar_prod = ScalarProduct()
        self.lay_norm = EuclideanNorm()
        self.lay_concat = Concatenate(axis=-1)
        self.lay_split = SplitEmbedding(3, axis=-1)

        self.lay_mult = Multiply(**self._kgcnn_info)
        self.lay_exp_v = ExpandDims(axis=-1, **self._kgcnn_info)
        self.lay_mult_vv = Multiply(**self._kgcnn_info)
        self.lay_add = Add(**self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(PAiNNUpdate, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: [nodes, equivariant]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - equivariant (tf.RaggedTensor): Equivariant node embedding of shape (batch, [N], F, 3)

        Returns:
            tuple: [ds, dv]

                - ds (tf.RaggedTensor) Updated node features of shape (batch, [N], F)
                - dv (tf.RaggedTensor) Updated equivariant features of shape (batch, [N], F, 3)
        """
        node, equivariant = inputs
        v_v = self.lay_lin_v(equivariant)
        v_u = self.lay_lin_u(equivariant)
        v_prod = self.lay_scalar_prod([v_u, v_v])
        v_norm = self.lay_norm(v_v)
        a = self.lay_concat([node, v_norm])
        a = self.lay_dense1(a)
        a = self.lay_a(a)
        a_vv, a_sv, a_ss = self.lay_split(a)
        a_vv = self.lay_exp_v(a_vv)
        dv = self.lay_mult_vv([a_vv, v_u])
        ds = self.lay_mult([v_prod, a_sv])
        ds = self.lay_add([ds, a_ss])
        return ds, dv

    def get_config(self):
        """Update layer config."""
        config = super(PAiNNUpdate, self).get_config()
        config.update({"units": self.units})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EquivariantInitialize')
class EquivariantInitialize(GraphBaseLayer):
    """Zero equivariant initializer.

    Args:
        dim (int): Dimension of equivariant features. Default is 3.
    """

    def __init__(self, dim=3, **kwargs):
        """Initialize Layer."""
        super(EquivariantInitialize, self).__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        """Build layer."""
        super(EquivariantInitialize, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs: nodes

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Zero equivariant tensor of shape (batch, [N], F, dim)
        """
        if isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                values, part = inputs.values, inputs.row_splits
                equiv = tf.expand_dims(tf.zeros_like(values), axis=-1)*tf.zeros([1]*(inputs.shape.rank-1)+[self.dim])
                return tf.RaggedTensor.from_row_splits(equiv, part, validate=self.ragged_validate)
        return tf.expand_dims(tf.zeros_like(inputs), axis=-1)*tf.zeros([1]*inputs.shape.rank+[self.dim])

    def get_config(self):
        """Update layer config."""
        config = super(EquivariantInitialize, self).get_config()
        config.update({"dim": self.dim})
        return config
