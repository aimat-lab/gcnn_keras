import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.axis import get_positive_axis

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.keras import Add, Multiply, Dense, Concatenate, ExpandDims
from kgcnn.layers.geom import EuclideanNorm, ScalarProduct
from kgcnn.layers.gather import GatherNodesOutgoing


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
        self.lay_sum = PoolingLocalEdges(pooling_method=conv_pool)
        self.lay_sum_v = PoolingLocalEdges(pooling_method=conv_pool)
        self.gather_n = GatherNodesOutgoing()
        self.gather_v = GatherNodesOutgoing()
        self.lay_mult = Multiply()
        self.lay_mult_cutoff = Multiply()
        self.lay_exp_vv = ExpandDims(axis=-1)
        self.lay_exp_vw = ExpandDims(axis=-1)
        self.lay_exp_r = ExpandDims(axis=-2)
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

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - equivariant (tf.RaggedTensor): Equivariant node embedding of shape (batch, [N], F, 3)
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


# @tf.keras.utils.register_keras_serializable(package='kgcnn', name='TrafoEquivariant')
# class TrafoEquivariant(GraphBaseLayer):
#     """Linear Combination of equivariant features as defined by `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ .
#
#     Args:
#         units (int): Units for kernel.
#         axis (int): Axis to perform linear transformation. Must be >=2. Default is 2.
#         use_bias (bool): Use bias. Default is True.
#         activation (str): Activation function. Default is 'kgcnn>shifted_softplus'.
#         kernel_regularizer: Kernel regularization. Default is None.
#         bias_regularizer: Bias regularization. Default is None.
#         activity_regularizer: Activity regularization. Default is None.
#         kernel_constraint: Kernel constrains. Default is None.
#         bias_constraint: Bias constrains. Default is None.
#         kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
#         bias_initializer: Initializer for bias. Default is 'zeros'.
#     """
#
#     def __init__(self,
#                  units,
#                  axis=2,
#                  activation="linear",
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         """Initialize layer same as tf.keras.Multiply."""
#         super(TrafoEquivariant, self).__init__(**kwargs)
#         self.axis = axis
#         self.units = int(units) if not isinstance(units, int) else units
#         self.use_bias = use_bias
#         self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
#         self.bias_initializer = tf.keras.initializers.get(bias_initializer)
#         self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
#         self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
#         self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
#         self.bias_constraint = tf.keras.constraints.get(bias_constraint)
#
#         self.layer_act = ks.layers.Activation(activation=activation, activity_regularizer=activity_regularizer)
#         self.kernel = None
#         self.bias = None
#
#     def build(self, input_shape):
#         super(TrafoEquivariant, self).build(input_shape)
#         self.axis = get_positive_axis(self.axis, len(input_shape))
#         trafo_dim = input_shape[self.axis]
#
#         if trafo_dim is None:
#             raise ValueError(
#                 'ERROR:kgcnn: The axis %s of the inputs to `TrafoEquivariant` must not be None.' % self.axis)
#
#         if self.axis <= 1:
#             raise ValueError(
#                 'ERROR:kgcnn: The axis argument to `TrafoEquivariant` must be >1. Found `%s`.' % self.axis)
#
#         kernel_shape = [1] * (len(input_shape) + 1)
#         kernel_shape[self.axis] = trafo_dim
#         kernel_shape[self.axis+1] = self.units
#         self.kernel = self.add_weight('kernel', shape=kernel_shape,
#                                       initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint, dtype=self.dtype, trainable=True)
#
#         bias_shape = [1] * len(input_shape)
#         bias_shape[self.axis] = self.units
#         if self.use_bias:
#             self.bias = self.add_weight('bias', shape=bias_shape,
#                                         initializer=self.bias_initializer, regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint,
#                                         dtype=self.dtype, trainable=True)
#         else:
#             self.bias = None
#
#     def call(self, inputs, **kwargs):
#         """Forward pass.
#
#         Args:
#             inputs: equivariant
#
#                 - equivariant (tf.RaggedTensor): Equivariant embedding of shape (batch, [N], F, D)
#
#         Returns:
#            tf.RaggedTensor: Linear transformation of shape (batch, [N], F', D)
#         """
#         # Require RaggedTensor with ragged_rank=1 as inputs.
#         # tf.transpose with perm argument does not allow ragged input in tf.__version__='2.5.0'.
#         # Also in tf.__version__='2.6.0' there seem to be a problem with transpose causing NaNs.
#         assert isinstance(inputs, tf.RaggedTensor), "ERROR:kgcnn: Require ragged input."
#         assert inputs.ragged_rank == 1, "ERROR:kgcnn: Require ragged input."
#         values = inputs.values
#
#         values = tf.expand_dims(values, axis=self.axis)*self.kernel[0]
#         values = tf.reduce_sum(values, axis=self.axis-1)
#         if self.use_bias:
#             values = values + self.bias[0]
#         values = self.layer_act(values, **kwargs)
#
#         out = tf.RaggedTensor.from_row_splits(values, inputs.row_splits, validate=self.ragged_validate)
#         return out
#
#     def get_config(self):
#         config = super(TrafoEquivariant, self).get_config()
#         config.update({
#             'units': self.units,
#             'use_bias': self.use_bias,
#             'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
#             'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
#             'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
#             'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint)
#         })
#         config_act = self.layer_act.get_config()
#         for x in ["activity_regularizer", "activation"]:
#             config.update({x: config_act[x]})
#         config.update({"axis": self.axis})
#         return config


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
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_lin_u = TrafoEquivariant(self.units, axis=-2, activation='linear', use_bias=False, **kernel_args)
        self.lay_lin_v = TrafoEquivariant(self.units, axis=-2, activation='linear', use_bias=False, **kernel_args)
        self.lay_a = Dense(units=self.units * 3, activation='linear', use_bias=self.use_bias, **kernel_args)

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
        config.update({"units": self.units})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EquivariantInitialize')
class EquivariantInitialize(GraphBaseLayer):
    """Zero equivariant initializer of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ .

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
                equiv = tf.expand_dims(tf.zeros_like(values), axis=-1) * tf.zeros(
                    [1] * (inputs.shape.rank - 1) + [self.dim])
                return tf.RaggedTensor.from_row_splits(equiv, part, validate=self.ragged_validate)
        return tf.expand_dims(tf.zeros_like(inputs), axis=-1) * tf.zeros([1] * inputs.shape.rank + [self.dim])

    def get_config(self):
        """Update layer config."""
        config = super(EquivariantInitialize, self).get_config()
        config.update({"dim": self.dim})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SplitEmbedding')
class SplitEmbedding(GraphBaseLayer):
    """Split embeddings of `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ .

    Args:
        num_or_size_splits: Number or size of splits.
        axis (int): Axis to split.
        num (int): Number the number of output splits.
    """

    def __init__(self,
                 num_or_size_splits,
                 axis=-1,
                 num=None,
                 **kwargs):
        super(SplitEmbedding, self).__init__(**kwargs)
        # self._supports_ragged_inputs = True
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.out_num = num

    def call(self, inputs, **kwargs):
        """Split embeddings across feature dimension e.g. `axis=-1`."""
        if isinstance(inputs, tf.RaggedTensor):
            if self.axis == -1 and inputs.shape[-1] is not None and inputs.ragged_rank == 1:
                value_tensor = inputs.values  # will be Tensor
                out_tensor = tf.split(value_tensor, self.num_or_size_splits, axis=self.axis, num=self.out_num)
                return [tf.RaggedTensor.from_row_splits(x, inputs.row_splits, validate=self.ragged_validate) for x in
                        out_tensor]
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting tf.split... ")

        out = tf.split(inputs, self.num_or_size_splits, axis=self.axis, num=self.out_num)
        return out

    def get_config(self):
        config = super(SplitEmbedding, self).get_config()
        config.update({"num_or_size_splits": self.num_or_size_splits, "axis": self.axis, "num": self.out_num})
        return config
