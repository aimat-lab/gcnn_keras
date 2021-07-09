import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.embedding import SplitEmbedding
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.keras import Dense, Multiply, Add, Concatenate
from kgcnn.layers.geom import EuclideanNorm, ScalarProduct


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='TrafoMatMulMessages')
class TrafoMatMulMessages(GraphBaseLayer):
    """Apply message by edge matrix multiplication.
    
    The message dimension must be suitable for matrix multiplication.
    
    Args:
        target_shape (int): Target dimension. Message dimension must match target_dim*node_dim.
    """

    def __init__(self, target_shape, **kwargs):
        """Initialize layer."""
        super(TrafoMatMulMessages, self).__init__(**kwargs)
        self.target_shape = target_shape

    def build(self, input_shape):
        """Build layer."""
        super(TrafoMatMulMessages, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [trafo, edges]

                - trafo (tf.RaggedTensor): Transformation by matrix multiplication for each message.
                  Must be reshaped to (batch, [M], FxF).
                - edges (tf.RaggedTensor): Edge embeddings or messages (batch, [M], F)
            
        Returns:
            tf.RaggedTensor: Transformation of messages by matrix multiplication of shape (batch, [M], F)
        """
        dyn_inputs = inputs
        # We cast to values here
        dens_trafo, trafo_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        dens_e, epart = dyn_inputs[1].values, dyn_inputs[1].row_splits

        dens_m = tf.reshape(dens_trafo,
                            (ks.backend.shape(dens_trafo)[0], self.target_shape, ks.backend.shape(dens_e)[-1]))
        out = tf.keras.backend.batch_dot(dens_m, dens_e)

        out = tf.RaggedTensor.from_row_splits(out, epart, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(TrafoMatMulMessages, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GRUUpdate')
class GRUUpdate(GraphBaseLayer):
    """Gated recurrent unit update.

    Args:
        units (int): Units for GRU.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent state.
            Default: `orthogonal`.
        bias_initializer: Initializer for the bias vector. Default: `zeros`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector. Default:
            `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel`
            weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector. Default:
            `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
            the linear transformation of the recurrent state. Default: 0.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before",
            True = "after" (default and CuDNN compatible).
    """

    def __init__(self, units,
                 activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros', kernel_regularizer=None,
                 recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                 recurrent_dropout=0.0, reset_after=True,
                 **kwargs):
        """Initialize layer."""
        super(GRUUpdate, self).__init__(**kwargs)
        self.units = units

        self.gru_cell = tf.keras.layers.GRUCell(units=units,
                                                activation=activation, recurrent_activation=recurrent_activation,
                                                use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                recurrent_initializer=recurrent_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                recurrent_regularizer=recurrent_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                recurrent_constraint=recurrent_constraint,
                                                bias_constraint=bias_constraint,
                                                dropout=dropout,
                                                recurrent_dropout=recurrent_dropout, reset_after=reset_after)

    def build(self, input_shape):
        """Build layer."""
        # self.gru.build(channels)
        super(GRUUpdate, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, updates]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - updates (tf.RaggedTensor): Matching node updates of shape (batch, [N], F)

        Returns:
           tf.RaggedTensor: Updated nodes of shape (batch, [N], F)
        """
        dyn_inputs = inputs
        # We cast to values here
        n, npart = dyn_inputs[0].values, dyn_inputs[0].row_splits
        eu, _ = dyn_inputs[1].values, dyn_inputs[1].row_splits

        out, _ = self.gru_cell(eu, n, **kwargs)

        out = tf.RaggedTensor.from_row_splits(out, npart, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(GRUUpdate, self).get_config()
        conf_cell = self.gru_cell.get_config()
        param_list = ["units", "activation", "recurrent_activation",
                      "use_bias", "kernel_initializer",
                      "recurrent_initializer",
                      "bias_initializer", "kernel_regularizer",
                      "recurrent_regularizer", "bias_regularizer", "kernel_constraint",
                      "recurrent_constraint", "bias_constraint", "dropout",
                      "recurrent_dropout", "reset_after"]
        for x in param_list:
            config.update({x: conf_cell[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MultiplyEquivariant')
class MultiplyEquivariant(GraphBaseLayer):
    """Multiplication of equivariant and scalar features with pre-defined broadcasting.
    Used by PAiNN. Require ragged_rank=1 and rank=3 or rank=4.

    """

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Multiply."""
        super(MultiplyEquivariant, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [scalar, equivariant]

                - scalar (tf.RaggedTensor): Scalar embeddings of shape (batch, [N], F)
                    or shape (batch, [N], F, D)
                - equivariant (tf.RaggedTensor): Geometric embedding (batch, [N], D)
                    or shape (batch, [N], F, D)

        Returns:
           tf.RaggedTensor: Multiplication of shape (batch, [N], F, D)
        """
        s, v = inputs
        assert all([isinstance(x, tf.RaggedTensor) for x in inputs]) and all([x.ragged_rank == 1 for x in inputs])
        sval = s.values
        vval = v.values
        if len(s.shape) == 3:
            sval = tf.expand_dims(sval, axis=-1)
        if len(v.shape) == 3:
            vval = tf.expand_dims(vval, axis=-2)
        out = sval*vval
        out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
        return out


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='TrafoEquivariant')
class TrafoEquivariant(GraphBaseLayer):
    """Linear Combination of equivariant features.
    Used by PAiNN. Require ragged_rank=1 and rank=4.
    TODO: Will remove this later and replace by a more general version.

    """

    def __init__(self,
                 units,
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

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: equivariant

                - equivariant (tf.RaggedTensor): Equivariant embedding of shape (batch, [N], F, D)

        Returns:
           tf.RaggedTensor: Multiplication of shape (batch, [N], F', D)
        """
        assert isinstance(inputs, tf.RaggedTensor) and inputs.ragged_rank == 1
        vals = inputs.values
        vals = tf.transpose(vals, perm=[0,2,1])
        vals = self._kgcnn_wrapper_layer(vals)
        vals = tf.transpose(vals, perm=[0,2,1])
        out = tf.RaggedTensor.from_row_splits(vals, inputs.row_splits, validate=self.ragged_validate)
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
        self.lay_lin_u = TrafoEquivariant(self.units, activation='linear', use_bias=False, **kernel_args,
                                          **self._kgcnn_info)
        self.lay_lin_v = TrafoEquivariant(self.units, activation='linear', use_bias=False, **kernel_args,
                                          **self._kgcnn_info)
        self.lay_a = Dense(units=self.units*3, activation='linear', use_bias=self.use_bias, **kernel_args,
                                **self._kgcnn_info)

        self.lay_scalar_prod = ScalarProduct()
        self.lay_norm = EuclideanNorm()
        self.lay_concat = Concatenate(axis=-1)
        self.lay_split = SplitEmbedding(3, axis=-1)

        self.lay_mult = Multiply(**self._kgcnn_info)
        self.lay_mult_vv = MultiplyEquivariant(**self._kgcnn_info)
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

