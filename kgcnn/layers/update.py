import keras as ks
from keras.layers import Dense, Add, Layer


class GRUUpdate(Layer):
    r"""Gated recurrent unit for updating node or edge embeddings.
    As proposed by `NMPNN <http://arxiv.org/abs/1704.01212>`__ .
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
        r"""Initialize layer.

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
        super(GRUUpdate, self).__init__(**kwargs)
        self.units = units

        self.gru_cell = ks.layers.GRUCell(
            units=units,
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
            recurrent_dropout=recurrent_dropout, reset_after=reset_after
        )

    def build(self, input_shape):
        """Build layer."""
        super(GRUUpdate, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, updates]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - updates (Tensor): Matching node updates of shape ([N], F)
            mask: Mask for inputs. Default is None.

        Returns:
           Tensor: Updated nodes of shape ([N], F)
        """
        n, eu = inputs
        out, _ = self.gru_cell(eu, n, **kwargs)
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
            if x in conf_cell.keys():
                config.update({x: conf_cell[x]})
        return config


class ResidualLayer(Layer):
    r"""Residual Layer as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`__ ."""

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
        """Initialize layer.

        Args:
            units: Dimension of the kernel.
            use_bias (bool, optional): Use bias. Defaults to True.
            activation (str): Activation function. Default is "kgcnn>swish".
            kernel_regularizer: Kernel regularization. Default is None.
            bias_regularizer: Bias regularization. Default is None.
            activity_regularizer: Activity regularization. Default is None.
            kernel_constraint: Kernel constrains. Default is None.
            bias_constraint: Bias constrains. Default is None.
            kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
            bias_initializer: Initializer for bias. Default is 'zeros'.
        """
        super(ResidualLayer, self).__init__(**kwargs)
        dense_args = {
            "units": units, "activation": activation, "use_bias": use_bias,
            "kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
            "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
            "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer
        }

        self.dense_1 = Dense(**dense_args)
        self.dense_2 = Dense(**dense_args)
        self.add_end = Add()

    def build(self, input_shape):
        """Build layer."""
        super(ResidualLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (Tensor): Node or edge embedding of shape ([N], F)

        Returns:
            Tensor: Node or edge embedding of shape ([N], F)
        """
        x = self.dense_1(inputs, **kwargs)
        x = self.dense_2(x, **kwargs)
        x = self.add_end([inputs, x], **kwargs)
        return x

    def get_config(self):
        config = super(ResidualLayer, self).get_config()
        conf_dense = self.dense_1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias", "units"]:
            if x in conf_dense.keys():
                config.update({x: conf_dense[x]})
        return config
