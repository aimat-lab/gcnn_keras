import tensorflow as tf

from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.norm import GraphBatchNormalization, GraphLayerNormalization
from kgcnn.layers.base import GraphBaseLayer
import kgcnn.ops.activ


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MLPBase')
class MLPBase(GraphBaseLayer):
    r"""Multilayer perceptron that consist of N dense keras layers. Supply list in place of arguments for each layer.
    If not list, then the single argument is used for each layer.
    The number of layers is given by units, which should be list.
    Additionally this base class holds arguments for batch-normalization which should be applied between kernel
    and activation. And dropout after the kernel output and before normalization.
    This base class does not initialize any sub-layers or implements :obj:`call()`. Only for managing arguments.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        use_normalization: Whether to use a normalization layer in between.
        normalization_technique: Which keras normalization technique to apply.
            This can be either 'batch', 'layer', 'group' etc.
        axis: Integer, the axis that should be normalized (typically the features
            axis). For instance, after a `Conv2D` layer with
            `data_format="channels_first"`, set `axis=1` in `GraphBatchNormalization`.
        momentum: Momentum for the moving average.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor. If False, `beta`
            is ignored.
        scale: If True, multiply by `gamma`. If False, `gamma` is not used. When the
            next layer is linear (also e.g. `nn.relu`), this can be disabled since the
            scaling will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        use_dropout: Whether to use dropout layers in between.
        rate: Float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape`(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 activation=None,
                 activity_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_constraint=None,
                 bias_constraint=None,
                 # Normalization
                 use_normalization=False,
                 normalization_technique="batch",
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 # Dropout
                 use_dropout=False,
                 rate=None, noise_shape=None, seed=None,
                 **kwargs):
        """Initialize MLP as for dense."""
        super(MLPBase, self).__init__(**kwargs)
        # everything should be defined by units.
        if isinstance(units, int):
            units = [units]
        if not isinstance(units, list):
            raise ValueError("Units must be a list or a single int for `MLP`.")

        self._depth = len(units)

        # Assert matching number of args
        def assert_args_is_list(args):
            if not isinstance(args, (list, tuple)):
                return [args for _ in range(self._depth)]
            return args

        # Dense
        use_bias = assert_args_is_list(use_bias)
        activation = assert_args_is_list(activation)
        kernel_regularizer = assert_args_is_list(kernel_regularizer)
        bias_regularizer = assert_args_is_list(bias_regularizer)
        activity_regularizer = assert_args_is_list(activity_regularizer)
        kernel_initializer = assert_args_is_list(kernel_initializer)
        bias_initializer = assert_args_is_list(bias_initializer)
        kernel_constraint = assert_args_is_list(kernel_constraint)
        bias_constraint = assert_args_is_list(bias_constraint)
        # Normalization
        use_normalization = assert_args_is_list(use_normalization)
        normalization_technique = assert_args_is_list(normalization_technique)
        if not isinstance(axis, list):  # Special case, if axis is supposed to be multiple axis, use tuple here.
            axis = [axis for _ in units]
        momentum = assert_args_is_list(momentum)
        epsilon = assert_args_is_list(epsilon)
        center = assert_args_is_list(center)
        scale = assert_args_is_list(scale)
        beta_initializer = assert_args_is_list(beta_initializer)
        gamma_initializer = assert_args_is_list(gamma_initializer)
        moving_mean_initializer = assert_args_is_list(moving_mean_initializer)
        moving_variance_initializer = assert_args_is_list(moving_variance_initializer)
        beta_regularizer = assert_args_is_list(beta_regularizer)
        gamma_regularizer = assert_args_is_list(gamma_regularizer)
        beta_constraint = assert_args_is_list(beta_constraint)
        gamma_constraint = assert_args_is_list(gamma_constraint)
        # Dropout
        use_dropout = assert_args_is_list(use_dropout)
        rate = assert_args_is_list(rate)
        seed = assert_args_is_list(seed)
        if not isinstance(noise_shape, list):  # Special case, for shape, use tuple here.
            noise_shape = [noise_shape for _ in units]

        for x in [activation, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_initializer,
                  bias_initializer, kernel_constraint, bias_constraint, use_bias, axis, momentum, epsilon,
                  center, scale, beta_initializer, gamma_initializer, moving_mean_initializer,
                  moving_variance_initializer, beta_regularizer, gamma_regularizer, beta_constraint,
                  gamma_constraint, rate, seed, noise_shape, use_dropout, use_normalization, normalization_technique]:
            if len(x) != len(units):
                raise ValueError("Error: Provide matching list of units", units, "and", x, "or simply a single value.")

        for x in normalization_technique:
            if x not in ["batch", "layer", "group", "instance"]:
                raise ValueError("ERROR Unknown normalization method, choose: batch, layer, group, instance, ...")

        # Deserialized args
        self._conf_units = list(units)
        self._conf_use_bias = list(use_bias)
        self._conf_activation = list([tf.keras.activations.get(x) for x in activation])
        self._conf_kernel_regularizer = list([tf.keras.regularizers.get(x) for x in kernel_regularizer])
        self._conf_bias_regularizer = list([tf.keras.regularizers.get(x) for x in bias_regularizer])
        self._conf_activity_regularizer = list([tf.keras.regularizers.get(x) for x in activity_regularizer])
        self._conf_kernel_initializer = list([tf.keras.initializers.get(x) for x in kernel_initializer])
        self._conf_bias_initializer = list([tf.keras.initializers.get(x) for x in bias_initializer])
        self._conf_kernel_constraint = list([tf.keras.constraints.get(x) for x in kernel_constraint])
        self._conf_bias_constraint = list([tf.keras.constraints.get(x) for x in bias_constraint])
        # Serialized args for norm
        self._conf_use_normalization = list(use_normalization)
        self._conf_normalization_technique = list(normalization_technique)
        self._conf_axis = list(axis)
        self._conf_momentum = list(momentum)
        self._conf_epsilon = list(epsilon)
        self._conf_center = list(center)
        self._conf_scale = list(scale)
        self._conf_beta_initializer = list([tf.keras.initializers.get(x) for x in beta_initializer])
        self._conf_gamma_initializer = list([tf.keras.initializers.get(x) for x in gamma_initializer])
        self._conf_moving_mean_initializer = list([tf.keras.initializers.get(x) for x in moving_mean_initializer])
        self._conf_moving_variance_initializer = list(
            [tf.keras.initializers.get(x) for x in moving_variance_initializer])
        self._conf_beta_regularizer = list([tf.keras.regularizers.get(x) for x in beta_regularizer])
        self._conf_gamma_regularizer = list([tf.keras.regularizers.get(x) for x in gamma_regularizer])
        self._conf_beta_constraint = list([tf.keras.constraints.get(x) for x in beta_constraint])
        self._conf_gamma_constraint = list([tf.keras.constraints.get(x) for x in gamma_constraint])
        # Dropout
        self._conf_use_dropout = list(use_dropout)
        self._conf_rate = list(rate)
        self._conf_seed = list(seed)
        self._conf_noise_shape = list(noise_shape)

        # Processed arguments for each layer.
        self._conf_mlp_dense_layer_kwargs = [{"units": self._conf_units[i],
                                              "use_bias": self._conf_use_bias[i],
                                              "name": self.name + "_dense_" + str(i),
                                              "activation": "linear",
                                              "activity_regularizer": None,
                                              "kernel_regularizer": self._conf_kernel_regularizer[i],
                                              "bias_regularizer": self._conf_bias_regularizer[i],
                                              "kernel_initializer": self._conf_kernel_initializer[i],
                                              "bias_initializer": self._conf_bias_initializer[i],
                                              "kernel_constraint": self._conf_kernel_constraint[i],
                                              "bias_constraint": self._conf_bias_constraint[i]}
                                             for i in range(self._depth)]
        self._conf_mlp_activ_layer_kwargs = [{"activation": self._conf_activation[i],
                                              "name": self.name + "_act_" + str(i),
                                              "activity_regularizer": self._conf_activity_regularizer[i]}
                                             for i in range(self._depth)]
        self._conf_mlp_norm_layer_kwargs = [{"name": self.name + '_norm_' + str(i),
                                             "axis": self._conf_axis[i],
                                             "epsilon": self._conf_epsilon[i],
                                             "center": self._conf_center[i],
                                             "scale": self._conf_scale[i],
                                             "beta_initializer": self._conf_beta_initializer[i],
                                             "gamma_initializer": self._conf_gamma_initializer[i],
                                             "beta_regularizer": self._conf_beta_regularizer[i],
                                             "gamma_regularizer": self._conf_gamma_regularizer[i],
                                             "beta_constraint": self._conf_beta_constraint[i],
                                             "gamma_constraint": self._conf_gamma_constraint[i]}
                                            for i in range(self._depth)]
        self._conf_mlp_drop_layer_kwargs = [{"name": self.name + '_drop_' + str(i),
                                             "rate": self._conf_rate[i],
                                             "noise_shape": self._conf_noise_shape[i],
                                             "seed": self._conf_seed[i]}
                                            for i in range(self._depth)]

    def build(self, input_shape):
        """Build layer."""
        super(MLPBase, self).build(input_shape)

    def get_config(self):
        """Update config."""
        config = super(MLPBase, self).get_config()
        config.update({
            # Dense
            "units": self._conf_units,
            'use_bias': self._conf_use_bias,
            'activation': [tf.keras.activations.serialize(x) for x in self._conf_activation],
            'activity_regularizer': [tf.keras.regularizers.serialize(x) for x in
                                     self._conf_activity_regularizer],
            'kernel_regularizer': [tf.keras.regularizers.serialize(x) for x in self._conf_kernel_regularizer],
            'bias_regularizer': [tf.keras.regularizers.serialize(x) for x in self._conf_bias_regularizer],
            "kernel_initializer": [tf.keras.initializers.serialize(x) for x in self._conf_kernel_initializer],
            "bias_initializer": [tf.keras.initializers.serialize(x) for x in self._conf_bias_initializer],
            "kernel_constraint": [tf.keras.constraints.serialize(x) for x in self._conf_kernel_constraint],
            "bias_constraint": [tf.keras.constraints.serialize(x) for x in self._conf_bias_constraint],
            # Norm
            "normalization_technique": self._conf_normalization_technique,
            "use_normalization": self._conf_use_normalization,
            "axis": list(self._conf_axis),
            "momentum": self._conf_momentum,
            "epsilon": self._conf_epsilon,
            "center": self._conf_center,
            "scale": self._conf_scale,
            "beta_initializer": [tf.keras.initializers.serialize(x) for x in self._conf_beta_initializer],
            "gamma_initializer": [tf.keras.initializers.serialize(x) for x in self._conf_gamma_initializer],
            "moving_mean_initializer": [tf.keras.initializers.serialize(x) for x in
                                        self._conf_moving_mean_initializer],
            "moving_variance_initializer": [tf.keras.initializers.serialize(x) for x in
                                            self._conf_moving_variance_initializer],
            "beta_regularizer": [tf.keras.regularizers.serialize(x) for x in self._conf_beta_regularizer],
            "gamma_regularizer": [tf.keras.regularizers.serialize(x) for x in self._conf_gamma_regularizer],
            "beta_constraint": [tf.keras.constraints.serialize(x) for x in self._conf_beta_constraint],
            "gamma_constraint": [tf.keras.constraints.serialize(x) for x in self._conf_gamma_constraint],
            # Dropout
            "use_dropout": self._conf_use_dropout,
            'rate': self._conf_rate,
            'noise_shape': self._conf_noise_shape,
            'seed': self._conf_seed
        })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GraphMLP')
class GraphMLP(MLPBase):
    r"""Multilayer perceptron that consist of N dense layers for ragged embedding tensors.
    See layer arguments of :obj:`MLPBase` for configuration.
    This layer adds normalization for embeddings tensors of node or edge embeddings represented by a ragged tensor.
    The definition of graph, batch or layer normalization is different from the standard keras definition.
    See :obj:`kgcnn.layers.norm` for more information.
    """

    def __init__(self, units, **kwargs):
        """Initialize MLP as for dense."""
        super(GraphMLP, self).__init__(units=units, **kwargs)

        self.mlp_dense_layer_list = [DenseEmbedding(
            **self._conf_mlp_dense_layer_kwargs[i]) for i in range(self._depth)]

        self.mlp_activation_layer_list = [ActivationEmbedding(
            **self._conf_mlp_activ_layer_kwargs[i]) for i in range(self._depth)]

        self.mlp_dropout_layer_list = [
            DropoutEmbedding(**self._conf_mlp_drop_layer_kwargs[i]) if self._conf_use_dropout[i] else None for i in
            range(self._depth)]

        self.mlp_norm_layer_list = [None] * self._depth
        for i in range(self._depth):
            if self._conf_use_normalization[i]:
                if self._conf_normalization_technique[i] in ["batch", "BatchNormalization", "GraphBatchNormalization"]:
                    self.mlp_norm_layer_list[i] = GraphBatchNormalization(
                        **self._conf_mlp_norm_layer_kwargs[i],
                        momentum=self._conf_momentum[i],
                        moving_mean_initializer=self._conf_moving_mean_initializer[i],
                        moving_variance_initializer=self._conf_moving_variance_initializer[i])
                elif self._conf_normalization_technique[i] in ["layer", "LayerNormalization",
                                                               "GraphLayerNormalization"]:
                    self.mlp_norm_layer_list[i] = GraphLayerNormalization(
                        **self._conf_mlp_norm_layer_kwargs[i])
                else:
                    raise NotImplementedError(
                        "ERROR: Normalization via %s not supported." % self._conf_normalization_technique[i])

    def build(self, input_shape):
        """Build layer."""
        super(GraphMLP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Input tensor with last dimension not None.

        Returns:
            tf.Tensor: MLP pass.
        """
        x = inputs
        for i in range(len(self._conf_units)):
            x = self.mlp_dense_layer_list[i](x, **kwargs)
            if self._conf_use_dropout[i]:
                x = self.mlp_dropout_layer_list[i](x, **kwargs)
            if self._conf_use_normalization[i]:
                x = self.mlp_norm_layer_list[i](x, **kwargs)
            x = self.mlp_activation_layer_list[i](x, **kwargs)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(GraphMLP, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MLP')
class MLP(MLPBase):
    r"""Multilayer perceptron that consist of N dense layers for ragged embedding tensors.
    See layer arguments of :obj:`MLPBase` for configuration.
    This layer adds normalization and dropout for normal tensor input.
    """

    def __init__(self, units, **kwargs):
        """Initialize MLP. See MLPBase."""
        super(MLP, self).__init__(units=units, **kwargs)

        self.mlp_dense_layer_list = [tf.keras.layers.Dense(
            **self._conf_mlp_dense_layer_kwargs[i]) for i in range(self._depth)]

        self.mlp_activation_layer_list = [tf.keras.layers.Activation(
            **self._conf_mlp_activ_layer_kwargs[i]) for i in range(self._depth)]

        self.mlp_dropout_layer_list = [
            tf.keras.layers.Dropout(**self._conf_mlp_drop_layer_kwargs[i]) if self._conf_use_dropout[i] else None for i
            in range(self._depth)]

        self.mlp_norm_layer_list = [None] * self._depth
        for i in range(self._depth):
            if self._conf_use_normalization[i]:
                if self._conf_normalization_technique[i] in ["batch", "BatchNormalization", "GraphBatchNormalization"]:
                    self.mlp_norm_layer_list[i] = tf.keras.layers.BatchNormalization(
                        **self._conf_mlp_norm_layer_kwargs[i],
                        momentum=self._conf_momentum[i],
                        moving_mean_initializer=self._conf_moving_mean_initializer[i],
                        moving_variance_initializer=self._conf_moving_variance_initializer[i])
                elif self._conf_normalization_technique[i] in ["layer", "LayerNormalization",
                                                               "GraphLayerNormalization"]:
                    self.mlp_norm_layer_list[i] = tf.keras.layers.LayerNormalization(
                        **self._conf_mlp_norm_layer_kwargs[i])
                else:
                    raise NotImplementedError(
                        "ERROR: Normalization via %s not supported." % self._conf_normalization_technique[i])

    def build(self, input_shape):
        """Build layer."""
        super(MLP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor with last dimension not None.

        Returns:
            tf.Tensor: MLP pass.
        """
        x = inputs
        for i in range(len(self._conf_units)):
            x = self.mlp_dense_layer_list[i](x, **kwargs)
            if self._conf_use_dropout[i]:
                x = self.mlp_dropout_layer_list[i](x, **kwargs)
            if self._conf_use_normalization[i]:
                x = self.mlp_norm_layer_list[i](x, **kwargs)
            x = self.mlp_activation_layer_list[i](x, **kwargs)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(MLP, self).get_config()
        return config
