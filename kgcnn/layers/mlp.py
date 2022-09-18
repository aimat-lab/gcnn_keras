import tensorflow as tf
import kgcnn.ops.activ
from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.norm import GraphBatchNormalization, GraphLayerNormalization
from kgcnn.layers.base import GraphBaseLayer

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='MLPBase')
class MLPBase(GraphBaseLayer):
    r"""Base class for multilayer perceptron that consist of multiple feed-forward networks.

    This base class simply manages layer arguments for :obj:`MLP`. They contain arguments for :obj:`Dense`,
    :obj:`Dropout` and :obj:`BatchNormalization` or :obj:`LayerNormalization`,
    since MLP is made up of stacked :obj:`Dense` layers with optional normalization and
    dropout to improve stability or regularization. Here, a list in place of arguments must be provided that applies
    to each layer. If not a list is given, then the single argument is used for each layer.
    The number of layers is determined by :obj:`units` argument, which should be list.

    Hence, this base class holds arguments for batch-normalization which should be applied between kernel
    and activation. And dropout after the kernel output and before normalization.

    This base class does not initialize any sub-layers or implements :obj:`call`, only for managing arguments.

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
                 axis=-1,
                 momentum=0.99,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 # Dropout
                 use_dropout=False,
                 rate=None,
                 noise_shape=None,
                 seed=None,
                 **kwargs):
        r"""Initialize with parameter for MLP layer that match :obj:`Dense` layer, including :obj:`Dropout` and
        :obj:`BatchNormalization` or :obj:`LayerNormalization`.

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
        super(MLPBase, self).__init__(**kwargs)
        local_kw = locals()

        # List for groups of arguments.
        key_list_act = ["activation", "activity_regularizer"]
        key_list_dense = ["units", "use_bias", "kernel_regularizer", "bias_regularizer",
                          "kernel_initializer", "bias_initializer", "kernel_constraint", "bias_constraint"]
        key_list_norm = ["axis", "momentum", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
                         "moving_mean_initializer", "moving_variance_initializer", "beta_regularizer",
                         "gamma_regularizer",
                         "beta_constraint", "gamma_constraint"]
        key_list_dropout = ["rate", "noise_shape", "seed"]
        key_list_use = ["use_dropout", "use_normalization", "normalization_technique"]
        self._key_list_init = ["kernel_initializer", "bias_initializer", "beta_initializer", "gamma_initializer",
                               "moving_mean_initializer", "moving_variance_initializer"]
        self._key_list_reg = ["activity_regularizer", "kernel_regularizer", "bias_regularizer",
                              "beta_regularizer", "gamma_regularizer"]
        self._key_list_const = ["kernel_constraint", "bias_constraint", "beta_constraint", "gamma_constraint"]

        # Dictionary of kwargs for MLP.
        self._key_list = key_list_act + key_list_dense + key_list_norm + key_list_dropout + key_list_use
        mlp_kwargs = {key: local_kw[key] for key in self._key_list}

        # Everything should be defined by units.
        if isinstance(units, int):
            units = [units]
        if not isinstance(units, list):
            raise ValueError("Units must be a list or a single int for `MLP`.")
        self._depth = len(units)
        # Special case, if axis is supposed to be multiple axis, use tuple here.
        if not isinstance(axis, list):
            axis = [axis for _ in units]
        # Special case, for shape, use tuple here.
        if not isinstance(noise_shape, list):
            noise_shape = [noise_shape for _ in units]

        # Assert matching number of args
        def assert_args_is_list(args):
            if not isinstance(args, (list, tuple)):
                return [args for _ in range(self._depth)]
            return args

        # Make every argument to list.
        for key, value in mlp_kwargs.items():
            mlp_kwargs[key] = assert_args_is_list(value)

        # Check correct length for all arguments.
        for key, value in mlp_kwargs.items():
            if len(units) != len(value):
                raise ValueError("Provide matching list of units %s and %s or simply a single value." % (units, key))

        # Deserialize initializer, regularizes, constraints and activation.
        for sl, sm in [
            (self._key_list_init, ks.initializers.get), (self._key_list_reg, ks.regularizers.get),
            (self._key_list_const, ks.constraints.get), (["activation"], ks.activations.get)
        ]:
            for key in sl:
                mlp_kwargs[key] = [sm(x) for x in mlp_kwargs[key]]

        # Assign to self as '_conf_'.
        for key, value in mlp_kwargs.items():
            setattr(self, "_conf_" + key, list(value))

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
        self._conf_mlp_batch_layer_kwargs = [{"name": self.name + '_norm_' + str(i),
                                             "axis": self._conf_axis[i],
                                             "epsilon": self._conf_epsilon[i],
                                             "center": self._conf_center[i],
                                             "scale": self._conf_scale[i],
                                             "beta_initializer": self._conf_beta_initializer[i],
                                             "gamma_initializer": self._conf_gamma_initializer[i],
                                             "beta_regularizer": self._conf_beta_regularizer[i],
                                             "gamma_regularizer": self._conf_gamma_regularizer[i],
                                             "beta_constraint": self._conf_beta_constraint[i],
                                             "gamma_constraint": self._conf_gamma_constraint[i],
                                             "momentum": self._conf_momentum[i],
                                             "moving_mean_initializer": self._conf_moving_mean_initializer[i],
                                             "moving_variance_initializer": self._conf_moving_variance_initializer[i]
                                              }
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
        for key in self._key_list:
            config.update({key: getattr(self, "_conf_" + key)})

        # Serialize initializer, regularizes, constraints and activation.
        for sl, sm in [
            (self._key_list_init, ks.initializers.serialize), (self._key_list_reg, ks.regularizers.serialize),
            (self._key_list_const, ks.constraints.serialize), (["activation"], ks.activations.serialize)
        ]:
            for key in sl:
                config.update({key: [sm(x) for x in getattr(self, "_conf_" + key)]})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='MLP')
class MLP(MLPBase):
    r"""Multilayer perceptron that consist of multiple :obj:`Dense` layers.

    .. note::
        Please see layer arguments of :obj:`MLPBase` for configuration!

    This layer adds normalization and dropout for normal tensor input. Please, see keras
    `documentation <https://www.tensorflow.org/api_docs/python/tf>`_ of
    :obj:`Dense`, :obj:`Dropout`, :obj:`BatchNormalization` and :obj:`LayerNormalization` for more information.

    Additionally, graph oriented normalization is supported. You can choose :obj:`normalization_technique` to be
    either 'BatchNormalization', 'LayerNormalization', 'GraphLayerNormalization', or 'GraphBatchNormalization'.

    """

    def __init__(self, units, **kwargs):
        """Initialize MLP. See MLPBase."""
        super(MLP, self).__init__(units=units, **kwargs)

        self.mlp_dense_layer_list = [DenseEmbedding(
            **self._conf_mlp_dense_layer_kwargs[i]) for i in range(self._depth)]

        self.mlp_activation_layer_list = [ActivationEmbedding(
            **self._conf_mlp_activ_layer_kwargs[i]) for i in range(self._depth)]

        self.mlp_dropout_layer_list = [
            DropoutEmbedding(**self._conf_mlp_drop_layer_kwargs[i]) if self._conf_use_dropout[i] else None for i
            in range(self._depth)]

        self.mlp_norm_layer_list = [None] * self._depth
        for i in range(self._depth):
            if self._conf_use_normalization[i]:
                if self._conf_normalization_technique[i] in ["batch", "BatchNormalization"]:
                    self.mlp_norm_layer_list[i] = ks.layers.BatchNormalization(
                        **self._conf_mlp_batch_layer_kwargs[i])
                elif self._conf_normalization_technique[i] in ["graph_batch", "GraphBatchNormalization"]:
                    self.mlp_norm_layer_list[i] = GraphBatchNormalization(
                        **self._conf_mlp_batch_layer_kwargs[i])
                elif self._conf_normalization_technique[i] in ["layer", "LayerNormalization"]:
                    self.mlp_norm_layer_list[i] = ks.layers.LayerNormalization(
                        **self._conf_mlp_norm_layer_kwargs[i])
                elif self._conf_normalization_technique[i] in ["graph_layer", "GraphLayerNormalization"]:
                    self.mlp_norm_layer_list[i] = GraphLayerNormalization(
                        **self._conf_mlp_norm_layer_kwargs[i])
                else:
                    raise NotImplementedError(
                        "Normalization via %s not supported." % self._conf_normalization_technique[i])

    def build(self, input_shape):
        """Build layer."""
        super(MLP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor with last dimension not `None`.

        Returns:
            tf.Tensor: MLP forward pass.
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


GraphMLP = MLP
