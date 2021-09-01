import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.keras import Dense, Activation, BatchNormalization
from kgcnn.layers.base import GraphBaseLayer
import kgcnn.ops.activ


# import tensorflow.keras.backend as ksb
@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MLP')
class MLP(GraphBaseLayer):
    """Multilayer perceptron that consist of N dense keras layers. Supply list in place of arguments for each layer.
    If not list, then the argument is used for each layer. The number of layers is given by units, which should be list.
        
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
                 **kwargs):
        """Initialize MLP as for dense."""
        super(MLP, self).__init__(**kwargs)
        # Make to one element list
        if isinstance(units, int):
            units = [units]

        self.depth = len(units)

        def assert_args_is_list(args):
            if not isinstance(args, (list, tuple)):
                return [args for _ in range(self.depth)]
            return args

        use_bias = assert_args_is_list(use_bias)
        activation = assert_args_is_list(activation)
        kernel_regularizer = assert_args_is_list(kernel_regularizer)
        bias_regularizer = assert_args_is_list(bias_regularizer)
        activity_regularizer = assert_args_is_list(activity_regularizer)
        kernel_initializer = assert_args_is_list(kernel_initializer)
        bias_initializer = assert_args_is_list(bias_initializer)
        kernel_constraint = assert_args_is_list(kernel_constraint)
        bias_constraint = assert_args_is_list(bias_constraint)

        for x in [activation, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_initializer,
                  bias_initializer, kernel_constraint, bias_constraint, use_bias]:
            if len(x) != len(units):
                raise ValueError("Error: Provide matching list of units", units, "and", x, "or simply a single value.")

        # Serialized args
        self.mlp_units = list(units)
        self.mlp_use_bias = list(use_bias)
        self.mlp_activation = list([tf.keras.activations.get(x) for x in activation])
        self.mlp_kernel_regularizer = list([tf.keras.regularizers.get(x) for x in kernel_regularizer])
        self.mlp_bias_regularizer = list([tf.keras.regularizers.get(x) for x in bias_regularizer])
        self.mlp_activity_regularizer = list([tf.keras.regularizers.get(x) for x in activity_regularizer])
        self.mlp_kernel_initializer = list([tf.keras.initializers.get(x) for x in kernel_initializer])
        self.mlp_bias_initializer = list([tf.keras.initializers.get(x) for x in bias_initializer])
        self.mlp_kernel_constraint = list([tf.keras.constraints.get(x) for x in kernel_constraint])
        self.mlp_bias_constraint = list([tf.keras.constraints.get(x) for x in bias_constraint])

        self.mlp_dense_list = [Dense(
            units=self.mlp_units[i],
            use_bias=self.mlp_use_bias[i],
            name=self.name + '_dense_' + str(i),
            activation=self.mlp_activation[i],
            activity_regularizer=self.mlp_activity_regularizer[i],
            kernel_regularizer=self.mlp_kernel_regularizer[i],
            bias_regularizer=self.mlp_bias_regularizer[i],
            kernel_initializer=self.mlp_kernel_initializer[i],
            bias_initializer=self.mlp_bias_initializer[i],
            kernel_constraint=self.mlp_kernel_constraint[i],
            bias_constraint=self.mlp_bias_constraint[i],
            ragged_validate=self.ragged_validate,
            input_tensor_type=self.input_tensor_type
        ) for i in range(len(self.mlp_units))]

    def build(self, input_shape):
        """Build layer."""
        super(MLP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.Tensor, tf.RaggedTensor): Input tensor with last dimension not None.

        Returns:
            tf.Tensor: MLP pass.
        """
        x = inputs
        for i in range(len(self.mlp_units)):
            x = self.mlp_dense_list[i](x, **kwargs)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(MLP, self).get_config()
        config.update({"units": self.mlp_units,
                       'use_bias': self.mlp_use_bias,
                       'activation': [tf.keras.activations.serialize(x) for x in self.mlp_activation],
                       'activity_regularizer': [tf.keras.regularizers.serialize(x) for x in
                                                self.mlp_activity_regularizer],
                       'kernel_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_kernel_regularizer],
                       'bias_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_bias_regularizer],
                       "kernel_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_kernel_initializer],
                       "bias_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_bias_initializer],
                       "kernel_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_kernel_constraint],
                       "bias_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_bias_constraint],
                       })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='BatchNormMLP')
class BatchNormMLP(GraphBaseLayer):
    """Multilayer perceptron that consist of N dense keras layers. Supply list in place of arguments for each layer.
    If not list, then the argument is used for each layer. The number of layers is given by units, which should be list.
    Additionally a batch-normalization is applied.

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
        axis: Integer, the axis that should be normalized (typically the features
            axis). For instance, after a `Conv2D` layer with
            `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
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
                 # Batch-Normalization args
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 **kwargs):
        """Initialize MLP as for dense."""
        super(BatchNormMLP, self).__init__(**kwargs)
        # Make to one element list
        if isinstance(units, int):
            units = [units]

        self.depth = len(units)

        def assert_args_is_list(args):
            if not isinstance(args, (list, tuple)):
                return [args for _ in range(self.depth)]
            return args

        use_bias = assert_args_is_list(use_bias)
        activation = assert_args_is_list(activation)
        kernel_regularizer = assert_args_is_list(kernel_regularizer)
        bias_regularizer = assert_args_is_list(bias_regularizer)
        activity_regularizer = assert_args_is_list(activity_regularizer)
        kernel_initializer = assert_args_is_list(kernel_initializer)
        bias_initializer = assert_args_is_list(bias_initializer)
        kernel_constraint = assert_args_is_list(kernel_constraint)
        bias_constraint = assert_args_is_list(bias_constraint)

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

        for x in [activation, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_initializer,
                  bias_initializer, kernel_constraint, bias_constraint, use_bias, axis, momentum, epsilon,
                  center, scale, beta_initializer, gamma_initializer, moving_mean_initializer,
                  moving_variance_initializer, beta_regularizer, gamma_regularizer, beta_constraint,
                  gamma_constraint]:
            if len(x) != len(units):
                raise ValueError("Error: Provide matching list of units", units, "and", x, "or simply a single value.")

        # Serialized args
        self.mlp_units = list(units)
        self.mlp_use_bias = list(use_bias)
        self.mlp_activation = list([tf.keras.activations.get(x) for x in activation])
        self.mlp_kernel_regularizer = list([tf.keras.regularizers.get(x) for x in kernel_regularizer])
        self.mlp_bias_regularizer = list([tf.keras.regularizers.get(x) for x in bias_regularizer])
        self.mlp_activity_regularizer = list([tf.keras.regularizers.get(x) for x in activity_regularizer])
        self.mlp_kernel_initializer = list([tf.keras.initializers.get(x) for x in kernel_initializer])
        self.mlp_bias_initializer = list([tf.keras.initializers.get(x) for x in bias_initializer])
        self.mlp_kernel_constraint = list([tf.keras.constraints.get(x) for x in kernel_constraint])
        self.mlp_bias_constraint = list([tf.keras.constraints.get(x) for x in bias_constraint])
        # Serialized args for batch-norm
        self.mlp_axis = list(axis)
        self.mlp_momentum = list(momentum)
        self.mlp_epsilon = list(epsilon)
        self.mlp_center = list(center)
        self.mlp_scale = list(scale)
        self.mlp_beta_initializer = list([tf.keras.initializers.get(x) for x in beta_initializer])
        self.mlp_gamma_initializer = list([tf.keras.initializers.get(x) for x in gamma_initializer])
        self.mlp_moving_mean_initializer = list([tf.keras.initializers.get(x) for x in moving_mean_initializer])
        self.mlp_moving_variance_initializer = list([tf.keras.initializers.get(x) for x in moving_variance_initializer])
        self.mlp_beta_regularizer = list([tf.keras.regularizers.get(x) for x in beta_regularizer])
        self.mlp_gamma_regularizer = list([tf.keras.regularizers.get(x) for x in gamma_regularizer])
        self.mlp_beta_constraint = list([tf.keras.constraints.get(x) for x in beta_constraint])
        self.mlp_gamma_constraint = list([tf.keras.constraints.get(x) for x in gamma_constraint])

        self.mlp_dense_list = [Dense(
            units=self.mlp_units[i],
            use_bias=self.mlp_use_bias[i],
            name=self.name + '_dense_' + str(i),
            activation="linear",
            activity_regularizer=None,
            kernel_regularizer=self.mlp_kernel_regularizer[i],
            bias_regularizer=self.mlp_bias_regularizer[i],
            kernel_initializer=self.mlp_kernel_initializer[i],
            bias_initializer=self.mlp_bias_initializer[i],
            kernel_constraint=self.mlp_kernel_constraint[i],
            bias_constraint=self.mlp_bias_constraint[i],
            ragged_validate=self.ragged_validate,
            input_tensor_type=self.input_tensor_type
        ) for i in range(len(self.mlp_units))]

        self.mlp_activation_layer_list = [Activation(
            activation=self.mlp_activation[i],
            activity_regularizer=self.mlp_activity_regularizer[i],
        ) for i in range(len(self.mlp_units))]

        self.mlp_batch_norm_list = [BatchNormalization(
            axis=self.mlp_axis[i],
            momentum=self.mlp_momentum[i],
            epsilon=self.mlp_epsilon[i],
            center=self.mlp_center[i],
            scale=self.mlp_scale[i],
            beta_initializer=self.mlp_beta_initializer[i],
            gamma_initializer=self.mlp_gamma_initializer[i],
            moving_mean_initializer=self.mlp_moving_mean_initializer[i],
            moving_variance_initializer=self.mlp_moving_variance_initializer[i],
            beta_regularizer=self.mlp_beta_regularizer[i],
            gamma_regularizer=self.mlp_gamma_regularizer[i],
            beta_constraint=self.mlp_beta_constraint[i],
            gamma_constraint=self.mlp_gamma_constraint[i],
        ) for i in range(len(self.mlp_units))]

    def build(self, input_shape):
        """Build layer."""
        super(BatchNormMLP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.Tensor, tf.RaggedTensor): Input tensor with last dimension not None.

        Returns:
            tf.Tensor: MLP pass.
        """
        x = inputs
        for i in range(len(self.mlp_units)):
            x = self.mlp_dense_list[i](x, **kwargs)
            x = self.mlp_batch_norm_list[i](x, **kwargs)
            x = self.mlp_activation_layer_list[i](x, **kwargs)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(BatchNormMLP, self).get_config()
        config.update({"units": self.mlp_units,
                       'use_bias': self.mlp_use_bias,
                       'activation': [tf.keras.activations.serialize(x) for x in self.mlp_activation],
                       'activity_regularizer': [tf.keras.regularizers.serialize(x) for x in
                                                self.mlp_activity_regularizer],
                       'kernel_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_kernel_regularizer],
                       'bias_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_bias_regularizer],
                       "kernel_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_kernel_initializer],
                       "bias_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_bias_initializer],
                       "kernel_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_kernel_constraint],
                       "bias_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_bias_constraint],
                       "axis": list(self.mlp_axis),
                       "momentum": self.mlp_momentum,
                       "epsilon": self.mlp_epsilon,
                       "center": self.mlp_center,
                       "scale": self.mlp_scale,
                       "beta_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_beta_initializer],
                       "gamma_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_gamma_initializer],
                       "moving_mean_initializer": [tf.keras.initializers.serialize(x) for x in
                                                   self.mlp_moving_mean_initializer],
                       "moving_variance_initializer": [tf.keras.initializers.serialize(x) for x in
                                                       self.mlp_moving_variance_initializer],
                       "beta_regularizer": [tf.keras.regularizers.serialize(x) for x in self.mlp_beta_regularizer],
                       "gamma_regularizer": [tf.keras.regularizers.serialize(x) for x in self.mlp_gamma_regularizer],
                       "beta_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_beta_constraint],
                       "gamma_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_gamma_constraint]
                       })
        return config
