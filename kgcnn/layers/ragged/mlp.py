import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.ragged.conv import DenseRagged
from kgcnn.utils.activ import kgcnn_custom_act


class MLPRagged(ks.layers.Layer):
    """
    Multilayer perceptron that consist of N dense keras layers.
        
    Args:
        mlp_units (list): Size of hidden layers for each layer.
        mlp_use_bias (list, optional): Use bias for hidden layers. Defaults to True.
        mlp_activation (list, optional): Activity identifier. Defaults to None.
        mlp_activity_regularizer (list, optional): Activity regularizer identifier. Defaults to None.
        mlp_kernel_regularizer (list, optional): Kernel regularizer identifier. Defaults to None.
        mlp_bias_regularizer (list, optional): Bias regularizer identifier. Defaults to None.
        **kwargs 
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 activation=None,
                 activity_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        """Init MLP as for dense."""
        super(MLPRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True

        # Make to one element list
        if isinstance(units, int):
            units = [units]

        if not isinstance(use_bias, list) and not isinstance(use_bias, tuple):
            use_bias = [use_bias for _ in units]
        else:
            if len(use_bias) != len(units):
                raise ValueError("Units and bias list must be same length, got", use_bias, units)

        if not isinstance(activation, list) and not isinstance(activation, tuple):
            activation = [activation for _ in units]
        else:
            if len(activation) != len(units):
                raise ValueError("Units and activation list must be same length, got", activation, units)

        if not isinstance(kernel_regularizer, list) and not isinstance(kernel_regularizer, tuple):
            kernel_regularizer = [kernel_regularizer for _ in units]
        else:
            if len(kernel_regularizer) != len(units):
                raise ValueError("Units and kernel_regularizer list must be same length, got", kernel_regularizer,
                                 units)

        if not isinstance(bias_regularizer, list) and not isinstance(bias_regularizer, tuple):
            bias_regularizer = [bias_regularizer for _ in units]
        else:
            if len(bias_regularizer) != len(units):
                raise ValueError("Units and bias_regularizer list must be same length, got", bias_regularizer, units)

        if not isinstance(activity_regularizer, list) and not isinstance(activity_regularizer, tuple):
            activity_regularizer = [activity_regularizer for _ in units]
        else:
            if len(activity_regularizer) != len(units):
                raise ValueError("Units and activity_regularizer list must be same length, got", activity_regularizer,
                                 units)

        # Serialized props
        self.mlp_units = list(units)
        self.mlp_use_bias = list(use_bias)
        self.mlp_activation = list([tf.keras.activations.get(x) for x in activation])
        self.mlp_kernel_regularizer = list([tf.keras.regularizers.get(x) for x in kernel_regularizer])
        self.mlp_bias_regularizer = list([tf.keras.regularizers.get(x) for x in bias_regularizer])
        self.mlp_activity_regularizer = list([tf.keras.regularizers.get(x) for x in activity_regularizer])

        self.mlp_dense_list = [DenseRagged(
            self.mlp_units[i],
            use_bias=self.mlp_use_bias[i],
            name=self.name + '_dense_' + str(i),
            activation=self.mlp_activation[i],
            activity_regularizer=self.mlp_activity_regularizer[i],
            kernel_regularizer=self.mlp_kernel_regularizer[i],
            bias_regularizer=self.mlp_bias_regularizer[i]
        ) for i in range(len(self.mlp_units))]

    def build(self, input_shape):
        """Build layer."""
        super(MLPRagged, self).build(input_shape)

    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs (tf.ragged): Input ragged tensor of shape (...,N).
            training (bool): Default False.

        Returns:
            tf.ragged: MLP pass.
        """
        x = inputs
        for i in range(len(self.mlp_units)):
            x = self.mlp_dense_list[i](x)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(MLPRagged, self).get_config()
        config.update({"units": self.mlp_units,
                       'use_bias': self.mlp_use_bias,
                       'activation': [tf.keras.activations.serialize(x) for x in self.mlp_activation],
                       'activity_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_activity_regularizer],
                       'kernel_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_kernel_regularizer],
                       'bias_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_bias_regularizer],
                       })
        return config
