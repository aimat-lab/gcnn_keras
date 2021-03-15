# import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.utils.activ import kgcnn_custom_act


# import tensorflow.keras.backend as ksb


class MLP(ks.layers.Layer):
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
                 mlp_units,
                 mlp_use_bias=True,
                 mlp_activation=None,
                 mlp_activity_regularizer=None,
                 mlp_kernel_regularizer=None,
                 mlp_bias_regularizer=None,
                 **kwargs):
        """Init MLP as for dense."""
        super(MLP, self).__init__(**kwargs)

        # Make to one element list
        if isinstance(mlp_units, int):
            mlp_units = [mlp_units]
        if not isinstance(mlp_use_bias, list) and not isinstance(mlp_use_bias, tuple):
            mlp_use_bias = [mlp_use_bias for _ in mlp_units]
        if not isinstance(mlp_activation, list) and not isinstance(mlp_activation, tuple):
            mlp_activation = [mlp_activation for _ in mlp_units]
        if not isinstance(mlp_activity_regularizer, list) and not isinstance(mlp_activity_regularizer, tuple):
            mlp_activity_regularizer = [mlp_activity_regularizer for _ in mlp_units]
        if not isinstance(mlp_kernel_regularizer, list) and not isinstance(mlp_kernel_regularizer, tuple):
            mlp_kernel_regularizer = [mlp_kernel_regularizer for _ in mlp_units]
        if not isinstance(mlp_bias_regularizer, list) and not isinstance(mlp_bias_regularizer, tuple):
            mlp_bias_regularizer = [mlp_bias_regularizer for _ in mlp_units]

        # Serialized props
        self.mlp_units = mlp_units
        self.mlp_use_bias = mlp_use_bias
        self.mlp_activation = [x if isinstance(x, str) or isinstance(x, dict) else ks.activations.serialize(x) for x in
                               mlp_activation]
        self.mlp_activity_regularizer = [x if isinstance(x, str) or isinstance(x, dict) else ks.activations.serialize(x)
                                         for x in mlp_activity_regularizer]
        self.mlp_kernel_regularizer = [x if isinstance(x, str) or isinstance(x, dict) else ks.activations.serialize(x)
                                       for x in mlp_kernel_regularizer]
        self.mlp_bias_regularizer = [x if isinstance(x, str) or isinstance(x, dict) else ks.activations.serialize(x) for
                                     x in mlp_bias_regularizer]

        # Deserialized props
        self.des_mlp_activation = [ks.activations.deserialize(x, custom_objects=kgcnn_custom_act) for x in
                                   self.mlp_activation]
        self.des_mlp_activity_regularizer = [ks.regularizers.deserialize(x) for x in self.mlp_activity_regularizer]
        self.des_mlp_kernel_regularizer = [ks.regularizers.deserialize(x) for x in self.mlp_kernel_regularizer]
        self.des_mlp_bias_regularizer = [ks.regularizers.deserialize(x) for x in self.mlp_bias_regularizer]

        self.mlp_dense_list = [ks.layers.Dense(
            self.mlp_units[i],
            use_bias=self.mlp_use_bias[i],
            name=self.name + '_dense_' + str(i),
            activation=self.des_mlp_activation[i],
            activity_regularizer=self.des_mlp_activity_regularizer[i],
            kernel_regularizer=self.des_mlp_kernel_regularizer[i],
            bias_regularizer=self.des_mlp_bias_regularizer[i]
        ) for i in range(len(self.mlp_units))]

    def build(self, input_shape):
        """Build layer."""
        super(MLP, self).build(input_shape)

    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs (tf.tensor): Input tensor of shape (...,N).
            training (bool)

        Returns:
            out (tf.tensor): MLP pass.
        
        """
        x = inputs
        for i in range(len(self.mlp_units)):
            x = self.mlp_dense_list[i](x)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(MLP, self).get_config()
        config.update({"mlp_units": self.mlp_units,
                       'mlp_use_bias': self.mlp_use_bias,
                       'mlp_activation': self.mlp_activation,
                       'mlp_activity_regularizer': self.mlp_activity_regularizer,
                       'mlp_kernel_regularizer': self.mlp_kernel_regularizer,
                       'mlp_bias_regularizer': self.mlp_bias_regularizer,
                       })
        return config
