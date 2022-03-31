import tensorflow as tf
ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='shifted_softplus')
def shifted_softplus(x):
    r"""Shifted softplus activation function.
    
    Args:
        x (tf.Tensor): Single values to apply activation with tf.keras functions.
    
    Returns:
        tf.Tensor: Output tensor computed as :math:`\log(e^{x}+1) - \log(2)`.
    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='softplus2')
def softplus2(x):
    r"""Softplus function that is :math:`0` at :math:`x=0`, the implementation aims at avoiding overflow
    :math:`\log(e^{x}+1) - \log(2)`.
    
    Args:
        x (tf.Tensor): Single values to apply activation with tf.keras functions.
    
    Returns:
         tf.Tensor: Output tensor computed as :math:`\log(e^{x}+1) - \log(2)`.
    """
    return ks.backend.relu(x) + ks.backend.log(0.5 * ks.backend.exp(-ks.backend.abs(x)) + 0.5)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='leaky_softplus')
class leaky_softplus(tf.keras.layers.Layer):
    r"""Leaky softplus activation function similar to :obj:`tf.nn.leaky_relu` but smooth. """

    def __init__(self, alpha=0.05, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            alpha (float, optional): Leak parameter alpha. Default is 0.05.
        """
        super(leaky_softplus, self).__init__(**kwargs)
        # self.alpha = self.add_weight(shape=None, dtype=self.dtype, trainable=trainable)
        # self.set_weights([np.array(alpha)])
        self.alpha = float(alpha)

    def call(self, inputs, **kwargs):
        """Compute leaky_softplus activation from inputs."""
        x = inputs
        return ks.activations.softplus(x) * (1 - self.alpha) + self.alpha * x

    def get_config(self):
        config = super(leaky_softplus, self).get_config()
        config.update({"alpha": self.alpha})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='leaky_relu')
class leaky_relu(tf.keras.layers.Layer):
    r"""Leaky RELU function. Equivalent to :obj:`tf.nn.leaky_relu(x,alpha)`."""

    def __init__(self, alpha: float = 0.05, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            alpha (float, optional): Leak parameter alpha. Default is 0.05.
        """
        super(leaky_relu, self).__init__(**kwargs)
        self.alpha = float(alpha)

    def call(self, inputs, **kwargs):
        """Compute leaky_relu activation from inputs."""
        x = inputs
        return tf.nn.leaky_relu(x, alpha=self.alpha)
        # return tf.nn.relu(x) - tf.nn.relu(-x)*self.alpha

    def get_config(self):
        config = super(leaky_relu, self).get_config()
        config.update({"alpha": self.alpha})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='swish')
class swish(tf.keras.layers.Layer):
    r"""Swish activation function. Computes :math:`x \; \text{sig}(\beta x)`,
    with :math:`\text{sig}(x) = 1/(1+e^{-x})`."""

    def __init__(self, beta: float = 1.0, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            beta (float, optional): Parameter beta in sigmoid. Default is 1.0.
        """
        super(swish, self).__init__(**kwargs)
        # self.beta = self.add_weight(shape=None, dtype=self.dtype, trainable=trainable)
        # self.set_weights([np.array(beta)])
        self.beta = float(beta)

    def call(self, inputs, **kwargs):
        """Compute swish activation from inputs."""
        x = inputs
        return x * tf.sigmoid(self.beta*x)

    def get_config(self):
        config = super(swish, self).get_config()
        config.update({"beta": self.beta})
        return config
