import tensorflow as tf
ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='shifted_softplus')
def shifted_softplus(x):
    r"""Shifted soft-plus activation function.
    
    Args:
        x (tf.Tensor): Single values to apply activation with tf.keras functions.
    
    Returns:
        tf.Tensor: Output tensor computed as :math:`\log(e^{x}+1) - \log(2)`.
    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='softplus2')
def softplus2(x):
    r"""Soft-plus function that is :math:`0` at :math:`x=0` , the implementation aims at avoiding overflow
    :math:`\log(e^{x}+1) - \log(2)` .
    
    Args:
        x (tf.Tensor): Single values to apply activation with tf.keras functions.
    
    Returns:
         tf.Tensor: Output tensor computed as :math:`\log(e^{x}+1) - \log(2)`.
    """
    return ks.backend.relu(x) + ks.backend.log(0.5 * ks.backend.exp(-ks.backend.abs(x)) + 0.5)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='leaky_softplus')
def leaky_softplus(x, alpha: float = 0.05):
    r"""Leaky softplus activation function similar to :obj:`tf.nn.leaky_relu` but smooth.

    .. warning::

        The leak parameter can not be changed if 'kgcnn>leaky_softplus' is passed as activation function to a layer.
        Use :obj:`kgcnn.layers.activ` activation layers instead.

    Args:
        x (tf.Tensor): Single values to apply activation with tf.keras functions.
        alpha (float): Leak parameter. Default is 0.05.

    Returns:
         tf.Tensor: Output tensor.
    """
    return ks.activations.softplus(x) * (1 - alpha) + alpha * x


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='leaky_relu')
def leaky_relu(x, alpha: float = 0.05):
    r"""Leaky RELU activation function.

    .. warning::

        The leak parameter can not be changed if 'kgcnn>leaky_softplus' is passed as activation function to a layer.
        Use :obj:`kgcnn.layers.activ` activation layers instead.

    Args:
        x (tf.Tensor): Single values to apply activation with tf.keras functions.
        alpha (float): Leak parameter. Default is 0.05.

    Returns:
         tf.Tensor: Output tensor.
    """
    return tf.nn.leaky_relu(x, alpha=alpha)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='swish')
def swish(x):
    """Swish activation function."""
    return tf.keras.activations.swish(x)
