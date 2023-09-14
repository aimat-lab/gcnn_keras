import keras_core as ks
from keras_core import ops
import keras_core.saving


@ks.saving.register_keras_serializable(package='kgcnn', name='shifted_softplus')
def shifted_softplus(x):
    r"""Shifted soft-plus activation function.

    Args:
        x (tf.Tensor): Single values to apply activation with ks functions.

    Returns:
        tf.Tensor: Output tensor computed as :math:`\log(e^{x}+1) - \log(2)`.
    """
    return ks.activations.softplus(x) - ops.log(ops.convert_to_tensor(2.0, dtype=x.dtype))


@ks.saving.register_keras_serializable(package='kgcnn', name='softplus2')
def softplus2(x):
    r"""Soft-plus function that is :math:`0` at :math:`x=0` , the implementation aims at avoiding overflow
    :math:`\log(e^{x}+1) - \log(2)` .

    Args:
        x (tf.Tensor): Single values to apply activation with ks functions.

    Returns:
         tf.Tensor: Output tensor computed as :math:`\log(e^{x}+1) - \log(2)`.
    """
    return ks.activations.relu(x) + ops.log(0.5 * ops.exp(-ops.abs(x)) + 0.5)


@ks.saving.register_keras_serializable(package='kgcnn', name='leaky_softplus')
def leaky_softplus(x, alpha: float = 0.05):
    r"""Leaky softplus activation function similar to :obj:`tf.nn.leaky_relu` but smooth.

    .. warning::

        The leak parameter can not be changed if 'kgcnn>leaky_softplus' is passed as activation function to a layer.
        Use :obj:`kgcnn.layers.activ` activation layers instead.

    Args:
        x (tf.Tensor): Single values to apply activation with ks functions.
        alpha (float): Leak parameter. Default is 0.05.

    Returns:
         tf.Tensor: Output tensor.
    """
    return ks.activations.softplus(x) * (1 - alpha) + alpha * x


@ks.saving.register_keras_serializable(package='kgcnn', name='leaky_relu')
def leaky_relu(x, alpha: float = 0.05):
    r"""Leaky RELU activation function.

    .. warning::

        The leak parameter can not be changed if 'kgcnn>leaky_softplus' is passed as activation function to a layer.
        Use :obj:`kgcnn.layers.activ` activation layers instead.

    Args:
        x (tf.Tensor): Single values to apply activation with ks functions.
        alpha (float): Leak parameter. Default is 0.05.

    Returns:
         tf.Tensor: Output tensor.
    """
    return ks.activations.leaky_relu(x, negative_slope=alpha)


@ks.saving.register_keras_serializable(package='kgcnn', name='swish')
def swish(x):
    """Swish activation function."""
    return ks.activations.silu(x)
