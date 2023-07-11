import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer

ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LeakySoftplus')
class LeakySoftplus(GraphBaseLayer):
    r"""Leaky softplus activation function similar to :obj:`tf.nn.leaky_relu` but smooth. """

    def __init__(self, alpha: float = 0.05, trainable: bool = False, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            alpha (float, optional): Leak parameter alpha. Default is 0.05.
            trainable (bool, optional): Whether set alpha trainable. Default is False.
        """
        super(LeakySoftplus, self).__init__(**kwargs)
        self._alpha_config = float(alpha)
        self._alpha_trainable = bool(trainable)
        self.alpha = self.add_weight(shape=None, dtype=self.dtype, trainable=self._alpha_trainable)
        self.set_weights([np.array(alpha)])

    def _activ_implementation(self, inputs, **kwargs):
        """Compute leaky_softplus activation from inputs."""
        x = inputs
        return ks.activations.softplus(x) * (1 - self.alpha) + self.alpha * x

    def call(self, inputs, *args, **kwargs):
        return self.map_values(self._activ_implementation, inputs, **kwargs)

    def get_config(self):
        config = super(LeakySoftplus, self).get_config()
        config.update({"alpha": self._alpha_config, "trainable": self._alpha_trainable})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LeakyRelu')
class LeakyRelu(GraphBaseLayer):
    r"""Leaky RELU function. Equivalent to :obj:`tf.nn.leaky_relu(x,alpha)`."""

    def __init__(self, alpha: float = 0.05, trainable: bool = False, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            alpha (float, optional): Leak parameter alpha. Default is 0.05.
            trainable (bool, optional): Whether set alpha trainable. Default is False.
        """
        super(LeakyRelu, self).__init__(**kwargs)
        self._alpha_config = float(alpha)
        self._alpha_trainable = bool(trainable)
        self.alpha = self.add_weight(shape=None, dtype=self.dtype, trainable=self._alpha_trainable)
        self.set_weights([np.array(alpha)])

    def _activ_implementation(self, inputs, **kwargs):
        """Compute leaky_relu activation from inputs."""
        x = inputs
        return tf.nn.leaky_relu(x, alpha=self.alpha)
        # return tf.nn.relu(x) - tf.nn.relu(-x)*self.alpha

    def call(self, inputs, *args, **kwargs):
        return self.map_values(self._activ_implementation, inputs, **kwargs)

    def get_config(self):
        config = super(LeakyRelu, self).get_config()
        config.update({"alpha": self._alpha_config, "trainable": self._alpha_trainable})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Swish')
class Swish(GraphBaseLayer):
    r"""Swish activation function. Computes :math:`x \; \text{sig}(\beta x)`,
    with :math:`\text{sig}(x) = 1/(1+e^{-x})`."""

    def __init__(self, beta: float = 1.0, trainable: bool = False, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            beta (float, optional): Parameter beta in sigmoid. Default is 1.0.
            trainable (bool, optional): Whether set beta trainable. Default is False.
        """
        super(Swish, self).__init__(**kwargs)
        self._beta_config = float(beta)
        self._beta_trainable = bool(trainable)
        self.beta = self.add_weight(shape=None, dtype=self.dtype, trainable=self._beta_trainable)
        self.set_weights([np.array(beta)])

    def _activ_implementation(self, inputs, **kwargs):
        """Compute swish activation from inputs."""
        x = inputs
        return x * tf.sigmoid(self.beta * x)

    def call(self, inputs, *args, **kwargs):
        return self.map_values(self._activ_implementation, inputs, **kwargs)

    def get_config(self):
        config = super(Swish, self).get_config()
        config.update({"beta": self._beta_config, "trainable": self._beta_trainable})
        return config
