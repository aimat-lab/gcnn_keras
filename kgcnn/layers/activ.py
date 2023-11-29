import numpy as np
import keras as ks
# import keras_core.saving


@ks.saving.register_keras_serializable(package='kgcnn', name='LeakySoftplus')
class LeakySoftplus(ks.layers.Layer):
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
        self.alpha = self.add_weight(
            shape=tuple(),
            initializer=ks.initializers.Constant(alpha),
            dtype=self.dtype,
            trainable=self._alpha_trainable
        )

    def call(self, inputs, *args, **kwargs):
        """Forward pass.

        Args:
            inputs (Tensor): Input tenor of arbitrary shape.

        Returns:
            Tensor: Leaky soft-plus activation of inputs.
        """
        x = inputs
        return ks.activations.softplus(x) * (1 - self.alpha) + self.alpha * x

    def get_config(self):
        """Get layer config."""
        config = super(LeakySoftplus, self).get_config()
        config.update({"alpha": self._alpha_config, "trainable": self._alpha_trainable})
        return config


@ks.saving.register_keras_serializable(package='kgcnn', name='LeakyRelu')
class LeakyRelu(ks.layers.Layer):
    r"""Leaky RELU function. Equivalent to :obj:`tf.nn.leaky_relu(x,alpha)` ."""

    def __init__(self, alpha: float = 0.05, trainable: bool = False, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            alpha (float, optional): Leak parameter alpha. Default is 0.05.
            trainable (bool, optional): Whether set alpha trainable. Default is False.
        """
        super(LeakyRelu, self).__init__(**kwargs)
        self._alpha_config = float(alpha)
        self._alpha_trainable = bool(trainable)
        self.alpha = self.add_weight(
            shape=tuple(), dtype=self.dtype,
            initializer=ks.initializers.Constant(alpha),
            trainable=self._alpha_trainable
        )

    def call(self, inputs, *args, **kwargs):
        """Forward pass.

        Args:
            inputs (Tensor): Input tenor of arbitrary shape.

        Returns:
            Tensor: Leaky relu activation of inputs.
        """
        x = inputs
        return ks.activations.leaky_relu(x, alpha=self.alpha)
        # return tf.nn.relu(x) - tf.nn.relu(-x)*self.alpha

    def get_config(self):
        """Get layer config."""
        config = super(LeakyRelu, self).get_config()
        config.update({"alpha": self._alpha_config, "trainable": self._alpha_trainable})
        return config


@ks.saving.register_keras_serializable(package='kgcnn', name='Swish')
class Swish(ks.layers.Layer):
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
        self.beta = self.add_weight(
            shape=tuple(), dtype=self.dtype,
            initializer=ks.initializers.Constant(beta),
            trainable=self._beta_trainable
        )

    def call(self, inputs, *args, **kwargs):
        """Forward pass.

        Args:
            inputs (Tensor): Input tenor of arbitrary shape.

        Returns:
            Tensor: Swish activation of inputs.
        """
        x = inputs
        return x * ks.activations.sigmoid(self.beta * x)

    def get_config(self):
        """Get layer config."""
        config = super(Swish, self).get_config()
        config.update({"beta": self._beta_config, "trainable": self._beta_trainable})
        return config
