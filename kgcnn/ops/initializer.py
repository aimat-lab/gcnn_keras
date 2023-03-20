import tensorflow as tf

ks = tf.keras


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Taken from original TensorFlow implementation and copied here for static reference.

    Args:
        shape: Integer shape tuple or TF tensor shape.

    Returns:
        A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


@ks.utils.register_keras_serializable(package='kgcnn', name='glorot_orthogonal')
class GlorotOrthogonal(ks.initializers.Orthogonal):
    r"""Combining Glorot variance and Orthogonal initializer.

    Generate a weight matrix with variance according to Glorot initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are de-correlated.

    This is stated by e.g.:

        * "Reducing over-fitting in deep networks by de-correlating representations" by M. Cogswell et al. (2016)
          `<https://arxiv.org/abs/1511.06068>`_ .
        * "Dropout: a simple way to prevent neural networks from over-fitting" by N. Srivastava et al. (2014)
          `<https://dl.acm.org/doi/10.5555/2627435.2670313>`_ .
        * "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
          by A. M. Saxe et al. (2013) `<https://arxiv.org/abs/1312.6120>`_ .

    This implementation has been borrowed and slightly modified from `DimeNetPP <https://arxiv.org/abs/2011.14115>`__ .

    """

    def __init__(self, gain=1.0, seed=None, scale=1.0, mode='fan_avg'):
        super(GlorotOrthogonal, self).__init__(gain=gain, seed=seed)
        self.scale = scale
        self.mode = mode

    def __call__(self, shape, dtype=tf.float32, **kwargs):
        weight_kernel = super(GlorotOrthogonal, self).__call__(shape, dtype=dtype, **kwargs)
        # Original implementation from DimeNet.
        # assert len(shape) == 2
        # W = self.orth_init(shape, dtype)
        # W *= tf.sqrt(self.scale / ((shape[0] + shape[1]) * tf.math.reduce_variance(W)))  # scale = 2.0
        # Adapted with mode and scale chosen by class. Default values should match original version, to be used
        # for DimeNet model implementation.
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        stddev = tf.math.sqrt(scale/tf.math.reduce_variance(weight_kernel))
        weight_kernel *= stddev
        return weight_kernel

    def get_config(self):
        """Get keras config."""
        config = super(GlorotOrthogonal, self).get_config()
        config.update({"scale": self.scale, "mode": self.mode})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='he_orthogonal')
class HeOrthogonal(ks.initializers.Orthogonal):
    """Combining He variance and Orthogonal initializer.

    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks are expected to learn better
    when features are de-correlated.

    This is stated by e.g.:

        * "Reducing over-fitting in deep networks by de-correlating representations" by M. Cogswell et al. (2016)
          `<https://arxiv.org/abs/1511.06068>`_ .
        * "Dropout: a simple way to prevent neural networks from over-fitting" by N. Srivastava et al. (2014)
           `<https://dl.acm.org/doi/10.5555/2627435.2670313>`_ .
        * "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
          by A. M. Saxe et al. (2013) `<https://arxiv.org/abs/1312.6120>`_ .

    This implementation has been borrowed and slightly modified from `GemNet <https://arxiv.org/abs/2106.08903>`__ .

    """

    def __init__(self, gain=1.0, seed=None, scale=1.0, mode='fan_in'):
        super(HeOrthogonal, self).__init__(gain=gain, seed=seed)
        self.scale = scale
        self.mode = mode

    def __call__(self, shape, dtype=tf.float32, **kwargs):
        weight_kernel = super(HeOrthogonal, self).__call__(shape, dtype=dtype, **kwargs)

        # Original reference implementation was designed for kernel rank={2,3}.
        # fan_in = shape[0]
        # if len(shape) == 3:
        #     fan_in = fan_in * shape[1]
        # Tried to generalize with keras _compute_fans that is extends to convolutional kernels.
        # Although, not really meaningful in standard GNN applications.
        # Optionally use other scales.
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        weight_kernel = self._standardize(weight_kernel, shape)

        # Original reference implementation with 1/fan_in changed to scale=scale/fan_in
        # W *= tf.sqrt(1 / fan_in)  # variance decrease is addressed in the dense layers
        weight_kernel *= tf.math.sqrt(scale)

        return weight_kernel

    @staticmethod
    def _standardize(kernel, shape):
        r"""Standardize kernel over `fan_in` dimensions.

        Args:
            kernel: Kernel variable.
            shape: Shape of the kernel.
        """
        # Original doc string: Makes sure that N*Var(W) = 1 and E[W] = 0
        # From original implementation as comments.
        # eps = 1e-6
        eps = ks.backend.epsilon()
        # if len(shape) == 3:
        #     axis = [0, 1]  # last dimension is output dimension
        if len(shape) == 0:
            # Constant does not really have variance.
            # Moreover, Orthogonal initializer should throw error.
            return kernel
        if len(shape) >= 3:
            axis = [i for i in range(len(shape)-1)]
        else:
            axis = 0
        mean = tf.reduce_mean(kernel, axis=axis, keepdims=True)
        var = tf.math.reduce_variance(kernel, axis=axis, keepdims=True)
        kernel = (kernel - mean) / tf.sqrt(var + eps)
        return kernel

    def get_config(self):
        """Get keras config."""
        config = super(HeOrthogonal, self).get_config()
        config.update({"scale": self.scale, "mode": self.mode})
        return config
