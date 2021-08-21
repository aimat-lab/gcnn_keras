import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='glorot_orthogonal')
class GlorotOrthogonal(tf.keras.initializers.Orthogonal):
    """Combining Glorot and Orthogonal initializer."""

    def __init__(self, gain=1.0, seed=None, scale=1.0, mode='fan_avg'):
        super(GlorotOrthogonal, self).__init__(gain=gain, seed=seed)
        self.scale = scale
        self.mode = mode

    def __call__(self, shape, dtype=tf.float32, **kwargs):
        W = super(GlorotOrthogonal, self).__call__(shape, dtype=tf.float32, **kwargs)
        scale = self.scale
        fan_in, fan_out = self._compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        stddev = tf.math.sqrt(scale/tf.math.reduce_variance(W))
        W *= stddev
        return W

    @staticmethod
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

    def get_config(self):
        config = super(GlorotOrthogonal, self).get_config()
        config.update({"scale": self.scale, "mode": self.mode})
        return config
