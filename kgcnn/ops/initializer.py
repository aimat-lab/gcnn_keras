import tensorflow as tf
import tensorflow.keras as ks

@tf.keras.utils.register_keras_serializable(package='kgcnn',name='glorot_orthogonal')
class GlorotOrthogonal(tf.initializers.Initializer):
    """
    @TODO: Generalize and inherit from Orthogonal
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """

    def __init__(self, scale=2.0, seed=None):
        super().__init__()
        self.orth_init = tf.initializers.Orthogonal(seed=seed)
        self.scale = scale

    def __call__(self, shape, dtype=tf.float32, **kwargs):
        assert len(shape) == 2
        W = self.orth_init(shape, dtype)
        W *= tf.sqrt(self.scale / ((shape[0] + shape[1]) * tf.math.reduce_variance(W)))
        return W
