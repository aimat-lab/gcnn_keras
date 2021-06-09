import tensorflow as tf
import tensorflow.keras as ks


def shifted_softplus(x):
    """
    Shifted softplus activation function.
    
    Args:
        x : single values to apply activation to using tf.keras functions
    
    Returns:
        out : log(exp(x)+1) - log(2)
    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)


def softplus2(x):
    """
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow
    
    Args:
        x: (Tensor) input tensor
    
    Returns:
         Tensor: output tensor
    """
    return ks.backend.relu(x) + ks.backend.log(0.5 * ks.backend.exp(-ks.backend.abs(x)) + 0.5)


class leaky_softplus(tf.keras.layers.Layer):
    """
    Leaky softplus activation function similar to leakyRELU but smooth.

    Args:
        alpha (float, optional): Leaking slope. The default is 0.3.
    """

    def __init__(self,alpha = 0.3,**kwargs):
        super(leaky_softplus, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        x = inputs
        return ks.activations.softplus(x) * (1 - self.alpha) + self.alpha * x

    def get_config(self):
        config = super(leaky_softplus, self).get_config()
        config.update({"alpha": self.alpha})
        return config



class leaky_relu(tf.keras.layers.Layer):
    """Leaky relu function: lambda of tf.nn.leaky_relu(x,alpha)

    Args:
        alpha (float, optional): leak alpha = 0.2
    """

    def __init__(self,alpha = 0.2, **kwargs):
        super(leaky_relu, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, **kwargs):

        x = inputs
        return tf.nn.leaky_relu(x, alpha=self.alpha)

    def get_config(self):
        config = super(leaky_relu, self).get_config()
        config.update({"alpha": self.alpha})
        return config


def swish(x):
    """Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"

    Args:
        x (tf.tensor): values to apply activation to using tf.keras functions

    Returns:
        tf.tensor: x*tf.sigmoid(x)
    """
    return x * tf.sigmoid(x)


# Register costum activation functions in get_custom_objects
kgcnn_custom_act = {'leaky_softplus': leaky_softplus,
                    'shifted_softplus': shifted_softplus,
                    'softplus2': softplus2,
                    "leaky_relu": leaky_relu,
                    "swish": swish}

print("Keras utils: Register custom activation: ", kgcnn_custom_act)
for k, v in kgcnn_custom_act.items():
    if k not in tf.keras.utils.get_custom_objects():
        tf.keras.utils.get_custom_objects()[k] = v
