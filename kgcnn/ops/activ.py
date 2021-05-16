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


def leaky_softplus(alpha=0.3):
    """
    Leaky softplus activation function similar to leakyRELU but smooth.
        
    Args:
        alpha (float, optional): Leaking slope. The default is 0.3.

    Returns:
        func: lambda function of x.

    """
    return lambda x: ks.activations.softplus(x) * (1 - alpha) + alpha * x


def leaky_relu(alpha=0.2):
    """
    Leaky relu function.

    Args:
        alpha: leak alpha = 0.2

    Returns:
        func: lambda of tf.nn.leaky_relu(x,alpha)
    """
    return lambda x: tf.nn.leaky_relu(x, alpha=alpha)


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
