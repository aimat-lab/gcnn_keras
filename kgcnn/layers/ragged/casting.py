"""@package: Keras Layers for Graph Convolutions using ragged tensors
@author: Patrick Reiser
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



class CastRaggedToDense(tf.keras.layers.Layer):
    """
    Layer to cast a ragged tensor to tensor.
    
    Args:
        **kwargs
    
    Input:
        Ragged Tensor of shape (batch,...)
    Output:
        input.to_tensor()
    """
    def __init__(self, **kwargs):
        super(CastRaggedToDense, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(CastRaggedToDense, self).build(input_shape)
    def call(self, inputs):
        return inputs.to_tensor()
