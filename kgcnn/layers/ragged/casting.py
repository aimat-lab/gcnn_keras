"""@package: Keras Layers for Graph Convolutions using ragged tensors
@author: Patrick
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


class CastRaggedToFlatten(ks.layers.Layer):
    """ Cast a ragged tensor input to a value plus row_length tensor.
    Args:
        **kwargs
    
    Input:
        inputs: A ragged Tensor (tf.ragged)
    
    Output:
        A tuple [Values,Row_length] extracted from the ragged tensor
    """
    def __init__(self, **kwargs):
        super(CastRaggedToFlatten, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(CastRaggedToFlatten, self).build(input_shape)
    def call(self, inputs):
        tens = inputs
        flat_tens = tens.values
        row_lengths = tens.row_lengths()
        return (flat_tens,row_lengths)