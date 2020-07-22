"""@package: Keras Layers for Graph Convolutions using ragged tensors
@author: Patrick Reiser
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



class CastRaggedToDense(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CastRaggedToDense, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(CastRaggedToDense, self).build(input_shape)
    def call(self, inputs):
        return inputs.to_tensor()
