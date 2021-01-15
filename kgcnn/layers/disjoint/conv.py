import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



DenseDisjoint = ks.layers.Dense

# class ConvFlatten(ks.layers.Layer):   
#     """
#     Layer to handle trainable weights. Possible ways to do it:
#         1. use add_weights and use K.dot
#         2. use keras.Dense (with optinal added first dimension)
#         3. use keras.Conv1D(kernel=1) and added first dimension
    
#     Args:
#         channels
#         use_bias (bool): True
#         activation: 'linear'
#         **kwargs
#     Input: 
#         Tensor of shape (batch*None,F) where features a given by fixed last dimension. 
#     Returns:
#         Output off NN forward pass: activ(kernel*features + bias)
#     """
#     def __init__(self, channels,
#                  use_bias=True,
#                  activation='linear', 
#                  **kwargs):
        
#         super(ConvFlatten, self).__init__(**kwargs)
#         #self.channels = channels
#         #self.use_bias = use_bias
#         #self.activation = activation
        
#         #Conv Layer from keras
#         #self.lay_conv1 = ks.layers.Conv1D(channels,kernel_size=1, activation=self.activation,use_bias=self.use_bias) 
#         self.lay_conv1 = ks.layers.Dense(channels, activation=activation,use_bias=use_bias) 
        
#     def build(self, input_shape):
#         super(ConvFlatten, self).build(input_shape)
#     def call(self, inputs):
#         node= inputs
#         #Expand dim to apply a Conv1D on the list.
#         #bn = K.expand_dims(node ,axis = 0)
#         bn = node
#         bn = self.lay_conv1(bn)
#         out = bn
#         #out = K.squeeze(bn,axis=0)
#         return out
