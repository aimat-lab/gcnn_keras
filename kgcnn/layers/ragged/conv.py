"""@package: Keras Convolutional Layers for Graph Convolutions using ragged tensors
@author: Patrick
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class DenseRagged(tf.keras.layers.Layer):
    """
    Custom Dense Layer for ragged input. The dense layer can be used as convolutional unit.
    
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
        
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
  """
    def __init__(self, 
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        
        super(DenseRagged, self).__init__(**kwargs)
                
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = ks.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.bias_initializer = ks.initializers.get(bias_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.bias_regularizer = ks.regularizers.get(bias_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)
        self.bias_constraint = ks.constraints.get(bias_constraint)
    
        self._supports_ragged_inputs = True 
        
    def build(self, input_shape):
        
        last_dim = input_shape[-1]
        
        # Add Kernel 
        self.kernel = self.add_weight( 'kernel',
                                        shape=[last_dim, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        # Add bias
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
    
        super(DenseRagged, self).build(input_shape)  #should set sef.built = True
        
    def call(self, inputs):
        outputs = tf.ragged.map_flat_values(tf.matmul,inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add,outputs, self.bias)
        
        outputs =  tf.ragged.map_flat_values(self.activation,outputs)
        return outputs    
