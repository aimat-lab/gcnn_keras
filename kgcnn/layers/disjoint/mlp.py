import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class MLPdisjoint(ks.layers.Layer):
    """
    Multilayer perceptron that consist of N dense keras layers.
        
    Args:
        mlp_units (list): Size of hidden layers for each layer.
        mlp_use_bias (list, optional): Use bias for hidden layers. Defaults to True.
        mlp_activation (list, optional): Activity identifier. Defaults to None.
        mlp_activity_regularizer (list, optional): Activity regularizer identifier. Defaults to None.
        mlp_kernel_regularizer (list, optional): Kernel regularizer identifier. Defaults to None.
        mlp_bias_regularizer (list, optional): Bias regularizer identifier. Defaults to None.
        **kwargs 

    Args:
        inputs (tf.tensor): Input tensor of shape (...,N).

    Returns:
        out (tf.tensor): MLP pass.
    """
    
    def __init__(self,
                 mlp_units,
                 mlp_use_bias = True,
                 mlp_activation = None,
                 mlp_activity_regularizer=None,
                 mlp_kernel_regularizer=None,
                 mlp_bias_regularizer=None,
                 **kwargs):
        """Init MLP as for dense."""
        super(MLPdisjoint, self).__init__(**kwargs) 
        
        if(isinstance(mlp_units,int)):
            mlp_units = [mlp_units]
        self.mlp_units = mlp_units 
        
        if
        self.mlp_use_bias = mlp_use_bias  
        
        self.mlp_activ = ks.activations.deserialize(dense_activ,custom_objects={'leaky_softplus':leaky_softplus,'shifted_softplus':shifted_softplus})
        self.mlp_activ_last_serialize = dense_activ_last
        self.mlp_activ_last = ks.activations.deserialize(dense_activ_last,custom_objects={'leaky_softplus':leaky_softplus,'shifted_softplus':shifted_softplus})
        self.mlp_activity_regularizer = ks.regularizers.get(dense_activity_regularizer)
        self.mlp_kernel_regularizer = ks.regularizers.get(dense_kernel_regularizer)
        self.mlp_bias_regularizer = ks.regularizers.get(dense_bias_regularizer)
        self.mlp_use = dropout_use
        self.mlp_dropout = dropout_dropout
        
        self.mlp_dense_activ = [ks.layers.Dense(
                                self.dense_units,
                                use_bias=self.dense_bias,
                                activation=self.dense_activ,
                                name=self.name+'_dense_'+str(i),
                                activity_regularizer= self.dense_activity_regularizer,
                                kernel_regularizer=self.dense_kernel_regularizer,
                                bias_regularizer=self.dense_bias_regularizer
                                ) for i in range(self.dense_depth-1)]


    def build(self, input_shape):
        """Build layer."""
        super(MLPdisjoint, self).build(input_shape)          
    def call(self, inputs,training=False):
        """Forward pass."""
        x = inputs
        for i in range(self.dense_depth-1):
            x = self.mlp_dense_activ[i](x)
            if(self.dropout_use == True):
                x = self.mlp_dropout(x,training=training)
        x = self.mlp_dense_last(x)
        out = x
        return out
    def get_config(self):
        """Update config."""
        config = super(MLPdisjoint, self).get_config()
        config.update({"dense_units": self.dense_units,
                       'dense_depth': self.dense_depth,
                       'dense_bias': self.dense_bias,
                       'dense_bias_last': self.dense_bias_last,
                       'dense_activ' : self.dense_activ_serialize,
                       'dense_activ_last' : self.dense_activ_last_serialize,
                       'dense_activity_regularizer': ks.regularizers.serialize(self.dense_activity_regularizer),
                       'dense_kernel_regularizer': ks.regularizers.serialize(self.dense_kernel_regularizer),
                       'dense_bias_regularizer': ks.regularizers.serialize(self.dense_bias_regularizer),
                       'dropout_use': self.dropout_use,
                       'dropout_dropout': self.dropout_dropout
                       })
        return config