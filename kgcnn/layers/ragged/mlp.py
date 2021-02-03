import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from kgcnn.utils.activ import kgcnn_custom_act 
from kgcnn.layers.ragged.conv import DenseRagged


class MLPRagged(ks.layers.Layer):
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
        super(MLPRagged, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True
        
        # Make to one element list
        if(isinstance(mlp_units,int)):
            mlp_units = [mlp_units] 
        if(isinstance(mlp_use_bias,list) == False and isinstance(mlp_use_bias,tuple)==False):
            mlp_use_bias = [mlp_use_bias for i in mlp_units ] 
        if(isinstance(mlp_activation,list) == False and isinstance(mlp_activation,tuple)==False):
            mlp_activation = [mlp_activation for i in mlp_units ] 
        if(isinstance(mlp_activity_regularizer,list) == False and isinstance(mlp_activity_regularizer,tuple)==False):
            mlp_activity_regularizer = [mlp_activity_regularizer for i in mlp_units ] 
        if(isinstance(mlp_kernel_regularizer,list) == False and isinstance(mlp_kernel_regularizer,tuple)==False):
            mlp_kernel_regularizer = [mlp_kernel_regularizer for i in mlp_units]
        if(isinstance(mlp_bias_regularizer,list) == False and isinstance(mlp_bias_regularizer,tuple)==False):
            mlp_bias_regularizer = [mlp_bias_regularizer for i in mlp_units ]
        
        # Serialized props
        self.mlp_units = mlp_units 
        self.mlp_use_bias = mlp_use_bias
        self.mlp_activation = [x if isinstance(x,str) or isinstance(x,dict) else ks.activations.serialize(x) for x in mlp_activation]
        self.mlp_activity_regularizer = [x if isinstance(x,str) or isinstance(x,dict) else ks.regularizers.serialize(x) for x in mlp_activity_regularizer]
        self.mlp_kernel_regularizer = [x if isinstance(x,str) or isinstance(x,dict) else ks.regularizers.serialize(x) for x in mlp_kernel_regularizer]
        self.mlp_bias_regularizer = [x if isinstance(x,str) or isinstance(x,dict) else ks.regularizers.serialize(x) for x in mlp_bias_regularizer]
          
        # Deserialized props
        self.deserial_mlp_activation = [ks.activations.deserialize(x,custom_objects=kgcnn_custom_act) for x in self.mlp_activation]
        self.deserial_mlp_activity_regularizer =[ ks.regularizers.deserialize(x) for x in self.mlp_activity_regularizer]
        self.deserial_mlp_kernel_regularizer = [ks.regularizers.deserialize(x) for x in self.mlp_kernel_regularizer ]
        self.deserial_mlp_bias_regularizer = [ks.regularizers.deserialize(x)  for x in self.mlp_bias_regularizer]
        
        self.mlp_dense_list = [DenseRagged(
                                self.mlp_units[i],
                                use_bias=self.mlp_use_bias[i],
                                name=self.name+'_dense_'+str(i),
                                activation=self.deserial_mlp_activation[i],
                                activity_regularizer=self.deserial_mlp_activity_regularizer[i],
                                kernel_regularizer=self.deserial_mlp_kernel_regularizer[i],
                                bias_regularizer=self.deserial_mlp_bias_regularizer[i]
                                ) for i in range(len(self.mlp_units))]


    def build(self, input_shape):
        """Build layer."""
        super(MLPRagged, self).build(input_shape)          
    def call(self, inputs,training=False):
        """Forward pass.
        
        Args:
            inputs (tf.ragged): Input ragged tensor of shape (...,N).
    
        Returns:
            tf.ragged: MLP pass.
        """
        x = inputs
        for i in range(len(self.mlp_units)):
            x = self.mlp_dense_list[i](x)
        out = x
        return out
    def get_config(self):
        """Update config."""
        config = super(MLPRagged, self).get_config()
        config.update({"mlp_units": self.mlp_units,
                       'mlp_use_bias': self.mlp_use_bias,
                       'mlp_activation' : self.mlp_activation,
                       'mlp_activity_regularizer': self.mlp_activity_regularizer,
                       'mlp_kernel_regularizer': self.mlp_kernel_regularizer,
                       'mlp_bias_regularizer': self.mlp_bias_regularizer,
                       })
        return config