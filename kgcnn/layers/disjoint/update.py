import tensorflow as tf
import tensorflow.keras as ks


class ApplyMessage(ks.layers.Layer):
    """
    Apply message by edge matrix multiplication.
    
    The message dimension must be suitable for matrix multiplication.
    
    Args:
        target_shape (int): Target dimension. Message dimension must match target_shape*node_shape.
    """
    def __init__(self,target_shape,**kwargs):
        """Initialize layer."""
        super(ApplyMessage, self).__init__(**kwargs) 
        self.target_shape = target_shape
    def build(self, input_shape):
        """Build layer."""
        super(ApplyMessage, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs list of [message, nodes]
       
        Args:
            message (tf.tensor): Message tensor flattened that can be reshaped to (batch*None,target_shape,node_shape)
            nodes (tf.tensor): Node feature list of shape (batch*None,F)
            
        Returns:
            node_updates (tf.tensor): Element-wise matmul of message and node features of output shape (batch,target_shape)
        """
        dens_e,dens_n = inputs
        dens_m = tf.reshape(dens_e,(ks.backend.shape(dens_e)[0],self.target_shape,ks.backend.shape(dens_n)[-1]))
        out = tf.keras.backend.batch_dot(dens_m,dens_n) 
        return out     
    def get_config(self):
        """Update layer config."""
        config = super(ApplyMessage, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config 


class GRUupdate(ks.layers.Layer):
    """
    Gated recurrent unit update.
    
    Args:
        units (int): Units for GRU.
    """
    
    def __init__(self,units,**kwargs):
        """Initialize layer."""
        super(GRUupdate, self).__init__(**kwargs) 
        self.units = units
        self.gru = tf.keras.layers.GRUCell(units)
    def build(self, input_shape):
        """Build layer."""
        #self.gru.build(channels)
        super(GRUupdate, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs list of [nodes, updates]
        
        Args:
            nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            updates (tf.tensor): Matching node updates of same shape (batch*None,F)
            
        Returns:
            updated_nodes (tf.tensor): Updated nodes of shape (batch*None,F)
        """
        n,eu = inputs
        # Apply GRU for update node state
        out,_ = self.gru(eu,[n])
        return out     
    def get_config(self):
        """Update layer config."""
        config = super(GRUupdate, self).get_config()
        config.update({"units": self.units})
        return config 