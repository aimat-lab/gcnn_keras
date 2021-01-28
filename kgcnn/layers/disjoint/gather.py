import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class GatherNodes(ks.layers.Layer):
    """
    Gather nodes by edge indexlist. Indexlist must match flatten nodes.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    
    Args:
        **kwargs
        
    Input:
        List of tensors [node,edge_index]
        node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
        edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
        
    Output:
        features (tf.tensor): Gathered node features of (ingoing,outgoing) nodes.        
                              Output shape is (batch*None,F+F).
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherNodes, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodes, self).build(input_shape)          
    def call(self, inputs):
        """Forward path."""
        node,edge_index = inputs
        indexlist = edge_index 
        node1Exp = tf.gather(node,indexlist[:,0],axis=0)
        node2Exp = tf.gather(node,indexlist[:,1],axis=0)
        out = K.concatenate([node1Exp,node2Exp],axis=1)
        return out     
    

class GatherNodesOutgoing(ks.layers.Layer):
    """
    Gather nodes by edge indexlist. Indexlist must match flatten nodes.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    For outgoing nodes, layer uses only index[1].
    
    Args:
        **kwargs
        
    Input: 
        List of tensors [node,edge_index]
        node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
        edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
                                For ingoing gather nodes according to index[1]
    
    Output:
        features (tf.tensor): A list of gathered outgoing node features from indexlist.        
                              Output shape is (batch*None,F).
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherNodesOutgoing, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesOutgoing, self).build(input_shape)          
    def call(self, inputs):
        """Forward path."""
        node,edge_index = inputs
        indexlist = edge_index 
        out = tf.gather(node,indexlist[:,1],axis=0)
        return out     


class GatherNodesIngoing(ks.layers.Layer):
    """
    Gather nodes by edge indexlist. Indexlist must match flatten nodes. 
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    For ingoing nodes, layer uses only index[0].
    
    Args:
        **kwargs
    
    Input:
        List of tensors [node,edge_index]
        node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
        edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
                                For ingoing gather nodes according to index[0]
    
    Output:
        features (tf.tensor): A list of gathered ingoing node features from indexlist.        
                              Output shape is (batch*None,F).
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherNodesIngoing, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesIngoing, self).build(input_shape)          
    def call(self, inputs):
        """Forward path."""
        node,edge_index = inputs
        indexlist = edge_index 
        out = tf.gather(node,indexlist[:,0],axis=0)
        return out     

    
class GatherState(ks.layers.Layer):
    """
    Layer to repeat environment or global state for node or edge lists. The node or edge lists are flattened.
    
    To repeat the correct environment for each sample, a tensor with the target length is required.

    Args:
        **kwargs
    
    Input:
        List of tensors [environment,target_length]
        environment (tf.tensor): List of graph specific feature tensor of shape (batch*None,F)
        target_length (tf.tensor): Number of nodes or edges in each graph of shape (batch,)
                                   This can be either node or edge specific.

    Return:
        features (tf.tensor): A tensor with repeated single state for each graph.
                              Output shape is (batch*N,F).
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)          
    def call(self, inputs):
        """Forward path."""
        env,target_len = inputs
        out = tf.repeat(env,target_len,axis=0)
        return out     
    