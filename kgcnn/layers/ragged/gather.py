"""@package: Keras Layers for gathering nodes using ragged tensors
@author: Patrick
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



class GatherNodes(ks.layers.Layer):
    """ 
    Gathers Nodes from ragged tensor by index provided by a ragged index tensor in mini-batches.
    A edge at index is the connection for node(index([0])) to node(index([1]))
    The feature of gathered ingoing and outgoing nodes are concatenated according to index tensor.
    
    Args:
        ragged_validate (bool): False
        **kwargs
        
    Example:
        out = GatherNodes()([input_node,input_edge_index])   
    
    Input:
        [NodeList,EdgeIndex] of shape [(batch,None,F_n),(batch,None,2)] with both being ragged tensors.
        None gives the ragged dimension and F_n is the node feature dimension.
        
    Output:
        Gathered nodes with dimension of index Tensor,
        with entries at index (node(index([0])),node(index([1]))) of shape (batch,None,F_n+F_n)
        the length matches the index Tensor.
    """
    def __init__(self, 
                 ragged_validate = False,
                 **kwargs):
        super(GatherNodes, self).__init__(**kwargs) 
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        super(GatherNodes, self).build(input_shape)          
    def call(self, inputs):
        nod,edgeind = inputs
        shift1 = edgeind.values
        shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)  
        dens = nod.values
        g1 = tf.gather(dens,shiftind[:,0])
        g2 = tf.gather(dens,shiftind[:,1])
        get = tf.concat([g1,g2],axis=1)
        out = tf.RaggedTensor.from_row_splits(get,edgeind.row_splits,validate=self.ragged_validate)         
        return out     
    def get_config(self):
        config = super(GatherNodes, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        return config 


class GatherNodesOutgoing(ks.layers.Layer):
    """ 
    Gathers Outgoing Nodes from ragged tensor by index provided by a ragged index tensor in mini-batches.
    A Edge at index is the connection for node(index([0])) to node(index([1]))
    The feature of gathered outgoing nodes are the connected nodes at index[1].
    
    Args:
        ragged_validate (bool): False
        **kwargs
        
    Example:
        out = GatherNodesOutgoing()([input_node,input_edge_index])   
    
    Input:
        [NodeList,EdgeIndex] of shape [(batch,None,F_n),(batch,None,2)] with both being ragged tensors.
        None represents the ragged dimension and F_n is the node feature dimension.
        
    Output:
        Gathered outgoing nodes with dimension of index Tensor,
        with entries at index node(index([1])) of shape (batch,None,F_n)
        The length matches the index Tensor.
    """
    def __init__(self, 
                 ragged_validate = False,
                 **kwargs):
        super(GatherNodesOutgoing, self).__init__(**kwargs) 
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        super(GatherNodesOutgoing, self).build(input_shape)          
    def call(self, inputs):
        nod,edgeind = inputs
        shift1 = edgeind.values
        shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)       
        nodind = shiftind
        dens = nod.values
        g2= tf.gather(dens,nodind[:,1])
        out = tf.RaggedTensor.from_row_splits(g2,edgeind.row_splits,validate=self.ragged_validate)         
        return out  
    def get_config(self):
        config = super(GatherNodesOutgoing, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        return config 
  
    
class GatherNodesIngoing(ks.layers.Layer):
    """ 
    Gathers Ingoing Nodes from ragged tensor by index provided by a ragged index tensor in mini-batches.
    A Edge at index is the connection for node(index([0])) to node(index([1]))
    The feature of gathered ingoing nodes at index[0] for the edges in edge tensor.
    
    Args:
        ragged_validate (bool): False
        **kwargs
        
    Example:
        out = GatherNodesIngoing()([input_node,input_edge_index])   
    
    Input:
        [NodeList,EdgeIndex] of shape [(batch,None,F_n),(batch,None,2)] with both being ragged tensors.
        None gives the ragged dimension and F_n is the node feature dimension.
        
    Output:
        Gathered ingoing nodes with dimension of index Tensor,
        with entries at index node(index([0])) of shape (batch,None,F_n)
        The length matches the index Tensor.
    """
    def __init__(self,
                 ragged_validate = False,
                 **kwargs):
        super(GatherNodesIngoing, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True  
        self.ragged_validate = ragged_validate        
    def build(self, input_shape):
        super(GatherNodesIngoing, self).build(input_shape)          
    def call(self, inputs):
        nod,edgeind = inputs
        shift1 = edgeind.values
        shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        nodind = shiftind
        dens = nod.values
        g1= tf.gather(dens,nodind[:,0])
        out = tf.RaggedTensor.from_row_splits(g1,edgeind.row_splits,validate=self.ragged_validate)         
        return out  
    def get_config(self):
        config = super(GatherNodesIngoing, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        return config 
    
class GatherState(ks.layers.Layer):
    """ 
    Gathers a global state for nodes or edges.
    
    Args:
        ragged_validate (bool): False
        **kwargs
        
    Example:
        out = GatherState()([state_node,input_edge])   
    
    Input:
        List of state and ragged node/edgelist [State,Node/Edge] of shape [(batch,F_s),(batch,None,F_n)]
        Here state is simply a tensor and node/edgelist is a ragged tensor.
        
    Return:
        A tensor with shape (batch,None,F_s)
    """
    def __init__(self,
                 ragged_validate = False,
                 **kwargs):
        super(GatherState, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True
        self.ragged_validate = ragged_validate          
    def build(self, input_shape):
        super(GatherState, self).build(input_shape)          
    def call(self, inputs):
        env,nod= inputs
        target_len = nod.row_lengths()
        out = tf.repeat(env,target_len,axis=0)
        out = tf.RaggedTensor.from_row_splits(out,nod.row_splits,validate=self.ragged_validate)
        return out           
    def get_config(self):
        config = super(GatherState, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        return config 

    
class LazyConcatenateNodes(ks.layers.Layer):
    """ 
    Concatenate ragged nodetensors without checking shape. Ragged dimension only at first axis.
    
    Args:
        axis (int): Axis to concatenate. Default is -1.
        ragged_validate (bool): False
        **kwargs
    
    Input:
        [NodeList1,NodeList2,...] of shape [(batch,None,F_n),(batch,None,F_n),...]
        Ragged tensors of nodes with similar ragged dimension. 
    
    Output:
        Concatenated Nodes with shape (batch,None,sum(F)) where the row_splits of first nodelist are kept.
    """
    def __init__(self,
                 axis = -1,
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(LazyConcatenateNodes, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True 
        self.ragged_validate = ragged_validate   
        self.axis = axis
    def build(self, input_shape):
        """Build layer."""
        super(LazyConcatenateNodes, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        out = tf.keras.backend.concatenate([x.values for x in inputs],axis=self.axis)
        out = tf.RaggedTensor.from_row_splits(out,inputs[0].row_splits,validate=self.ragged_validate) 
        return out     
    def get_config(self):
        """Update config."""
        config = super(LazyConcatenateNodes, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"axis": self.axis})
        return config 
