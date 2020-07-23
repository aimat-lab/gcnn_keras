"""@package: Keras Layers for gathering nodes using ragged tensors
@author: Patrick Reiser
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



class GatherNodes(ks.layers.Layer):
    """ 
    Gathers Nodes from ragged tensor by index provided by a ragged index tensor in mini-batches.
    A Edge at index is the connection for node(index([0])) to node(index([1]))
    The feature of gathered ingoing and outgoing nodes are concatenated according to index tensor.
    
    Args:
        **kwargs : arguments for layer base class
        
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
    def __init__(self, **kwargs):
        super(GatherNodes, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        super(GatherNodes, self).build(input_shape)          
    def call(self, inputs):
        nod,edgeind = inputs
        shiftind = edgeind.values +tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        nodind = shiftind
        dens = nod.values
        g1 = tf.gather(dens,nodind[:,0])
        g2 = tf.gather(dens,nodind[:,1])
        get = tf.concat([g1,g2],axis=1)
        out = tf.RaggedTensor.from_row_splits(get,edgeind.row_splits)         
        return out     



class GatherNodesOutgoing(ks.layers.Layer):
    """ 
    Gathers Outgoing Nodes from ragged tensor by index provided by a ragged index tensor in mini-batches.
    A Edge at index is the connection for node(index([0])) to node(index([1]))
    The feature of gathered outgoing nodes are the connected nodes at index[1].
    
    Args:
        **kwargs : arguments for layer base class
        
    Example:
        out = GatherNodesOutgoing()([input_node,input_edge_index])   
    
    Input:
        [NodeList,EdgeIndex] of shape [(batch,None,F_n),(batch,None,2)] with both being ragged tensors.
        None gives the ragged dimension and F_n is the node feature dimension.
        
    Output:
        Gathered outgoing nodes with dimension of index Tensor,
        with entries at index node(index([1])) of shape (batch,None,F_n)
        the length matches the index Tensor.
    """
    def __init__(self, **kwargs):
        super(GatherNodesOutgoing, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        super(GatherNodesOutgoing, self).build(input_shape)          
    def call(self, inputs):
        nod,edgeind = inputs
        shiftind = edgeind.values +tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        nodind = shiftind
        dens = nod.values
        g2= tf.gather(dens,nodind[:,1])
        out = tf.RaggedTensor.from_row_splits(g2,edgeind.row_splits)         
        return out     
    
    
class LazyConcatenateNodes(ks.layers.Layer):
    """ 
    
    """
    def __init__(self, **kwargs):
        super(LazyConcatenateNodes, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        super(LazyConcatenateNodes, self).build(input_shape)          
    def call(self, inputs):
        node,node2 = inputs
        dens = node.values
        dens2 = node2.values
        out = tf.keras.backend.concatenate([dens,dens2],axis=-1)
        out = tf.RaggedTensor.from_row_splits(out,node.row_splits) 
        return out     

