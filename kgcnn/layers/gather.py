"""
Layers to gather node and edge features from indexlist.

@author: Patrick
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class GatherNodes(ks.layers.Layer):
    """ 
    Gather nodes by edge indexlist. Indexlist must match flatten nodes. 
    If graphs were in batch mode, the indizes must be corrected for disjoint graphs.
    
    Args:
        **kwargs
    Input: 
        List tensor of node and matching indexlist [node,index]
        Inputshape is expected to be [(None,F), (None,2)]
    Output:
        A list of gathered nodefeatures from indexlist.
        Shape is (None,F+F)
    """
    def __init__(self, **kwargs):
        super(GatherNodes, self).__init__(**kwargs)          
    def build(self, input_shape):
        super(GatherNodes, self).build(input_shape)          
    def call(self, inputs):
        node,edge_index = inputs
        indexlist = edge_index 
        node1Exp = tf.gather(node,indexlist[:,0],axis=0)
        node2Exp = tf.gather(node,indexlist[:,1],axis=0)
        out = K.concatenate([node1Exp,node2Exp],axis=1)
        return out     
    

class GatherNodesOutgoing(ks.layers.Layer):
    """ 
    Gather nodes by edge indexlist. Indexlist must match flatten nodes. 
    If graphs were in batch mode, the indizes must be corrected for disjoint graphs.
    For outgoing nodes, layer uses only indexlist[1].
    
    Args:
        **kwargs
    Input: 
        List tensor of node and matching indexlist [node,index]
        Inputshape is expected to be [(None,F), (None,2)]
        For ingoing gather nodes according to index[1]
    Output:
        A list of gathered nodefeatures from indexlist.
        Shape is (None,F)
    """
    def __init__(self, **kwargs):
        super(GatherNodesOutgoing, self).__init__(**kwargs)          
    def build(self, input_shape):
        super(GatherNodesOutgoing, self).build(input_shape)          
    def call(self, inputs):
        node,edge_index = inputs
        indexlist = edge_index 
        out = tf.gather(node,indexlist[:,1],axis=0)
        return out     


class GatherNodesIngoing(ks.layers.Layer):
    """
    Gather nodes by edge indexlist. Indexlist must match flatten nodes. 
    If graphs were in batch mode, the indizes must be corrected for disjoint graphs.
    For ingoing nodes, layer uses only indexlist[0].
    
    Args:
        **kwargs
    Input: 
        List tensor of node and matching indexlist [node,index]
        Inputshape is expected to be [(None,F), (None,2)]
        For ingoing gather nodes according to index[1]
    Output:
        A list of gathered nodefeatures from indexlist.
        Shape is (None,F)
    """
    def __init__(self, **kwargs):
        super(GatherNodesIngoing, self).__init__(**kwargs)          
    def build(self, input_shape):
        super(GatherNodesIngoing, self).build(input_shape)          
    def call(self, inputs):
        node,edge_index = inputs
        indexlist = edge_index 
        out = tf.gather(node,indexlist[:,0],axis=0)
        return out     

    
class GatherState(ks.layers.Layer):
    """
    Layer to repeat environment or global state for node or edge lists. The node or edge lists are flattened.
    To repeat the correct environment for eachs sample, a tensor with the target lengths is required.

    Args:
        **kwargs
    Input: 
        List of feature tensor plus node or edge length of the target list [environment,target_length].
        Input is expected to have shape [(batch,F),(batch,)]
    Return:
        A tensor with shape (batch*N,F)
    """
    def __init__(self, **kwargs):
        super(GatherState, self).__init__(**kwargs)          
    def build(self, input_shape):
        super(GatherState, self).build(input_shape)          
    def call(self, inputs):
        env,target_len = inputs
        out = tf.repeat(env,target_len,axis=0)
        return out     
    