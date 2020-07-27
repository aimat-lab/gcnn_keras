"""@package: Keras Layers for graph pooling using ragged tensors
@author: Patrick Reiser
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


    
class PoolingNodes(ks.layers.Layer):
    """
    Layer for pooling of nodefeatures over all nodes in graph. Which gives $1/n \sum_i node(i)$.
    
    Args:
        pool_method : tf.function to pool all nodes compatible with ragged tensors.
        **kwargs
    
    Inputs:
        Node ragged tensor of shape (batch,None,F_n)
    
    Outputs:
        Pooled Nodes of shape (batch,<F_n>)
    """
    def __init__(self,
                 pool_method = tf.math.reduce_mean ,
                 **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)
        self.pool_method = pool_method
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PoolingNodes, self).build(input_shape)
    def call(self, inputs):
        node = inputs
        out = self.pool_methode(node,axis=1)
        return out


class PoolingAllEdges(ks.layers.Layer):
    """
    Layer for pooling of edgefeatures over all edges in graph. Which gives $1/n \sum_{ij} edge(i,j)$.
    
    Args:
        pool_method : tf.function to pool all edges with ragged tensors.
        **kwargs
    
    Inputs:
        Edge ragged tensor of shape (batch,None,F_e)
    
    Outputs:
        Pooled edges of shape (batch,<F_e>)
    """
    def __init__(self, 
                 pool_method = tf.math.reduce_mean,
                 **kwargs):
        super(PoolingAllEdges, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.pool_method  = pool_method 
    def build(self, input_shape):
        super(PoolingAllEdges, self).build(input_shape)
    def call(self, inputs):
        edge = inputs        #Apply segmented mean
        out = self.pool_method(edge,axis=1)
        return out



class PoolingEdgesPerNode(ks.layers.Layer):
    """ 
    Layer for pooling of edgefeatures for each ingoing node in graph. Which gives $1/n \sum_{j} edge(i,j)$.
    
    Args:
        pool_method : tf.function to pool edges compatible with ragged tensors.
        **kwargs
    
    Inputs:
        [Node, edge and edgeindex] ragged tensors of shape [(batch,None,F_n),(batch,None,F_e),(batch,None,2)]
        Note that the ragged dimension of edge and edgeindex has to match. 
        And that the edgeindexlist is sorted for the first, ingoing node index to apply e.g. segment_mean
    
    Outputs:
        Pooled Nodes of shape (batch,None,<F_e>) where the ragged dimension matches the nodes.
    """
    def __init__(self, 
                 pool_method=tf.math.segment_mean,
                 **kwargs):
        super(PoolingEdgesPerNode, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
        self.pool_method  = pool_method 
    def build(self, input_shape):
        super(PoolingEdgesPerNode, self).build(input_shape)          
    def call(self, inputs):
        nod,edge,edgeind = inputs
        shiftind = edgeind.values +tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        dens = edge.values
        nodind = shiftind[:,0]
        get = pool_method(dens,nodind)
        out = tf.RaggedTensor.from_row_splits(get,nod.row_splits)       
        return out     
