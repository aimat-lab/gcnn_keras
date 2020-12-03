"""@package: Keras Layers for graph pooling using ragged tensors
@author: Patrick
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


    
class PoolingNodes(ks.layers.Layer):
    """
    Layer for pooling of nodefeatures over all nodes in graph. Which gives $1/n \sum_i node(i)$.
    
    Args:
        pooling_method : tf.function to pool all nodes compatible with ragged tensors.
        **kwargs
    
    Inputs:
        Node ragged tensor of shape (batch,None,F_n)
    
    Outputs:
        Pooled Nodes of shape (batch,<F_n>)
    """
    def __init__(self,
                 pooling_method = "reduce_mean" ,
                 **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
        if(self.pooling_method == "reduce_mean"):
            self._pool = tf.math.reduce_mean
        elif(self.pooling_method == "reduce_sum"):
            self._pool = tf.math.reduce_sum
        else:
            raise TypeError("Unknown pooling, choose: reduce_mean, reduce_sum, ...")
        
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PoolingNodes, self).build(input_shape)
    def call(self, inputs):
        node = inputs
        out = self._pool(node,axis=1)
        return out
    def get_config(self):
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 
    

class PoolingAllEdges(ks.layers.Layer):
    """
    Layer for pooling of edgefeatures over all edges in graph. Which gives $1/n \sum_{ij} edge(i,j)$.
    
    Args:
        pooling_method : tf.function to pool all edges with ragged tensors.
        **kwargs
    
    Inputs:
        Edge ragged tensor of shape (batch,None,F_e)
    
    Outputs:
        Pooled edges of shape (batch,<F_e>)
    """
    def __init__(self, 
                 pooling_method = tf.math.reduce_mean,
                 **kwargs):
        super(PoolingAllEdges, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.pooling_method  = pooling_method 
        
        if(self.pooling_method == "reduce_mean"):
            self._pool = tf.math.reduce_mean
        elif(self.pooling_method == "reduce_sum"):
            self._pool = tf.math.reduce_sum
        else:
            raise TypeError("Unknown pooling, choose: reduce_mean, reduce_sum, ...")
        
    def build(self, input_shape):
        super(PoolingAllEdges, self).build(input_shape)
    def call(self, inputs):
        edge = inputs        #Apply segmented mean
        out = self._pool(edge,axis=1)
        return out
    def get_config(self):
        config = super(PoolingAllEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 


class PoolingEdgesPerNode(ks.layers.Layer):
    """ 
    Layer for pooling of edgefeatures for each ingoing node in graph. Which gives $1/n \sum_{j} edge(i,j)$.
    
    Args:
        pooling_method (str): tf.function to pool edges compatible with ragged tensors.
        ragged_validate (bool): False
        **kwargs
    
    Inputs:
        [node, edge and edgeindex] ragged tensors of shape [(batch,None,F_n),(batch,None,F_e),(batch,None,2)]
        Note that the ragged dimension of edge and edgeindex has to match. 
        And that the edgeindexlist is sorted for the first, ingoing node index to apply e.g. segment_mean
    
    Outputs:
        Pooled Nodes of shape (batch,None,<F_e>) where the ragged dimension matches the nodes.
    """
    
    def __init__(self, 
                 pooling_method="segment_mean",
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingEdgesPerNode, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
        self.pooling_method  = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: segment_mean, segment_sum, ...")
        
        self.ragged_validate = ragged_validate
    def build(self, input_shape):
        """Build layer."""
        super(PoolingEdgesPerNode, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        nod,edge,edgeind = inputs
        shift1 = edgeind.values
        shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        dens = edge.values
        nodind = shiftind[:,0]
        get = self._pool(dens,nodind)
        out = tf.RaggedTensor.from_row_splits(get,nod.row_splits,validate=self.ragged_validate)       
        return out     
    def get_config(self):
        config = super(PoolingEdgesPerNode, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        config.update({"ragged_validate": self.ragged_validate})
        return config


class PoolingWeightedEdgesPerNode(ks.layers.Layer):
    """ 
    Layer for pooling of edgefeatures for each ingoing node in graph. Which gives $1/n \sum_{j} edge(i,j)$.
    
    Args:
        pooling_method (str): tf.function to pool edges compatible with ragged tensors.
        ragged_validate (bool): False
        **kwargs
    
    Inputs:
        [node, edge, edgeindex, weight] ragged tensors of shape [(batch,None,F_n),(batch,None,F_e),(batch,None,2),(batch,None,1)]
        Note that the ragged dimension of edge and edgeindex and weight has to match. 
        And that the edgeindexlist is sorted for the first, ingoing node index to apply e.g. segment_mean
        The weight is the entry in the ajacency matrix for the edges in the list and must be broadcasted or match in dimension.
    
    Outputs:
        Pooled Nodes of shape (batch,None,<F_e>) where the ragged dimension matches the nodes.     
    """
    
    def __init__(self, 
                 pooling_method="segment_mean",
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingEdgesPerNode, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
        self.pooling_method  = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: segment_mean, segment_sum, ...")
        
        self.ragged_validate = ragged_validate
    def build(self, input_shape):
        """Build layer."""
        super(PoolingEdgesPerNode, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        nod,edge,edgeind,weights = inputs
        shift1 = edgeind.values
        shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        dens = edge.values * weights.values
        nodind = shiftind[:,0]
        get = self._pool(dens,nodind)
        out = tf.RaggedTensor.from_row_splits(get,nod.row_splits,validate=self.ragged_validate)       
        return out     
    def get_config(self):
        """Update layer config."""
        config = super(PoolingEdgesPerNode, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        config.update({"ragged_validate": self.ragged_validate})
        return config  