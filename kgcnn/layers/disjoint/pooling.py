"""
Pooling layers for node or edge feature lists.

@author: Patrick
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class PoolingEdgesPerNode(ks.layers.Layer):
    """ 
    Pooling all edges or edgelike features per node, corresponding to node assigned by edge indexlist.
    If graphs were in batch mode, the indizes must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge-index[:,0] must be sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        **kwargs
    Input: 
        List of [edgelist,indextensor] of shape [(batch*None,F),(batch*None,2)]
        The index for segment reduction is taken to be indextensor[:,0].
    Output:
        A pooled tensor of shape (batch*<None>, <F>) which should match nodelist.
        Provided that edgeindex were correctly sorted and match each node in flatten tensor.
    """
    def __init__(self, 
                 pooling_method = "segment_mean", 
                 **kwargs):
        super(PoolingEdgesPerNode, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")
        
    def build(self, input_shape):
        super(PoolingEdgesPerNode, self).build(input_shape)
    def call(self, inputs):
        edge,edge_index = inputs
        indexlist = edge_index[:,0]
        #Apply segmented mean
        out = self._pool(edge,indexlist)
        return out
    def get_config(self):
        config = super(PoolingEdgesPerNode, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config  
        

class PoolingNodes(ks.layers.Layer):
    """
    Polling all nodes per batch. The batch assignment is given by an id- or length-tensor.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        **kwargs
    Input: 
        List of tensor [nodelist,nodelength] of shape [(batch*None,F),(batch,)]
        The batch-tensor keeps the batch assignment.
    Output:
        A list of averaged nodes matching batchdimension (batch,<F>)
    """
    def __init__(self,  
                 pooling_method = "segment_mean",
                 **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")
            
    def build(self, input_shape):
        super(PoolingNodes, self).build(input_shape)
    def call(self, inputs):
        node,len_node = inputs        
        len_shape_int = K.shape(len_node)
        batchi = tf.repeat(K.arange(0,len_shape_int[0],1),len_node)
        out = self._pool(node,batchi)
        #Output should have correct shape
        return out
    def get_config(self):
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 


class PoolingAllEdges(ks.layers.Layer):
    """
    Pooling all edges per batch. The batch assignment is given by an id- or length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        **kwargs
    Input: 
        List of tensor [egdelist,edgelength] of shape [(batch*None,F),(batch,)]
        The batch-tensor keeps the batch assignment.
    Output:
        A list of averaged edges matching batchdimension (batch,<F>)
    """
    def __init__(self,
                 pooling_method = "segment_mean",
                 **kwargs):
        super(PoolingAllEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")
            
    def build(self, input_shape):
        super(PoolingAllEdges, self).build(input_shape)
    def call(self, inputs):
        edge,len_edge = inputs
        len_shape = K.shape(len_edge)
        batchi = tf.repeat(K.arange(0,len_shape[0],1),len_edge)
        out = self._pool(edge,batchi)
        #Output already has correct shape
        return out
    def get_config(self):
        config = super(PoolingAllEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 