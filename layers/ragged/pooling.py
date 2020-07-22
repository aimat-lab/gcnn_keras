"""@package: Keras Layers for graph pooling using ragged tensors
@author: Patrick Reiser
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


    
class PoolingNodes(ks.layers.Layer):
    """
    Layer for averaging of Nodefeatures over nodes in graph. Which gives 1/N \sum_i node(i)
    """
    def __init__(self, **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PoolingNodes, self).build(input_shape)
    def call(self, inputs):
        node = inputs
        out = tf.math.reduce_mean(node,axis=1)
        return out


class PoolingAllEdges(ks.layers.Layer):
    """
    Layer for averaging of edgefeatures for all edges. Which gives 1/M \sum_ij edge(i,j)
    """
    def __init__(self, **kwargs):
        super(PoolingAllEdges, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PoolingAllEdges, self).build(input_shape)
    def call(self, inputs):
        edge = inputs        #Apply segmented mean
        out = tf.math.reduce_mean(edge,axis=1)
        return out



class PoolingEdgesPerNode(ks.layers.Layer):
    """ 
    
    """
    def __init__(self, **kwargs):
        super(PoolingEdgesPerNode, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        super(PoolingEdgesPerNode, self).build(input_shape)          
    def call(self, inputs):
        nod,edge,edgeind = inputs
        shiftind = edgeind.values +tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        dens = edge.values
        nodind = shiftind[:,0]
        get = tf.math.segment_mean(dens,nodind)
        out = tf.RaggedTensor.from_row_splits(get,nod.row_splits)       
        return out     
