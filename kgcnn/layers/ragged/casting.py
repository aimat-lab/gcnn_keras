"""@package: Keras Layers for Graph Convolutions using ragged tensors
@author: Patrick, 
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class CastRaggedToDense(tf.keras.layers.Layer):
    """
    Layer to cast a ragged tensor to tensor.
    
    Args:
        **kwargs
    
    Input:
        Ragged Tensor of shape (batch,...)
    Output:
        input.to_tensor()
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastRaggedToDense, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToDense, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        return inputs.to_tensor()



class CastRaggedToList(ks.layers.Layer):
    """
    Cast a ragged tensor input to a value plus row_length tensor.
    
    Args:
        **kwargs
    
    Input:
        inputs: A ragged Tensor (tf.ragged)
    
    Output:
        A tuple [values,row_length] extracted from the ragged tensor
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastRaggedToList, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToList, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        tens = inputs
        flat_tens = tens.values
        row_lengths = tens.row_lengths()
        return (flat_tens,row_lengths)



class CastAdjacencyMatrixToRaggedList(ks.layers.Layer):
    """
    Cast a sparse batched adjacency matrices to a ragged index list plus connection weights.
    
    Args:
        sort_index (bool): If indices are sorted in sparse matrix.
        ragged_validate (bool): Validate ragged tensor.
        **kwargs
    
    Input:
        A sparse Tensor (tf.sparse) of shape (batch,N_max,N_max).
        The sparse tensor has then the shape of maximum nuber of nodes in the batch.
    
    Output:
        A tuple [edge_index,edge_weight] of both ragged tensors.
    """
    
    def __init__(self,sort_index = False,ragged_validate=False ,**kwargs):
        """Initialize layer."""
        super(CastAdjacencyMatrixToRaggedList, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.sort_index = sort_index
        self.ragged_validate = ragged_validate
    def build(self, input_shape):
        """Build layer."""
        super(CastAdjacencyMatrixToRaggedList, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        indexlist = inputs.indices
        valuelist = inputs.values
        if(self.sort_index==True):
            #Sort batch-dimension
            batch_order = tf.argsort(indexlist[:,0],axis=0,direction='ASCENDING',stable=True)
            indexlist = tf.gather(indexlist,batch_order,axis=0)
            valuelist = tf.gather(valuelist,batch_order,axis=0)
            batch_length = tf.math.segment_sum(tf.ones_like(indexlist[:,0]),indexlist[:,0])
            batch_splits = tf.cumsum(batch_length,exclusive=True)
            #Sort per ingoing node
            batch_shifted_index = tf.repeat(batch_splits,batch_length)
            node_order = tf.argsort(indexlist[:,1]+batch_shifted_index,axis=0,direction='ASCENDING',stable=True)
            indexlist = tf.gather(indexlist,node_order,axis=0)
            valuelist = tf.gather(valuelist,node_order,axis=0)
        
        edge_index = tf.RaggedTensor.from_value_rowids(indexlist[:,1:],indexlist[:,0],validate=self.ragged_validate)
        edge_weight = tf.RaggedTensor.from_value_rowids(tf.expand_dims(valuelist,axis=-1),indexlist[:,0],validate=self.ragged_validate)
        
        return edge_index,edge_weight
    
    def get_config(self):
        """Update config."""
        config = super(CastAdjacencyMatrixToRaggedList, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"sort_index": self.sort_index})
        return config 


