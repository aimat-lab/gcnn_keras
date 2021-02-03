import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class CastRaggedToDense(tf.keras.layers.Layer):
    """
    Layer to cast a ragged tensor to a dense tensor.
    
    Args:
        **kwargs
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastRaggedToDense, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToDense, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Args:
            features (tf.ragged): Feature ragged tensor of shape e.g. (batch,None,F)
        
        Returns:
            tf.tensor: Input.to_tensor() with zero padding.        
        """
        return inputs.to_tensor()


class CastRaggedToValues(ks.layers.Layer):
    """
    Cast a ragged tensor with one ragged dimension, like node feature list to a single value plus partition tensor.
    
    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, partition_type = "row_length", **kwargs):
        """Initialize layer."""
        super(CastRaggedToValues, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToValues, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Inputs tf.ragged feature tensor.
        
        Args:
            features (tf.ragged): Ragged tensor of shape (batch,None,F) ,
                                  where None is the number of nodes or edges in each graph and
                                  F denotes the feature dimension.
    
        Returns:
            list: [values, value_partition]
            
            - values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, row_id, row_splits etc.
              Yields the assignment of nodes/edges per graph. Default is row_length.
        """
        tens = inputs
        flat_tens = tens.values
        
        if(self.partition_type == "row_length"):
            outpart = tens.row_lengths()
        elif(self.partition_type == "row_splits"):
            outpart = tens.row_splits
        elif(self.partition_type == "value_rowids"):
            outpart = tens.value_rowids()
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
            
        return [flat_tens,outpart]
    def get_config(self):
        """Update layer config."""
        config = super(CastRaggedToValues, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config  
    


class CastAdjacencyMatrixToRaggedList(ks.layers.Layer):
    """
    Cast a sparse batched adjacency matrices to a ragged index list plus connection weights.
    
    Args:
        sort_index (bool): If indices are sorted in sparse matrix.
        ragged_validate (bool): Validate ragged tensor.
        **kwargs
    """
    
    def __init__(self,sort_index = True,ragged_validate=False ,**kwargs):
        """Initialize layer."""
        super(CastAdjacencyMatrixToRaggedList, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.sort_index = sort_index
        self.ragged_validate = ragged_validate
    def build(self, input_shape):
        """Build layer."""
        super(CastAdjacencyMatrixToRaggedList, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Args:
            adjacency (tf.sparse): A sparse Tensor (tf.sparse) of shape (batch,N_max,N_max).
                                   The sparse tensor that has the shape of maximum number of nodes in the batch.
        
        Returns:
            list: [edge_index,edges]
            
            - edge_index (tf.ragged): Edge indices list of shape (batch,None,2)
            - edges (tf.ragged): Edge feature list of shape (batch,None,1)
        """
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
        
        return [edge_index,edge_weight]
    def get_config(self):
        """Update config."""
        config = super(CastAdjacencyMatrixToRaggedList, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"sort_index": self.sort_index})
        return config 



class ChangeIndexing(ks.layers.Layer):
    """
    Change indexing between sample-wise and in-batch labeling. 'Bath' is equivalent to disjoint indexing.
    
    Note that ragged Gather- and Pooling-layers require node_indexing = "batch" as argument if index is shifted by the number of nodes in batch.
    This can enable faster gathering and pooling for some layers.
    
    Example:
        edge_index = ChangeIndexingRagged()([input_node,input_edge_index]) 
        [[0,1],[1,0],...],[[0,2],[1,2],...],...] to [[0,1],[1,0],...],[[5,7],[6,7],...],...] 
    
    Args:
        to_indexing (str): The index refer to the overall 'batch' or to single 'sample'.
                           The disjoint representation assigns nodes within the 'batch'.
                           It changes "sample" to "batch" or "batch" to "sample."
                           Default is 'batch'.
        from_indexing (str): Index convention that has been set for the input.
                             Default is 'sample'.  
        ragged_validate (bool): Validate ragged tensor. Default is False.
        **kwargs
    """
    
    def __init__(self, to_indexing = 'batch',from_indexing = 'sample' ,
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(ChangeIndexing, self).__init__(**kwargs) 
        self.ragged_validate = ragged_validate
        self.to_indexing = to_indexing 
        self.from_indexing = from_indexing
        self._supports_ragged_inputs = True   
    def build(self, input_shape):
        """Build layer."""
        super(ChangeIndexing, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Inputs list [nodes,edge_indices]    
        
        Args:
            nodes (tf.ragged): Ragged node feature list of shape (batch,None,F).
            edge_indices (tf.ragged): Ragged edge indices of shape (batch,None,2).
            
        Returns:
            edge_indices (tf.ragged): Ragged tensor of edge indices with modified index reference.  
        """
        nod,edgeind = inputs
        shift1 = edgeind.values
        shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
        
        if(self.to_indexing == 'batch' and self.from_indexing == 'sample'):        
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)   
        elif(self.to_indexing == 'sample'and self.from_indexing == 'batch'):
            shiftind = shift1 - tf.cast(shift2,dtype=shift1.dtype)    
        else:
            raise TypeError("Unknown index change, use: 'sample', 'batch', ...")
        
        out = tf.RaggedTensor.from_row_splits(shiftind,edgeind.row_splits,validate=self.ragged_validate)
        return out
    def get_config(self):
        """Update config."""
        config = super(ChangeIndexing, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"from_indexing": self.from_indexing})
        config.update({"to_indexing": self.to_indexing})
        return config    