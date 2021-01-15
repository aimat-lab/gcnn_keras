import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class CastRaggedToValues(ks.layers.Layer):
    """
    Cast a ragged tensor with one ragged dimension, like node or edge lists to a single value tensor plus row length.
    
    The shape follows (batch,None,F) -> ("batch*None",F).
    
    Args:
        **kwargs
    Input:    
        Ragged tensor of shape (batch,None,F) where None is the number of nodes or edges in each graph.
    Output:
        A tuple of [values,row_length] extracted from ragged tensor.
        The output shape is ("batch*None",F) for values.
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastRaggedToValues, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToValues, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        tens = inputs
        flat_tens = tens.values
        #node_id or edge_id is (btach,)
        row_lengths = tens.row_lengths()
        return (flat_tens,row_lengths)


class CastMaskedToValues(ks.layers.Layer):
    """
    Cast a zero-padded tensor plus mask input to a single list plus row_length tensor.
    
    Args:
        **kwargs
    Input: 
        Padded tensor plus mask [padded,mask] of shape [(batch,N,F),(batch,N)],
        where N is the maximum number of nodes or edges and mask encodes active entries.
    Output:
        A tuple of [values,row_length] extracted from padded tensor.
        The output shape is something like ("batch[Mask]",F).
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastMaskedToValues, self).__init__(**kwargs)
    def build(self, input_shape):
        """Build layer."""
        super(CastMaskedToValues, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        tens,mask = inputs
        #Ensure mask is of type bool
        mask = K.cast(mask,dtype="bool")
        fmask = K.cast(mask,dtype="int64")
        row_lengths = K.sum(fmask,axis=1)
        #shape of nodematrix
        shape_tens = K.shape(tens)
        shape_tens_int = K.int_shape(tens)
        # Flatten batch dimension
        batchred_tens = K.reshape(tens,(shape_tens[0]*shape_tens[1],shape_tens_int[2]))
        batchred_mask = K.reshape(mask,(shape_tens[0]*shape_tens[1],))
        #Apply boolean mask
        flat_tens = tf.boolean_mask(batchred_tens,batchred_mask)
        #Output 
        return (flat_tens,row_lengths)
    

class CastBatchToValues(ks.layers.Layer):
    """
    Layer to squeeze the batch dimension according to (batch,N,F) -> (bath*N,F).
    
    Also a fake batch of size 1 with multiple graphs (None,) could be possible with data-loader.
    
    Args:
        **kwargs
    Input: 
        Batched tensor plus size info [features,batch_length] of shape [(batch,N,F),(batch,N)],
        The length of additional sub-graphs can be provided by the Batch-length tensor.
    Output:
        A tuple of [values,row_length] obtained by flattened Features.
        The row-length information is given by the flattened batch-length tensor.
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastBatchToValues, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(CastBatchToValues, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        feat,feat_len= inputs
        sh_feat = K.shape(feat)
        sh_feat_int = K.int_shape(feat)
        sh_feat_len = K.shape(feat_len)
        out = K.reshape(feat,(sh_feat[0]*sh_feat[1],sh_feat_int[-1]))
        out_len = K.reshape(feat_len,(sh_feat_len[0]*sh_feat_len[1],))    
        return out,out_len


class CastValuesToBatch(ks.layers.Layer):
    """
    Add batchdim according to a reference of correct batch length. 
    
    The change of tensorshape follows (batch*N,F) + (batch,N,...) -> (batch,N,F).

    Args:
        **kwargs
    Input:
        A list of [tensor,reference] of shape [(batch*N,F),(batch,N,...)]
    Output:
        The same tensor of reshaped first and second dimensions according to a reference tensor.
        Output has shape (batch,N,F)
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastValuesToBatch, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(CastValuesToBatch, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        infeat,ref = inputs
        outsh = K.int_shape(infeat)
        insh = K.shape(ref)
        out = K.reshape(infeat,(insh[0],insh[1],outsh[-1]))
        return out     



class CastValuesToPadded(ks.layers.Layer):
    """
    Layer to add zero padding to enable a batch-dimension for a dense tensor from a flattened list.
    
    The layer changes the shape of input: (batch*None,feat) + Mask: (batch,N) -> (batch,N,Feat).
    
    Args:
        **kwargs
    Input:
        A tuple of [list,reference] of shape [(batch*None,F),(batch,N)]
    Output:
        The same tensor with padded second dimension and first batch-dimension.
        Output has shape (batch,N,F).
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastValuesToPadded, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(CastValuesToPadded, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        infeat,mask = inputs
        feat_shape_int = K.int_shape(infeat)
        feat_shape = K.shape(infeat)
        ref_shape = K.shape(mask)
        batchred_mask = K.reshape(mask,(ref_shape[0]*ref_shape[1],))
        bi = tf.range(0,ref_shape[0]*ref_shape[1])
        bi = tf.boolean_mask(bi,batchred_mask)
        bi = tf.expand_dims(bi,axis=1)
        refill = tf.scatter_nd(bi,infeat,shape=(ref_shape[0]*ref_shape[1],feat_shape_int[-1]))
        out = K.reshape(refill,(ref_shape[0],ref_shape[1],feat_shape_int[-1]))
        return out     


class CastValuesToRagged(ks.layers.Layer):
    """
    Layer to make ragged tensor from a flatten value tensor plus row length.
    
    Args:
        **kwargs
    Input: 
        A list of  [value_tensor,row_length] of shape (batch*None,Feat),(batch*None)
    Output:
        A ragged tensor of shape (batch,None,Feat)
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastValuesToRagged, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(CastValuesToRagged, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        nod,n_len = inputs
        out = tf.RaggedTensor.from_row_lenghts(nod,n_len)   
        return out     
    

class CorrectIndexListForSubGraph(ks.layers.Layer):
    """
    Shifts the index for flatten index-tensors to match over the batch dimension.
    
    For example: flatten: [[0,1,2],[0,1],[0,1]] -> [0,1,2,0,1,0,1]
    shift: [0,1,2,1,1,0,1] -> [0,1,2,3+0,3+1,5+0,5+1]
    This is equivalent to one big graph with disconnected subgraphs.
    Therfore tf.gather would gather the correct nodes for a 1D tensor.
    
    Args:
        **kwargs
    Input: 
        A list of flatten [row_length_node,edge_index,row_length_edge]
    Output:
        A shifted edge_index_list to match the correct nodes in the flatten nodelist.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CorrectIndexListForSubGraph, self).__init__(**kwargs)
    def build(self, input_shape):
        """Build layer."""
        super(CorrectIndexListForSubGraph, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        len_node,edge_index,len_edge = inputs
        shift_index = tf.expand_dims(tf.repeat(tf.cumsum(len_node,exclusive=True),len_edge),axis=1)
        indexlist = edge_index + tf.cast(shift_index,dtype=edge_index.dtype)  
        #indexlist = edge_index + shift_index 
        return indexlist


class CastRaggedToDisjoint(ks.layers.Layer):
    """ 
    Transform ragged tensor input to singe disjoint graph lists. 
    
    Args:
        **kwargs
    Input: 
        All lists representing a graph [node,edge,edgeindex] with all being ragged tensor input.
        The input shape is expected to be [(batch,None,F),(batch,None,F),(batch,None,2)]
    Output:
        A single graph plus node_id and edge_id that keeps the number of nodes in each subgraph.
        Output is reduced list (nodes,node-id,edges,edge-id,edgeindices) for single graph. 
        The output shape is [(batch*N,F),(batch,),(batch*M,F),(batch),(batch*M,2)]
    """
    def __init__(self,**kwargs):
        """Initialize layer."""
        super(CastRaggedToDisjoint, self).__init__(**kwargs)
        self.cast_list = CastRaggedToValues()
        self.correct_index = CorrectIndexListForSubGraph()
    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToDisjoint, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        node_input,edge_input,edge_index_input = inputs
        n,node_len = self.cast_list(node_input)
        ed,edge_len = self.cast_list(edge_input)
        edi,_ = self.cast_list(edge_index_input)
        edi = self.correct_index([node_len,edi,edge_len])
        return (n,node_len,ed,edge_len,edi)
