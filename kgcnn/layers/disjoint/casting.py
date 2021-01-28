import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



class CastRaggedToValues(ks.layers.Layer):
    """
    Cast a ragged tensor with one ragged dimension, like node feature list to a single value plus row length tensor.
    
    Args:
        **kwargs
        
    Input:
        features (tf.ragged): Ragged tensor of shape (batch,None,F) ,
                              where None is the number of nodes or edges in each graph and
                              F denotes the feature dimension.
    
    Output:
        List of tensors [values,row_length]
        values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
        row_length (tf.tensor): Row length tensor of the number of nodes/edges per graph with shape (batch,).
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
        return [flat_tens,row_lengths]



class CastMaskedToValues(ks.layers.Layer):
    """
    Cast a zero-padded tensor plus mask input to a single list plus row_length tensor.
    
    Args:
        **kwargs
        
    Input: 
        List of tensors [padded_values,mask] 
        padded_values (tf.tensor): Zero padded feature tensor of shape (batch,N,F).
                                   where F denotes the feature dimension and N the maximum
                                   number of edges/nodes in graph.
        mask (tf.tensor): Boolean mask of shape (batch,N),
                          where N is the maximum number of nodes or edges.
        
    Output:
        List of tensors [values,row_length] 
        values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
                            The output shape is given (batch[Mask],F).
        row_length (tf.tensor): Row length tensor of the number of node/edges per graph with shape (batch,)
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
        return [flat_tens,row_lengths]
    


class CastBatchToValues(ks.layers.Layer):
    """
    Layer to squeeze the batch dimension. For graphs of the same size in batch.
    
    Args:
        **kwargs
        
    Input: 
        values (tf.tensor): Feature tensor with explicit batch dimension of shape (batch,N,F)
    
    Output:
        List of tensors [values,row_length] 
        values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
        row_length (tf.tensor): Row length tensor of the number of nodes/edges per batch with shape (batch,)
                                For a graphs of the same size this is tf.tensor([N,N,N,N,...])
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastBatchToValues, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(CastBatchToValues, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        feat = inputs
        sh_feat = K.shape(feat)
        sh_feat_int = K.int_shape(feat)
        out = K.reshape(feat,(sh_feat[0]*sh_feat[1],sh_feat_int[-1]))  
        out_len = tf.repeat(sh_feat[1],sh_feat[0])
        return [out,out_len]



class CastValuesToBatch(ks.layers.Layer):
    """
    Add batchdim according to a reference. For graphs of the same size in batch.

    Args:
        **kwargs
        
    Input:
        List of tensors [values,reference] 
        values (tf.tensor): Flatten feature tensor of shape (batch*N,F).
        reference (tf.tensor): Reference tensor of explicit batch dimension of shape (batch,N,...).
                                  
    Output:
        features (tf.tensor): Feature tensor of shape (batch,N,F).
                              The first and second dimensions is reshaped according to a reference tensor.
                              F denotes the feature dimension. Requires graph of identical size in batch.
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
    Layer to add zero padding for a fixed size tensor having an explicit batch-dimension.
    
    The layer maps disjoint representation to padded tensor plus mask.
    
    Args:
        **kwargs
        
    Input:
        List of tensors [values,mask]
        values (tf.tensor): Feature tensor with flatten batch dimension of shape (batch*None,F).
        row_length (tf.tensor): Number of nodes/edges in each graph of shape (batch,).
        
    Output:
        List of tensors [values,mask]
        values (tf.tensor): Padded feature tensor with shape (batch,N,F)
        mask (tf.tensor): Boolean mask of shape (batch,N)
    """
    
    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastValuesToPadded, self).__init__(**kwargs)          
    def build(self, input_shape):
        """Build layer."""
        super(CastValuesToPadded, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        nod,n_len = inputs
        #Make padded
        out = tf.RaggedTensor.from_row_lengths(nod,n_len)
        out = out.to_tensor()
        #Make mask
        max_len = tf.shape(out)[1]
        n_padd = max_len - n_len
        mask = ks.backend.flatten(tf.concat([tf.expand_dims(tf.ones_like(n_len,dtype=tf.bool),axis=-1),tf.expand_dims(tf.zeros_like(n_len,dtype=tf.bool),axis=-1)],axis=-1))
        reps = ks.backend.flatten(tf.concat([tf.expand_dims(n_len,axis=-1),tf.expand_dims(n_padd,axis=-1)],axis=-1))
        mask = tf.repeat(mask,reps)
        mask = tf.reshape(mask,tf.shape(out)[:2])
        return [out,mask]



class CastValuesToRagged(ks.layers.Layer):
    """
    Layer to make ragged tensor from a flatten value tensor plus row_length tensor.
    
    Args:
        **kwargs
        
    Input: 
        List of tensors [values,row_length] 
        values (tf.tensor): Feature tensor of nodes/edges of shape (batch*None,F)
                            where F stands for the feature dimension and None represents
                            the flexible size of the graphs.
        row_length (tf.tensor): Number of nodes/edges in each graph of shape (batch,).
        
    Output:
        features (tf.ragged): A ragged feature tensor of shape (batch,None,Feat).
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
        out = tf.RaggedTensor.from_row_lengths(nod,n_len)   
        return out     
    
    

class ChangeIndexingDisjoint(ks.layers.Layer):
    """
    Shift the index for flatten index-tensors to assign nodes in a disjoint graph representation or vice-versa.
    
    Example: 
        Flatten operation changes index tensor as [[0,1,2],[0,1],[0,1]] -> [0,1,2,0,1,0,1] with
        requires a subsequent index-shift of [0,1,2,1,1,0,1] -> [0,1,2,3+0,3+1,5+0,5+1].
        This is equivalent to a single graph with disconnected subgraphs.
        Therfore tf.gather will find the correct nodes for a 1D tensor.
    
    Args:
        to_indexing (str): The index refer to the overall 'batch' or to single 'sample'.
                           The disjoint representation assigns nodes within the 'batch'.
                           It changes "sample" to "batch" or "batch" to "sample."
                           Default is 'batch'.
        from_indexing (str): Index convention that has been set for the input.
                             Default is 'sample'.               
        **kwargs
        
    Input: 
        List of tensors [node_length,edge_index,edge_length]
        node_length (tf.tensor): Number of nodes in each graph of shape (batch,)
        edge_index (tf.tensor): Flatten edge-index list of shape (batch*None,2)
        edge_length (tf.tensor): Number of edges in each graph of shape (batch,)
        
    Output:
        edge_index (tf.tensor): Corrected edge-index list to match the nodes 
                                in the flatten nodelist. Shape is (batch*None,2).
    """

    def __init__(self,to_indexing = 'batch',from_indexing = 'sample' ,**kwargs):
        """Initialize layer."""
        super(ChangeIndexingDisjoint, self).__init__(**kwargs)
        self.to_indexing = to_indexing 
        self.from_indexing = from_indexing
    def build(self, input_shape):
        """Build layer."""
        super(ChangeIndexingDisjoint, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        len_node,edge_index,len_edge = inputs
        shift_index = tf.expand_dims(tf.repeat(tf.cumsum(len_node,exclusive=True),len_edge),axis=1)
        
        if(self.to_indexing == 'batch' and self.from_indexing == 'sample'):        
            indexlist = edge_index + tf.cast(shift_index,dtype=edge_index.dtype)  
        elif(self.to_indexing == 'sample'and self.from_indexing == 'batch'):
            indexlist = edge_index - tf.cast(shift_index,dtype=edge_index.dtype)  
        else:
            raise TypeError("Unknown index change, use: 'sample', 'batch', ...")
        
        return indexlist
    def get_config(self):
        """Update layer config."""
        config = super(ChangeIndexingDisjoint, self).get_config()
        config.update({"to_indexing": self.to_indexing})
        config.update({"from_indexing": self.from_indexing})
        return config 
    
    

class CastRaggedToDisjoint(ks.layers.Layer):
    """ 
    Transform ragged tensor input to disjoint graph representation. 
    
    Disjoint graph representation has disjoint subgraphs within a single graph.
    Batch dimension is flatten for this representation.
    
    Args:
        **kwargs
        
    Input: 
        List of tensors [node,edge,edgeindex]
        node (tf.ragged): Node feature ragged tensor of shape (batch,None,F)
                          where None stands for a flexible graph size and
                          F the node feature dimension.
        edge (tf.ragged): Edge feature ragged tensor of shape (batch,None,F)
                          where None stands for a flexible graph size and
                          F the edge feature dimension.
        edge_index (tf.ragged): Edge indices as a list of shape (batch,None,2)
                                which has index pairs [i,j] matching nodes
                                within each sample. Assumes 'sample' indexing.
        
    Output:
        List of tensors [nodes,node_length,edges,edge_length,edge_index]
        nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
        node_length (tf.tensor): Number of nodes in each graph (batch,)
        edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
        edge_length (tf.tensor): Number of edges in each graph (batch,)
        edge_index (tf.tensor): Edge indices for disjoint representation of shape
                                (batch*None,2) that corresponds to indexing 'batch'.
    """
    
    def __init__(self,**kwargs):
        """Initialize layer."""
        super(CastRaggedToDisjoint, self).__init__(**kwargs)
        self.cast_list = CastRaggedToValues()
        self.correct_index = ChangeIndexingDisjoint()
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
        return [n,node_len,ed,edge_len,edi]
