import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



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
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids, row_splits etc.
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


class CastMaskedToValues(ks.layers.Layer):
    """
    Cast a zero-padded tensor plus mask input to a single list plus row_partition tensor.
    
    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, partition_type = "row_length", **kwargs):
        """Initialize layer."""
        super(CastMaskedToValues, self).__init__(**kwargs)
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(CastMaskedToValues, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Inputs list: [padded_values,mask]
        
        Args: 
            padded_values (tf.tensor): Zero padded feature tensor of shape (batch,N,F).
                                        where F denotes the feature dimension and N the maximum
                                        number of edges/nodes in graph.
            mask (tf.tensor): Boolean mask of shape (batch,N),
                              where N is the maximum number of nodes or edges.
        
        Returns:
            list: [values, value_partition] 
            
            - values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
              The output shape is given (batch[Mask],F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids, row_splits etc.
              Yields the assignment of nodes/edges per graph in batch. Default is row_length.
        """
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
        if(self.partition_type == "row_length"):
            outpart = row_lengths
        elif(self.partition_type == "row_splits"):
            outpart = tf.pad(tf.cumsum(row_lengths),[[1,0]])
        elif(self.partition_type == "value_rowids"):
            outpart = tf.repeat(tf.range(shape_tens[0]),row_lengths)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
        
        return [flat_tens,outpart]
    def get_config(self):
        """Update layer config."""
        config = super(CastMaskedToValues, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config   


class CastBatchToValues(ks.layers.Layer):
    """
    Layer to squeeze the batch dimension. For graphs of the same size in batch.
    
    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        **kwargs    
    """
    
    def __init__(self,partition_type = "row_length", **kwargs):
        """Make layer."""
        super(CastBatchToValues, self).__init__(**kwargs)          
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(CastBatchToValues, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs tf.tensor values.
            
        Args: 
            values (tf.tensor): Feature tensor with explicit batch dimension of shape (batch,N,F)
        
        Returns:
            list: [values, value_partition] 
            
            - values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids, row_splits etc.
              Yields the assignment of nodes/edges per graph in batch. Default is row_length.
        """
        feat = inputs
        sh_feat = K.shape(feat)
        sh_feat_int = K.int_shape(feat)
        out = K.reshape(feat,(sh_feat[0]*sh_feat[1],sh_feat_int[-1]))  
        out_len = tf.repeat(sh_feat[1],sh_feat[0])
        #Output 
        if(self.partition_type == "row_length"):
            outpart = out_len
        elif(self.partition_type == "row_splits"):
            outpart = tf.pad(tf.cumsum(out_len),[[1,0]]) 
        elif(self.partition_type == "value_rowids"):
            outpart = tf.repeat(tf.range(tf.shape(out_len)[0]),out_len)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
        return [out,outpart]
    def get_config(self):
        """Update layer config."""
        config = super(CastBatchToValues, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config   



class CastValuesToBatch(ks.layers.Layer):
    """
    Add batchdim according to a reference. For graphs of the same size in batch.

    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        **kwargs
    """

    def __init__(self,partition_type = "row_length" ,**kwargs):
        """Initialize layer."""
        super(CastValuesToBatch, self).__init__(**kwargs)     
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(CastValuesToBatch, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs list [values, value_partition]
                
        Args: 
            values (tf.tensor): Flatten feature tensor of shape (batch*N,F).
            value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids, row_splits etc.
                                         Yields the assignment of nodes/edges per graph in batch. Default is row_length.
                                      
        Returns:
            features (tf.tensor): Feature tensor of shape (batch,N,F).
            The first and second dimensions is reshaped according to a reference tensor.
            F denotes the feature dimension. Requires graphs of identical size in batch.
        """
        infeat,inpartition = inputs
        outsh = K.int_shape(infeat)
        
        if(self.partition_type == "row_length"):
            ref = inpartition
            insh = K.shape(ref)
            out = K.reshape(infeat,(insh[0],-1,outsh[-1]))
        elif(self.partition_type == "row_splits"):
            ref = inpartition[:-1]
            insh = K.shape(ref)
            out = K.reshape(infeat,(insh[0],-1,outsh[-1]))
        elif(self.partition_type == "value_rowids"):
            ref = tf.math.segment_sum(tf.ones_like(inpartition),inpartition)
            insh = K.shape(ref)
            out = K.reshape(infeat,(insh[0],-1,outsh[-1]))
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 

        return out     
    def get_config(self):
        """Update layer config."""
        config = super(CastBatchToValues, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config  


class CastValuesToPadded(ks.layers.Layer):
    """
    Layer to add zero padding for a fixed size tensor having an explicit batch-dimension.
    
    The layer maps disjoint representation to padded tensor plus mask.
    
    Args:
        partition_type (str): Partition tensor type. Default is "row_length".
        **kwargs
    """
    
    def __init__(self,partition_type = "row_length", **kwargs):
        """Initialize layer."""
        super(CastValuesToPadded, self).__init__(**kwargs)          
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(CastValuesToPadded, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [values, value_partition]
        
        Args:
            values (tf.tensor): Feature tensor with flatten batch dimension of shape (batch*None,F).
            value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids, row_splits etc.
                                         Yields the assignment of nodes/edges per graph in batch. Default is row_length.
            
        Returns:
            list: [values,mask]
            
            - values (tf.tensor): Padded feature tensor with shape (batch,N,F)
            - mask (tf.tensor): Boolean mask of shape (batch,N)
        """
        nod,npartin = inputs
        
        #Just make ragged tensor.
        if(self.partition_type == "row_length"):
            n_len = npartin
            out = tf.RaggedTensor.from_row_lengths(nod,n_len)
        elif(self.partition_type == "row_splits"):
            out = tf.RaggedTensor.from_row_splits(nod,npartin)
            n_len = out.row_lengths()
        elif(self.partition_type == "value_rowids"):
            out = tf.RaggedTensor.from_value_rowids(nod,npartin)
            n_len = out.row_lengths()
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
        
        #Make padded
        out = out.to_tensor()
        #Make mask
        max_len = tf.shape(out)[1]
        n_padd = max_len - n_len
        mask = ks.backend.flatten(tf.concat([tf.expand_dims(tf.ones_like(n_len,dtype=tf.bool),axis=-1),tf.expand_dims(tf.zeros_like(n_len,dtype=tf.bool),axis=-1)],axis=-1))
        reps = ks.backend.flatten(tf.concat([tf.expand_dims(n_len,axis=-1),tf.expand_dims(n_padd,axis=-1)],axis=-1))
        mask = tf.repeat(mask,reps)
        mask = tf.reshape(mask,tf.shape(out)[:2])
        return [out,mask]
    def get_config(self):
        """Update layer config."""
        config = super(CastValuesToPadded, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config  


class CastValuesToRagged(ks.layers.Layer):
    """
    Layer to make ragged tensor from a flatten value tensor plus row partition tensor.
    
    Args:
        partition_type (str): Partition tensor type. Default is "row_length".
        **kwargs
    """
    
    def __init__(self,partition_type = "row_length", **kwargs):
        """Initialize layer."""
        super(CastValuesToRagged, self).__init__(**kwargs) 
        self.partition_type = partition_type         
    def build(self, input_shape):
        """Build layer."""
        super(CastValuesToRagged, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs list of [values, value_partition] 
           
        Args: 
            values (tf.tensor): Feature tensor of nodes/edges of shape (batch*None,F)
                                where F stands for the feature dimension and None represents
                                the flexible size of the graphs.
            value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids, row_splits etc.
                                         Yields the assignment of nodes/edges per graph in batch. Default is row_length.
            
        Returns:
            features (tf.ragged): A ragged feature tensor of shape (batch,None,F).
        """
        nod,n_part = inputs
        
        if(self.partition_type == "row_length"):
            out = tf.RaggedTensor.from_row_lengths(nod,n_part)
        elif(self.partition_type == "row_splits"):
            out = tf.RaggedTensor.from_row_splits(nod,n_part)
        elif(self.partition_type == "value_rowids"):
            out = tf.RaggedTensor.from_value_rowids(nod,n_part)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
            
        return out     
    def get_config(self):
        """Update layer config."""
        config = super(CastValuesToRagged, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config  
    

class ChangeIndexing(ks.layers.Layer):
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
        partition_type (str): Partition tensor type. Default is "row_length".
        **kwargs
    """

    def __init__(self,to_indexing = 'batch',from_indexing = 'sample',partition_type = "row_length" ,**kwargs):
        """Initialize layer."""
        super(ChangeIndexing, self).__init__(**kwargs)
        self.to_indexing = to_indexing 
        self.from_indexing = from_indexing
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(ChangeIndexing, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Inputs list of [node_partition, edge_index, edge_partition]
                    
        Args: 
            node_partition (tf.tensor): Node assignment to each graph, for example number of nodes in each graph of shape (batch,).
            edge_index (tf.tensor): Flatten edge-index list of shape (batch*None,2)
            edge_partition (tf.tensor): Edge assignment to each graph, for example number of edges in each graph of shape (batch,).
            
        Returns:
            edge_index (tf.tensor): Corrected edge-index list to match the nodes 
            in the flatten nodelist. Shape is (batch*None,2).     
        """
        part_node,edge_index,part_edge = inputs
        
        #splits[1:] - splits[:-1]
        if(self.partition_type == "row_length"):
            shift_index = tf.expand_dims(tf.repeat(tf.cumsum(part_node,exclusive=True),part_edge),axis=1)
        elif(self.partition_type == "row_splits"):
            edge_len = part_edge[1:] - part_edge[:-1]
            shift_index = tf.expand_dims(tf.repeat(part_node[:-1], edge_len ),axis=1)
        elif(self.partition_type == "value_rowids"):
            node_len = tf.math.segment_sum(tf.ones_like(part_node),part_node)
            edge_len = tf.math.segment_sum(tf.ones_like(part_edge),part_edge)
            shift_index = tf.expand_dims(tf.repeat(tf.cumsum(node_len,exclusive=True),edge_len),axis=1)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
            
        # Add or substract batch offset from index tensor
        if(self.to_indexing == 'batch' and self.from_indexing == 'sample'):        
            indexlist = edge_index + tf.cast(shift_index,dtype=edge_index.dtype)  
        elif(self.to_indexing == 'sample'and self.from_indexing == 'batch'):
            indexlist = edge_index - tf.cast(shift_index,dtype=edge_index.dtype)  
        else:
            raise TypeError("Unknown index change, use: 'sample', 'batch', ...")
        
        return indexlist
    def get_config(self):
        """Update layer config."""
        config = super(ChangeIndexing, self).get_config()
        config.update({"to_indexing": self.to_indexing})
        config.update({"from_indexing": self.from_indexing})
        config.update({"partition_type": self.partition_type})
        return config 
    
    

class CastRaggedToDisjoint(ks.layers.Layer):
    """ 
    Transform ragged tensor input to disjoint graph representation.
    
    Disjoint graph representation has disjoint subgraphs within a single graph.
    Batch dimension is flatten for this representation.
    
    Args:
        to_indexing (str): The index refer to the overall 'batch' or to single 'sample'.
                           The disjoint representation assigns nodes within the 'batch'.
                           It changes "sample" to "batch" or "batch" to "sample."
                           Default is 'batch'.
        from_indexing (str): Index convention that has been set for the input.
                             Default is 'sample'.
        partition_type (str): Partition tensor type. Default is "row_length".
        **kwargs
    """

    def __init__(self,partition_type = "row_length",to_indexing = 'batch',from_indexing = 'sample',**kwargs):
        """Initialize layer."""
        super(CastRaggedToDisjoint, self).__init__(**kwargs)
        self.to_indexing = to_indexing 
        self.from_indexing = from_indexing
        self.partition_type = partition_type
        self.cast_list = CastRaggedToValues(partition_type=self.partition_type)
        self.correct_index = ChangeIndexing(to_indexing = self.to_indexing,from_indexing = self.from_indexing,partition_type=self.partition_type)
    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToDisjoint, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Inputs list of [node, edge, edgeindex]        
                
        Args: 
            node (tf.ragged): Node feature ragged tensor of shape (batch,None,F)
                              where None stands for a flexible graph size and
                              F the node feature dimension.
            edge (tf.ragged): Edge feature ragged tensor of shape (batch,None,F)
                              where None stands for a flexible graph size and
                              F the edge feature dimension.
            edge_index (tf.ragged): Edge indices as a list of shape (batch,None,2)
                                    which has index pairs [i,j] matching nodes
                                    within each sample. Assumes 'sample' indexing.
        
        Returns:
            list: [nodes,node_partition,edges,edge_partition,edge_index]
            
            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - node_partition (tf.tensor): Node assignment to each graph, for example number of nodes in each graph of shape (batch,).
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Edge assignment to each graph, for example number of edges in each graph of shape (batch,).
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.
        """
        node_input,edge_input,edge_index_input = inputs
        n,node_len = self.cast_list(node_input)
        ed,edge_len = self.cast_list(edge_input)
        edi,_ = self.cast_list(edge_index_input)
        edi = self.correct_index([node_len,edi,edge_len])
        return [n,node_len,ed,edge_len,edi]
    def get_config(self):
        """Update layer config."""
        config = super(ChangeIndexing, self).get_config()
        config.update({"to_indexing": self.to_indexing})
        config.update({"from_indexing": self.from_indexing})
        config.update({"partition_type": self.partition_type})
        return config 