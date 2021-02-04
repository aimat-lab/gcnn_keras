import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



class AdjacencyPower(ks.layers.Layer):
    """
    Computes powers of the adjacency matrix. 
    
    Note: Layer casts to dense until sparse matmul is supported. This is very inefficient.
        
    Args:
        n (int): Power of the adjacency matrix. Default is 2.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self,partition_type = "row_length" ,n=2, **kwargs):
        """Initialize layer."""
        super(AdjacencyPower, self).__init__(**kwargs)          
        self.n = n
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(AdjacencyPower, self).build(input_shape)          
    def call(self, inputs):
        """Forward path.
        
        Inputs List [edge_indices, edges, edge_length, node_length] 
        
        Args: 
            edge_indices (tf.tensor): Flatten index list of shape (batch*None,2)
            edges (tf.tensor): Flatten adjacency entries of shape (batch*None,1)
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            
        Returns:
            List: [edge_indices, edges, edge_len]
            
            - edge_indices (tf.tensor): Flatten index list of shape (batch*None,2)
            - edges (tf.tensor): Flatten adjacency entries of shape (batch*None,1)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
              Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
        """
        edge_index,edge,edge_part,node_part = inputs
        
        # Cast to length tensor
        if(self.partition_type == "row_length"):
            edge_len = edge_part
            node_len = node_part
        elif(self.partition_type == "row_splits"):
            edge_len = edge_part[1:] - edge_part[:-1]
            node_len = node_part[1:] - node_part[:-1]
        elif(self.partition_type == "value_rowids"):            
            edge_len = tf.math.segment_sum(tf.ones_like(edge_part),edge_part)
            node_len = tf.math.segment_sum(tf.ones_like(node_part),node_part)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
        
        #batchwise indexing
        shift_index = tf.expand_dims(tf.repeat(tf.cumsum(node_len,exclusive=True),edge_len),axis=1)
        edge_index = edge_index - tf.cast(shift_index,dtype=edge_index.dtype)
        ind_batch = tf.cast(tf.expand_dims(tf.repeat(tf.range(tf.shape(edge_len)[0]),edge_len),axis=-1),dtype=edge_index.dtype)
        ind_all = tf.concat([ind_batch,edge_index],axis=-1)
        ind_all = tf.cast(ind_all,dtype = tf.int64)
        
        max_index = tf.reduce_max(edge_len)
        dense_shape = tf.stack([tf.cast(tf.shape(edge_len)[0],dtype=max_index.dtype),max_index,max_index])
        dense_shape = tf.cast(dense_shape ,dtype = tf.int64)
        
        edge_val = edge[:,0] # Must be 1D tensor
        adj = tf.sparse.SparseTensor(ind_all, edge_val, dense_shape)
        
        out0 = tf.sparse.to_dense(adj,validate_indices=False)
        out = out0
        
        for i in range(self.n-1):
            out = tf.matmul(out,out0)
        
        ind1 = tf.repeat(tf.expand_dims(tf.range(max_index),axis=-1),max_index,axis=-1)
        ind2 = tf.repeat(tf.expand_dims(tf.range(max_index),axis=0),max_index,axis=0)
        ind12 = tf.concat([tf.expand_dims(ind1,axis=-1),tf.expand_dims(ind2,axis=-1)],axis=-1)
        ind = tf.repeat(tf.expand_dims(ind12,axis=0),tf.shape(edge_len)[0],axis=0)
        new_shift = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.cumsum(node_len,exclusive=True),axis=-1),axis=-1),axis=-1)
        ind = ind + new_shift
        
        mask = out>0
        imask = tf.cast(mask,dtype = max_index.dtype)
        new_edge_len = tf.reduce_sum(tf.reduce_sum(imask,axis=-1),axis=-1)

        new_edge_index = ind[mask]
        new_edge = tf.expand_dims(out[mask],axis=-1 )
        
        if(self.partition_type == "row_length"):
            new_edge_part = new_edge_len
        elif(self.partition_type == "row_splits"):
            new_edge_part =  tf.pad(tf.cumsum(new_edge_len),[[1,0]]) 
        elif(self.partition_type == "value_rowids"):            
            new_edge_part = tf.repeat(tf.range(tf.shape(new_edge_len)[0]),new_edge_len)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...") 
        
        return [new_edge_index,new_edge,new_edge_part]
    
    def get_config(self):
        """Update layer config."""
        config = super(AdjacencyPower, self).get_config()
        config.update({"n": self.n})
        config.update({"partition_type": self.partition_type})
        return config  
    
