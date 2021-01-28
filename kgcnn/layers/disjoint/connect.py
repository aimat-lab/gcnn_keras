import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



class AdjacencyPower(ks.layers.Layer):
    """
    Computes powers of the adjacency matrix. 
    
    Note: Layer casts to dense until sparse matmul is supported. This is very inefficient.
        
    Args:
        n (int): Power of the adjacency matrix. Default is 2.
        **kwargs
        
    Input: 
        List of tensors [edge_indices, edges, edge_length,node_length] 
        edge_indices (tf.tensor): Flatten index list of shape (batch*None,2)
        edges (tf.tensor): Flatten adjacency entries of shape (batch*None,1)
        edge_length (tf.tensor): Number of edges in each graph (batch,)
        node_length (tf.tensor): Number of nodes in each graph of shape (batch,)
        
    Output:
        List of tensors [edge_indices, edges, edge_len]
        edge_indices (tf.tensor): Flatten index list of shape (batch*None,2)
        edges (tf.tensor): Flatten adjacency entries of shape (batch*None,1)
        edge_length (tf.tensor): Number of edges in each graph (batch,)
    """
    
    def __init__(self, n=2, **kwargs):
        """Initialize layer."""
        super(AdjacencyPower, self).__init__(**kwargs)          
        self.n = n
    def build(self, input_shape):
        """Build layer."""
        super(AdjacencyPower, self).build(input_shape)          
    def call(self, inputs):
        """Forward path."""
        edge_index,edge,edge_len,node_len = inputs
        
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
        
        return [new_edge_index,new_edge,new_edge_len]
    
    def get_config(self):
        """Update layer config."""
        config = super(AdjacencyPower, self).get_config()
        config.update({"n": self.n})
        return config  
    
