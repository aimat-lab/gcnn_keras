import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class PoolingEdgesPerNode(ks.layers.Layer):
    """
    Pooling all edges or edgelike features per node, corresponding to node assigned by edge indexlist.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] are sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function. Default is 'segment_mean'.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        **kwargs
        
    Input: 
        List of tensors [node,edges,edge_indices]
        node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
        edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
        edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
                                  The index for segment reduction is taken from edge_indices[:,0].
    
    Output:
        features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
                              The size will match the flatten node tensor.
                              Output shape is (batch*None, F).
    """
    
    def __init__(self, 
                 pooling_method = "segment_mean",
                 is_sorted = True,
                 has_unconnected = False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingEdgesPerNode, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")
        
    def build(self, input_shape):
        """Build layer."""
        super(PoolingEdgesPerNode, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        nod,edge,edgeind = inputs
        
        nodind = edgeind[:,0]
        dens = edge
        if(self.is_sorted==False):        
            #Sort edgeindices
            node_order = tf.argsort(nodind,axis=0,direction='ASCENDING',stable=True)
            nodind = tf.gather(nodind,node_order,axis=0)
            dens = tf.gather(dens,node_order,axis=0)
        
        #Pooling via e.g. segment_sum
        get = self._pool(dens,nodind)
        
        if(self.has_unconnected == True):
            #Need to fill tensor since the maximum node may not be also in pooled
            #Does not happen if all nodes are also connected
            pooled_index = tf.range(tf.shape(get)[0])# tf.unique(nodind)
            outtarget_shape = (tf.shape(nod,out_type=nodind.dtype)[0],ks.backend.int_shape(dens)[-1])
            get = tf.scatter_nd(ks.backend.expand_dims(pooled_index,axis=-1), get, outtarget_shape)
            
        out = get
        return out
    def get_config(self):
        """Update layer config."""
        config = super(PoolingEdgesPerNode, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config  
        


class PoolingNodes(ks.layers.Layer):
    """
    Polling all nodes per batch. The batch assignment is given by a length-tensor.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        **kwargs
    
    Input: 
        List of tensors [nodes, node_length] 
        nodes (tf.tensor): Flatten node features of shape (batch*None,F)
        node_length (tf.tensor): Number of nodes in each graph of shape (batch,)
    
    Output:
        features (tf.tensor): Pooled node feature list of shape (batch,F)
                              where F is the feature dimension and holds a pooled 
                              node feature for each graph.
    """

    def __init__(self,  
                 pooling_method = "segment_mean",
                 **kwargs):
        """Initialize layer."""
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")
            
    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodes, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        node,len_node = inputs        
        len_shape_int = K.shape(len_node)
        batchi = tf.repeat(K.arange(0,len_shape_int[0],1),len_node)
        out = self._pool(node,batchi)
        #Output should have correct shape
        return out
    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 



class PoolingAllEdges(ks.layers.Layer):
    """
    Pooling all edges per graph. The batch assignment is given by a length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        **kwargs

    Input: 
        List of tensors [egdes,edge_length] 
        edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
        edge_length (tf.tensor): Number of edges in each graph of shape (batch*None,F)
                                 Keeps the batch assignment.

    Output:
        features (tf.tensor): A pooled edges feature list of shape (batch,F).
                              where F is the feature dimension and holds a pooled 
                              edge feature for each graph.
    """
    
    def __init__(self,
                 pooling_method = "segment_mean",
                 **kwargs):
        """Initialize layer."""
        super(PoolingAllEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")
            
    def build(self, input_shape):
        """Build layer."""
        super(PoolingAllEdges, self).build(input_shape)
    def call(self, inputs):
        """Forward pass."""
        edge,len_edge = inputs
        len_shape = K.shape(len_edge)
        batchi = tf.repeat(K.arange(0,len_shape[0],1),len_edge)
        out = self._pool(edge,batchi)
        #Output already has correct shape
        return out
    def get_config(self):
        """Update layer config."""
        config = super(PoolingAllEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 