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
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, 
                 pooling_method = "segment_mean",
                 node_indexing = "batch",
                 is_sorted = True,
                 has_unconnected = False,
                 partition_type = "row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingEdgesPerNode, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        
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
        """Forward pass.
        
        Inputs List of [node, node_partition, edges, edge_partition, edge_indices]
        
        Args: 
            node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
            edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
                                      The index for segment reduction is taken from edge_indices[:,0].
    
        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod,node_part,edge,edge_part,edgeind = inputs
        
        if(self.node_indexing == 'batch'):
            shiftind = edgeind 
        elif(self.node_indexing == 'sample'):
            shift1 = edgeind
            if(self.partition_type == "row_length"):
                edge_len = edge_part
                node_len = node_part
                shift2 = tf.expand_dims(tf.repeat(tf.cumsum(node_len,exclusive=True),edge_len),axis=1)
            elif(self.partition_type == "row_splits"):
                edge_len = edge_part[1:] - edge_part[:-1]
                shift2 = tf.expand_dims(tf.repeat(node_part[:-1],edge_len),axis=1)
            elif(self.partition_type == "value_rowids"):
                edge_len = tf.math.segment_sum(tf.ones_like(edge_part),edge_part)
                node_len = tf.math.segment_sum(tf.ones_like(node_part),node_part)
                shift2 = tf.expand_dims(tf.repeat(tf.cumsum(node_len,exclusive=True),edge_len),axis=1)
            else:
                raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
            
        nodind = shiftind[:,0]
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
        config.update({"is_sorted": self.is_sorted})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"node_indexing": self.node_indexing})
        config.update({"partition_type": self.partition_type})
        return config  
        
    
    

class PoolingWeightedEdgesPerNode(ks.layers.Layer):
    """
    Pooling all edges or message/edgelike features per node, corresponding to node assigned by edge indexlist.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] could be sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function. Default is 'segment_mean'.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, 
                 pooling_method = "segment_mean",
                 is_sorted = True,
                 node_indexing = "batch",
                 has_unconnected = False,
                 normalize_by_weights = False,
                 partition_type = "row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingWeightedEdgesPerNode, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.normalize_by_weights = normalize_by_weights
        self.partition_type = partition_type
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")
        
    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedEdgesPerNode, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [node, node_partition, edges, edge_partition, edge_indices]
        
        Args: 
            node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
            edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
                                      The index for segment reduction is taken from edge_indices[:,0] (ingoing node).
            weights (tf.tensor): The weights could be the entry in the ajacency matrix for each edge in the list 
                                 and must be broadcasted or match in dimension. Shape is e.g. (batch*None,1).
    
        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod,node_part,edge,edge_part,edgeind,weights = inputs
        
        if(self.node_indexing == 'batch'):
            shiftind = edgeind 
        elif(self.node_indexing == 'sample'):
            shift1 = edgeind
            if(self.partition_type == "row_length"):
                edge_len = edge_part
                node_len = node_part
                shift2 = tf.expand_dims(tf.repeat(tf.cumsum(node_len,exclusive=True),edge_len),axis=1)
            elif(self.partition_type == "row_splits"):
                edge_len = edge_part[1:] - edge_part[:-1]
                shift2 = tf.expand_dims(tf.repeat(node_part[:-1],edge_len),axis=1)
            elif(self.partition_type == "value_rowids"):
                edge_len = tf.math.segment_sum(tf.ones_like(edge_part),edge_part)
                node_len = tf.math.segment_sum(tf.ones_like(node_part),node_part)
                shift2 = tf.expand_dims(tf.repeat(tf.cumsum(node_len,exclusive=True),edge_len),axis=1)
            else:
                raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        
        wval = weights
        dens = edge* wval
        nodind = shiftind[:,0]
        
        if(self.is_sorted==False):        
            #Sort edgeindices
            node_order = tf.argsort(nodind,axis=0,direction='ASCENDING',stable=True)
            nodind = tf.gather(nodind,node_order,axis=0)
            dens = tf.gather(dens,node_order,axis=0)
            wval = tf.gather(wval,node_order,axis=0)
        
        #Pooling via e.g. segment_sum
        get = self._pool(dens,nodind)
        
        if(self.normalize_by_weights == True):
            get = tf.math.divide_no_nan(get , tf.math.segment_sum(wval,nodind)) # +tf.eps
        
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
        config = super(PoolingWeightedEdgesPerNode, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        config.update({"is_sorted": self.is_sorted})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"node_indexing": self.node_indexing})
        config.update({"normalize_by_weights": self.normalize_by_weights})
        config.update({"partition_type": self.partition_type})
        return config  



class PoolingNodes(ks.layers.Layer):
    """
    Polling all nodes per batch. The batch assignment is given by a length-tensor.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,  
                 pooling_method = "segment_mean",
                 partition_type = "row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.partition_type = partition_type
        
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
        """Forward pass.
        
        Inputs List of [nodes, node_partition] 
        
        Args: 
            nodes (tf.tensor): Flatten node features of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
    
        Returns:
            features (tf.tensor): Pooled node feature list of shape (batch,F)
            where F is the feature dimension and holds a pooled 
            node feature for each graph.
        """
        node,node_part = inputs 
        
        if(self.partition_type == "row_length"):
            node_len = node_part
            batchi = tf.repeat(tf.range(tf.shape(node_len)[0]),node_len)
        elif(self.partition_type == "row_splits"):
            node_len = node_part[1:] - node_part[:-1]
            batchi = tf.repeat(tf.range(tf.shape(node_len)[0]),node_len)
        elif(self.partition_type == "value_rowids"):
            batchi = node_part
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")
        
        out = self._pool(node,batchi)
        #Output should have correct shape
        return out
    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        config.update({"partition_type": self.partition_type})
        return config 



class PoolingAllEdges(ks.layers.Layer):
    """
    Pooling all edges per graph. The batch assignment is given by a length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self,
                 pooling_method = "segment_mean",
                 partition_type = "row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingAllEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.partition_type = partition_type
        
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
        """Forward pass.
        
        Inputs List of [egdes, edge_partition] 
        
        Args: 
            edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            edge_partition (tf.tensor): Row partition for edges. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
    
        Returns:
            features (tf.tensor): A pooled edges feature list of shape (batch,F).
            where F is the feature dimension and holds a pooled 
            edge feature for each graph.
        """
        edge,edge_part = inputs
        
        if(self.partition_type == "row_length"):
            edge_len = edge_part
            batchi = tf.repeat(tf.range(tf.shape(edge_len)[0]),edge_len)
        elif(self.partition_type == "row_splits"):
            edge_len = edge_part[1:] - edge_part[:-1]
            batchi = tf.repeat(tf.range(tf.shape(edge_len)[0]),edge_len)
        elif(self.partition_type == "value_rowids"):
            batchi = edge_part
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")
        
        out = self._pool(edge,batchi)
        #Output already has correct shape
        return out
    def get_config(self):
        """Update layer config."""
        config = super(PoolingAllEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        config.update({"partition_type": self.partition_type})
        return config 