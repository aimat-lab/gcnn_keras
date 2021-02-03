import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


    
class PoolingNodes(ks.layers.Layer):
    r"""
    Layer for pooling of nodefeatures over all nodes in graph. Which gives $1/n \sum_i node(i)$.
    
    Args:
        pooling_method : tf.function to pool all nodes compatible with ragged tensors.
        **kwargs
    """
    
    def __init__(self,
                 pooling_method = "reduce_mean" ,
                 **kwargs):
        """Initialize layer."""
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        
        if(self.pooling_method == "reduce_mean"):
            self._pool = tf.math.reduce_mean
        elif(self.pooling_method == "reduce_sum"):
            self._pool = tf.math.reduce_sum
        else:
            raise TypeError("Unknown pooling, choose: reduce_mean, reduce_sum, ...")
        
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodes, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
        
        Args:
            nodes (tf.ragged): Node ragged tensor of shape (batch,None,F)
    
        Returns:
            features (tf.tensor): Pooled node features of shape (batch,F)
        """
        node = inputs
        out = self._pool(node,axis=1)
        #nv = node.values
        #nids = node.value_rowids()
        #out = tf.math.segment_sum(nv,nids)
        return out
    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 
    

class PoolingAllEdges(ks.layers.Layer):
    r"""
    Layer for pooling of edgefeatures over all edges in graph. Which gives $1/n \sum_{ij} edge(i,j)$.
    
    Args:
        pooling_method : tf.function to pool all edges with ragged tensors.
        **kwargs
    """
    
    def __init__(self, 
                 pooling_method = tf.math.reduce_mean,
                 **kwargs):
        """Initialize layer."""
        super(PoolingAllEdges, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.pooling_method  = pooling_method 
        
        if(self.pooling_method == "reduce_mean"):
            self._pool = tf.math.reduce_mean
        elif(self.pooling_method == "reduce_sum"):
            self._pool = tf.math.reduce_sum
        else:
            raise TypeError("Unknown pooling, choose: reduce_mean, reduce_sum, ...")
        
    def build(self, input_shape):
        """Build layer."""
        super(PoolingAllEdges, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
            
        Args:
            edges (tf.ragged): Edge feature ragged tensor of shape (batch,None,F)
    
        Returns:
            features (tf.tensor): Pooled edge features of shape (batch,F)
        """
        edge = inputs        #Apply segmented mean
        out = self._pool(edge,axis=1)
        return out
    def get_config(self):
        """Update layer config."""
        config = super(PoolingAllEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config 


class PoolingEdgesPerNode(ks.layers.Layer):
    r"""
    Layer for pooling of edgefeatures or messages for each ingoing node in graph. Which gives $1/n \sum_{j} edge(i,j)$.
    
    Some layer arguments allow faster performance if set differently.
    
    Args:
        pooling_method (str): tf.function to pool edges compatible with ragged tensors. Default is 'segment_mean'.
        node_indexing (str): If indices refer to sample- or batch-wise indexing. Default is 'sample'.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        ragged_validate (bool): To validate the ragged output tensor. Defualt is False.
        **kwargs
    """
    
    def __init__(self, 
                 pooling_method="segment_mean",
                 node_indexing = "sample",
                 is_sorted = False,
                 has_unconnected = True,
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingEdgesPerNode, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
        self.pooling_method  = pooling_method
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: segment_mean, segment_sum, ...")
        
        self.ragged_validate = ragged_validate
    def build(self, input_shape):
        """Build layer."""
        super(PoolingEdgesPerNode, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [node, edge, edgeindex] 
        
        Args:
            node (tf.ragged): Node ragged tensors of shape (batch,None,F_n)
            edge (tf.ragged): Edge feature ragged tensor of shape (batch,None,F)
            edge_index (tf.ragged): Edge index list as ragged tensor of shape (batch,None,2)
                                    Note that the ragged dimension of edge and edgeindex has to match. 
        
        Returns:
            features (tf.ragged): Pooled node features of shape (batch,None,F) where the ragged dimension matches the nodes.
            The edge feautres are chosen for index tensor according to index(0).
        """
        nod,edge,edgeind = inputs
        if(self.node_indexing == 'batch'):
            shiftind = edgeind.values
        elif(self.node_indexing == 'sample'): 
            shift1 = edgeind.values
            shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        nodind = shiftind[:,0]
        dens = edge.values
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
            outtarget_shape = (tf.shape(nod.values,out_type=nodind.dtype)[0],ks.backend.int_shape(dens)[-1])
            get = tf.scatter_nd(ks.backend.expand_dims(pooled_index,axis=-1), get,outtarget_shape)        
            
        out = tf.RaggedTensor.from_row_splits(get,nod.row_splits,validate=self.ragged_validate)       
        return out     
    def get_config(self):
        """Update layer config."""
        config = super(PoolingEdgesPerNode, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"node_indexing": self.node_indexing})
        config.update({"is_sorted": self.is_sorted})
        config.update({"has_unconnected": self.has_unconnected})
        return config



class PoolingWeightedEdgesPerNode(ks.layers.Layer):
    r"""
    Layer for pooling of edgefeatures for each ingoing node in graph. Which gives $1/n \sum_{j} edge(i,j)$.
    
    Args:
        pooling_method (str): tf.function to pool edges compatible with ragged tensors. Default is "segment_sum".
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        node_indexing (str): If indices refer to sample- or batch-wise indexing. Default is 'sample'.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        ragged_validate (bool): To validate the ragged output tensor. Defualt is False.
        **kwargs
    """
    
    def __init__(self, 
                 pooling_method="segment_sum",
                 normalize_by_weights = False,
                 node_indexing = "sample",
                 is_sorted = False,
                 has_unconnected = True,
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingWeightedEdgesPerNode, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True          
        self.pooling_method  = pooling_method
        
        if(self.pooling_method == "segment_mean"):
            self._pool = tf.math.segment_mean
        elif(self.pooling_method == "segment_sum"):
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: segment_mean, segment_sum, ...")
            
        self.normalize_by_weights = normalize_by_weights
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.ragged_validate = ragged_validate
    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedEdgesPerNode, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List [node, edge, edge_index, weight] 
        
        Args:
            node (tf.ragged): Node ragged tensors of shape (batch,None,F_n)
            edge (tf.ragged): Edge feature ragged tensor of shape (batch,None,F)
            edge_index (tf.ragged): Edge index list as ragged tensor of shape (batch,None,2)
                                    Note that the ragged dimension of edge and edge index has to match. 
            weights (tf.ragged): The weights could be the entry in the ajacency matrix for each edgs in the list 
                                 and must be broadcasted or match in dimension. Shape is e.g. (batch,None,1).
          
        Returns:
            features (tf.ragged): Pooled node features of shape (batch,None,F) where the ragged dimension matches the nodes.
            The edge feautres are chosen for index tensor according to index(0). 
        """
        nod,edge,edgeind,weights = inputs
        if(self.node_indexing == 'batch'):
            shiftind = edgeind.values
        elif(self.node_indexing == 'sample'): 
            shift1 = edgeind.values
            shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        
        #Multiply by weights
        wval = weights.values
        dens = edge.values * wval
        nodind = shiftind[:,0]
        
        if(self.is_sorted==False):        
            #Sort edgeindices
            node_order = tf.argsort(nodind,axis=0,direction='ASCENDING',stable=True)
            nodind = tf.gather(nodind,node_order,axis=0)
            dens = tf.gather(dens,node_order,axis=0)
            wval = tf.gather(wval,node_order,axis=0)
        
        #Do the pooling
        get = self._pool(dens,nodind)
        
        if(self.normalize_by_weights == True):
            get = tf.math.divide_no_nan(get , tf.math.segment_sum(wval,nodind)) # +tf.eps
        
        if(self.has_unconnected == True):
            #Need to fill tensor since not all nodes are also in pooled
            #Does not happen if all nodes are also connected
            pooled_index,_ = tf.unique(nodind)
            outtarget_shape = (tf.shape(nod.values,out_type=nodind.dtype)[0],ks.backend.int_shape(dens)[-1])
            get = tf.scatter_nd(ks.backend.expand_dims(pooled_index,axis=-1), get,outtarget_shape)
        
        out = tf.RaggedTensor.from_row_splits(get,nod.row_splits,validate=self.ragged_validate)       
        return out     
    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedEdgesPerNode, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"node_indexing": self.node_indexing})
        config.update({"is_sorted": self.is_sorted})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"normalize_by_weights": self.weights_normalized})
        return config  