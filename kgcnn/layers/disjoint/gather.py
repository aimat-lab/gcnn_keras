import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class GatherNodes(ks.layers.Layer):
    """
    Gather nodes by edge indexlist. Indexlist must match flatten nodes.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    
    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, node_indexing = 'batch',partition_type = "row_length" , **kwargs):
        """Initialize layer."""
        super(GatherNodes, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodes, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [node, node_length, edge_index]
        
        Args:
            node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
            
        Returns:
            features (tf.tensor): Gathered node features of (ingoing,outgoing) nodes.        
            Output shape is (batch*None,F+F).  
        """
        node,node_part,edge_index,edge_part = inputs
        
        if(self.node_indexing == 'batch'):
            indexlist = edge_index 
        elif(self.node_indexing == 'sample'):
            shift1 = edge_index
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
            indexlist = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        
        node1Exp = tf.gather(node,indexlist[:,0],axis=0)
        node2Exp = tf.gather(node,indexlist[:,1],axis=0)
        out = K.concatenate([node1Exp,node2Exp],axis=1)
        return out     
    def get_config(self):
        """Update config."""
        config = super(GatherNodes, self).get_config()
        config.update({"node_indexing": self.node_indexing})
        config.update({"partition_type": self.partition_type})
        return config 



class GatherNodesOutgoing(ks.layers.Layer):
    """
    Gather nodes by edge indexlist. Indexlist must match flatten nodes.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    For outgoing nodes, layer uses only index[1].
    
    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, node_indexing = 'batch',partition_type = "row_length",**kwargs):
        """Initialize layer."""
        super(GatherNodesOutgoing, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type          
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesOutgoing, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [node, node_length, edge_index]
        
        Args: 
            node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
                                    For ingoing gather nodes according to index[1]
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
        
        Returns:
            features (tf.tensor): A list of gathered outgoing node features from indexlist.        
            Output shape is (batch*None,F).
        
        """
        node,node_part,edge_index,edge_part = inputs
        # node,edge_index= inputs
        
        if(self.node_indexing == 'batch'):
            indexlist = edge_index 
        elif(self.node_indexing == 'sample'):
            shift1 = edge_index
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
            indexlist = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        
        out = tf.gather(node,indexlist[:,1],axis=0)
        return out     
    def get_config(self):
        """Update config."""
        config = super(GatherNodesOutgoing, self).get_config()
        config.update({"node_indexing": self.node_indexing})
        config.update({"partition_type": self.partition_type})
        return config 
    

class GatherNodesIngoing(ks.layers.Layer):
    """
    Gather nodes by edge indexlist. Indexlist must match flatten nodes.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    For ingoing nodes, layer uses only index[0].
    
    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, node_indexing = 'batch',partition_type = "row_length",**kwargs):
        """Initialize layer."""
        super(GatherNodesIngoing, self).__init__(**kwargs)
        self.node_indexing = node_indexing     
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesIngoing, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.

        Inputs List of [node, node_len, edge_index]
        
        Args:
            node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
                                    For ingoing gather nodes according to index[0]
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
    
        Returns:
            features (tf.tensor): A list of gathered ingoing node features from indexlist.        
            Output shape is (batch*None,F).
        """
        node,node_part,edge_index,edge_part = inputs
        # node,edge_index= inputs
        
        if(self.node_indexing == 'batch'):
            indexlist = edge_index 
        elif(self.node_indexing == 'sample'):
            shift1 = edge_index
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
            indexlist = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        
        out = tf.gather(node,indexlist[:,0],axis=0)
        return out     
    def get_config(self):
        """Update config."""
        config = super(GatherNodesIngoing, self).get_config()
        config.update({"node_indexing": self.node_indexing})
        config.update({"partition_type": self.partition_type})
        return config 
    
    
class GatherState(ks.layers.Layer):
    """
    Layer to repeat environment or global state for node or edge lists. The node or edge lists are flattened.
    
    To repeat the correct environment for each sample, a tensor with the target length/partition is required.

    Args:
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self,partition_type = "row_length", **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)          
        self.partition_type = partition_type
    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [environment, target_length]
        
        Args:
            environment (tf.tensor): List of graph specific feature tensor of shape (batch*None,F)
            target_partition (tf.tensor): Assignment of nodes or edges to each graph in batch. 
                                          Default is row_length of shape (batch,).

        Returns:
            features (tf.tensor): A tensor with repeated single state for each graph.
            Output shape is (batch*N,F).
        """
        env,target_part = inputs
        
        if(self.partition_type == "row_length"):
            target_len = target_part
        elif(self.partition_type == "row_splits"):
            target_len = target_part[1:] - target_part[:-1]
        elif(self.partition_type == "value_rowids"):
            target_len = tf.math.segment_sum(tf.ones_like(target_part),target_part)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")
    
        out = tf.repeat(env,target_len,axis=0)
        return out     
    def get_config(self):
        """Update config."""
        config = super(GatherState, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config 