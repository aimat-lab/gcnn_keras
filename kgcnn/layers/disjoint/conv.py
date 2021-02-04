import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges,PoolingWeightedEdgesPerNode
from kgcnn.utils.activ import kgcnn_custom_act 


class GCN(ks.layers.Layer):
    """
    Graph convolution according to Kipf et al.
    
    Computes graph conv as $sigma(A*(WX+b))$ where A is the precomputed adjacency matrix.
    In place of A, edges and edge indices are used. A is considered pre-sacled. Otherwise use e.g. segment-mean, scale by weights etc.
    Edges must be broadcasted to node feautres X.
    
    Args:
        units (int): Output dimension/ units of dense layer.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        activation (str): Activation function 'relu'.
        pooling_method (str): Pooling method for summing edges 'segment_sum'.
        use_bias (bool): Whether to use bias. Default is False,
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """
    
    def __init__(self, 
                 units,
                 node_indexing = 'batch',
                 activation='relu',
                 pooling_method= 'segment_sum',
                 use_bias = False,
                 is_sorted=False,
                 has_unconnected=True,
                 normalize_by_weights = False,
                 partition_type = "row_length" ,
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.units = units
        self.node_indexing = node_indexing 
        self.normalize_by_weights = normalize_by_weights
        self.use_bias = use_bias
        self.partition_type = partition_type
        self.pooling_method = pooling_method
        self.has_unconnected = has_unconnected
        self.is_sorted = is_sorted
        self.activation = activation
        
        self.deserial_activation = ks.activations.deserialize(activation,custom_objects=kgcnn_custom_act) if isinstance(activation,str) or isinstance(activation,dict) else activation
        #Layers
        self.lay_gather = GatherNodesOutgoing(node_indexing = self.node_indexing,partition_type=self.partition_type)
        self.lay_dense = ks.layers.Dense(self.units,use_bias=self.use_bias,activation='linear')
        self.lay_pool = PoolingWeightedEdgesPerNode(pooling_method= self.pooling_method,is_sorted=self.is_sorted,
                                                    has_unconnected=self.has_unconnected,node_indexing = self.node_indexing,
                                                    normalize_by_weights = self.normalize_by_weights,partition_type=self.partition_type)
        self.lay_act = ks.layers.Activation(self.deserial_activation)
    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs list [node, node_partition, edge, edge_partition, edge_index]
        
        Args: 
            nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
            edge_index (tf.tensor): Edge indices for disjoint representation of shape
                                    (batch*None,2) that corresponds to indexing 'batch'.
        
        Returns:
            features (tf.tensor): A list of updated node features.        
            Output shape is (batch*None,F).
        """
        node,node_len,edges,edge_len,edge_index = inputs
        no = self.lay_gather([node,node_len,edge_index,edge_len])
        no = self.lay_dense(no)
        nu = self.lay_pool([node,node_len,no,edge_len,edges]) # Summing for each node connection
        out = self.lay_act(nu)
        return out     
    def get_config(self):
        """Update config."""
        config = super(GatherNodesOutgoing, self).get_config()
        config.update({"units": self.units})
        config.update({"node_indexing": self.node_indexing})
        config.update({"normalize_by_weights": self.normalize_by_weights})
        config.update({"use_bias": self.use_bias})
        config.update({"pooling_method": self.pooling_method})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"is_sorted": self.is_sorted})
        config.update({"activation": self.activation})
        config.update({"partition_type": self.partition_type})
        return config 
    


class cfconv(ks.layers.Layer):
    """
    Convolution layer of Schnet. Disjoint representation.
    
    Edges are proccessed by 2 Dense layers, multiplied on outgoing nodefeatures and pooled for ingoing node. 
    
    Args:
        units (int): Units for Dense layer.
        activation (str): Activation function. Default is 'selu'.
        use_bias (bool): Use bias. Default is True.
        cfconv_pool (str): Pooling method. Default is 'segment_sum'.
        is_sorted (bool): If edge indices are sorted. Default is True.
        has_unconnected (bool): If graph has unconnected nodes. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
    """
    
    def __init__(self, units, 
                 activation='selu',
                 use_bias = True,
                 cfconv_pool = 'segment_sum',
                 is_sorted = False,
                 has_unconnected= True,
                 partition_type = "row_length" ,
                 node_indexing = 'batch',
                 **kwargs):
        """Initialize Layer."""
        super(cfconv, self).__init__(**kwargs)
        self.activation = activation
        self.use_bias = use_bias
        self.cfconv_pool= cfconv_pool
        self.units = units
        self.is_sorted = is_sorted
        self.partition_type = partition_type
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        
        self.deserial_activation = ks.activations.deserialize(activation,custom_objects=kgcnn_custom_act) if isinstance(activation,str) or isinstance(activation,dict) else activation
        #Layer
        self.lay_dense1 = ks.layers.Dense(units=self.units,activation=self.deserial_activation,use_bias=self.use_bias)
        self.lay_dense2 = ks.layers.Dense(units=self.units,activation='linear',use_bias=self.use_bias)
        self.lay_sum = PoolingEdgesPerNode(pooling_method=self.cfconv_pool,
                                           is_sorted = self.is_sorted , 
                                           has_unconnected=self.has_unconnected,
                                           partition_type=self.partition_type,
                                           node_indexing = self.node_indexing)
        self.gather_n = GatherNodesOutgoing(node_indexing = self.node_indexing,partition_type=self.partition_type)
    def build(self, input_shape):
        """Build layer."""
        super(cfconv, self).build(input_shape)
    def call(self, inputs):
        """Forward pass: Calculate edge update.
        
        Inputs [node, node_partition, edge, edge_partition, edge_index]
        
        Args:
            nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of nodes to each graph in batch. Default is row_length of shape (batch,)
            edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids, row_splits etc.
                                        Yields the assignment of edges to each graph in batch. Default is row_length of shape (batch,)
            edge_index (tf.tensor): Edge indices for disjoint representation of shape
                                    (batch*None,2) that corresponds to indexing 'batch'.
        
        Returns:
            node_update (tf.tensor): Updated node features of shape (batch*None,F)
        """
        node, bn, edge, edge_len, indexlis = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2Exp = self.gather_n([node,bn,indexlis,edge_len])
        x = node2Exp*x
        x= self.lay_sum([node, bn,x,edge_len,indexlis])
        return x
    def get_config(self):
        """Update layer config."""
        config = super(cfconv, self).get_config()
        config.update({"activation": self.activation})
        config.update({"use_bias": self.use_bias})
        config.update({"cfconv_pool": self.cfconv_pool})
        config.update({"units": self.units})
        config.update({"is_sorted}": self.is_sorted})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"partition_type": self.partition_type})
        return config 