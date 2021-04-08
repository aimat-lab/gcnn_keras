import tensorflow.keras as ks

from kgcnn.layers.disjoint.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.disjoint.pooling import PoolingLocalEdges, PoolingWeightedLocalEdges, PoolingGlobalEdges, \
    PoolingNodes
from kgcnn.utils.activ import kgcnn_custom_act


class GCN(ks.layers.Layer):
    """
    Graph convolution according to Kipf et al.
    
    Computes graph conv as $sigma(adj_matrix*(WX+b))$ where adj_matrix is the precomputed adjacency matrix.
    In place of adj_matrix, edges and edge edge_indices are used. adj_matrix is considered pre-sacled.
    Otherwise use e.g. segment-mean, scale by weights etc.
    Edges must be broadcasted to node feautres X.
    
    Args:
        units (int): Output dimension/ units of dense layer.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        activation (str): Activation function 'relu'.
        pooling_method (str): Pooling method for summing edges 'segment_sum'.
        use_bias (bool): Whether to use bias. Default is False,
        is_sorted (bool): If the edge edge_indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 units,
                 node_indexing='batch',
                 activation='relu',
                 pooling_method='segment_sum',
                 use_bias=False,
                 is_sorted=False,
                 has_unconnected=True,
                 normalize_by_weights=False,
                 partition_type="row_length",
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

        self.deserial_activation = ks.activations.deserialize(activation, custom_objects=kgcnn_custom_act) \
            if isinstance(activation, str) or isinstance(activation, dict) else activation
        # Layers
        self.lay_gather = GatherNodesOutgoing(node_indexing=self.node_indexing, partition_type=self.partition_type)
        self.lay_dense = ks.layers.Dense(self.units, use_bias=self.use_bias, activation='linear')
        self.lay_pool = PoolingWeightedLocalEdges(pooling_method=self.pooling_method, is_sorted=self.is_sorted,
                                                  has_unconnected=self.has_unconnected,
                                                  node_indexing=self.node_indexing,
                                                  normalize_by_weights=self.normalize_by_weights,
                                                  partition_type=self.partition_type)
        self.lay_act = ks.layers.Activation(self.deserial_activation)

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Edge edge_indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.
        
        Returns:
            features (tf.tensor): A list of updated node features.
            Output shape is (batch*None,F).
        """
        node, node_len, edges, edge_len, edge_index = inputs
        no = self.lay_gather([node, node_len, edge_index, edge_len])
        no = self.lay_dense(no)
        nu = self.lay_pool([node, node_len, no, edge_len, edges])  # Summing for each node connection
        out = self.lay_act(nu)
        return out

    def get_config(self):
        """Update config."""
        config = super(GCN, self).get_config()
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


class SchNetCFconv(ks.layers.Layer):
    """
    Continous Filter convolution of SchNet. Disjoint representation.
    
    Edges are proccessed by 2 Dense layers, multiplied on outgoing nodefeatures and pooled for ingoing node. 
    
    Args:
        units (int): Units for Dense layer.
        activation (str): Activation function. Default is 'selu'.
        use_bias (bool): Use bias. Default is True.
        cfconv_pool (str): Pooling method. Default is 'segment_sum'.
        is_sorted (bool): If edge edge_indices are sorted. Default is True.
        has_unconnected (bool): If graph has unconnected nodes. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
    """

    def __init__(self, units,
                 activation='selu',
                 use_bias=True,
                 cfconv_pool='segment_sum',
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='batch',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.activation = activation
        self.use_bias = use_bias
        self.cfconv_pool = cfconv_pool
        self.units = units
        self.is_sorted = is_sorted
        self.partition_type = partition_type
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing

        # Layer
        self.lay_dense1 = ks.layers.Dense(units=self.units, activation=self.activation, use_bias=self.use_bias)
        self.lay_dense2 = ks.layers.Dense(units=self.units, activation='linear', use_bias=self.use_bias)
        self.lay_sum = PoolingLocalEdges(pooling_method=self.cfconv_pool,
                                         is_sorted=self.is_sorted,
                                         has_unconnected=self.has_unconnected,
                                         partition_type=self.partition_type,
                                         node_indexing=self.node_indexing)
        self.gather_n = GatherNodesOutgoing(node_indexing=self.node_indexing, partition_type=self.partition_type)

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.
        
        Returns:
            node_update (tf.tensor): Updated node features of shape (batch*None,F)
        """
        node, bn, edge, edge_len, indexlist = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2exp = self.gather_n([node, bn, indexlist, edge_len])
        x = node2exp * x
        x = self.lay_sum([node, bn, x, edge_len, indexlist])
        return x

    def get_config(self):
        """Update layer config."""
        config = super(SchNetCFconv, self).get_config()
        config.update({"activation": self.activation})
        config.update({"use_bias": self.use_bias})
        config.update({"cfconv_pool": self.cfconv_pool})
        config.update({"units": self.units})
        config.update({"is_sorted}": self.is_sorted})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"partition_type": self.partition_type})
        return config


class SchNetInteraction(ks.layers.Layer):
    """
    Interaction block.

    Args:
        node_dim (int): Dimension of node embedding. Default is 64.
        activation (str): Activation function. Default is 'selu'.
        use_bias (bool): Use bias in last layers. Default is True.
        use_bias_cfconv (bool): Use bias in CFconv layer. Default is True.
        cfconv_pool (str): Pooling method information for CFconv layer. Default is'segment_sum'.
        is_sorted (bool): Whether node indices are sorted. Default is False.
        has_unconnected (bool): Whether graph has unconnected nodes. Default is False.
        partition_type (str): Partition type of the partition information. Default is row_length".
        node_indexing (str): Indexing information. Whether indices refer to per sample or per batch. Default is "batch".
    """

    def __init__(self, node_dim=128,
                 activation='shifted_softplus',
                 use_bias_cfconv=True,
                 use_bias=True,
                 cfconv_pool='segment_sum',
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='batch',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)
        self.activation = activation
        self.use_bias = use_bias
        self.use_bias_cfconv = use_bias_cfconv
        self.cfconv_pool = cfconv_pool
        self.node_dim = node_dim
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.partition_type = partition_type
        self.node_indexing = node_indexing
        # Layers
        self.lay_cfconv = SchNetCFconv(self.node_dim, activation=self.activation, use_bias=self.use_bias_cfconv,
                                       cfconv_pool=self.cfconv_pool, has_unconnected=self.has_unconnected,
                                       is_sorted=self.is_sorted, partition_type=self.partition_type,
                                       node_indexing=self.node_indexing)
        self.lay_dense1 = ks.layers.Dense(units=self.node_dim, activation='linear', use_bias=False)
        self.lay_dense2 = ks.layers.Dense(units=self.node_dim, activation=self.activation, use_bias=self.use_bias)
        self.lay_dense3 = ks.layers.Dense(units=self.node_dim, activation='linear', use_bias=self.use_bias)
        self.lay_add = ks.layers.Add()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate node update.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.

        Returns:
            node_update (tf.tensor): Updated node features.
        """
        node, bn, edge, edge_len, indexlist = inputs
        x = self.lay_dense1(node)
        x = self.lay_cfconv([x, bn, edge, edge_len, indexlist])
        x = self.lay_dense2(x)
        x = self.lay_dense3(x)
        out = self.lay_add([node, x])
        return out

    def get_config(self):
        config = super(SchNetInteraction, self).get_config()
        config.update({"node_dim": self.node_dim})
        config.update({"activation": self.activation})
        config.update({"use_bias": self.use_bias})
        config.update({"cfconv_pool": self.cfconv_pool})
        config.update({"is_sorted}": self.is_sorted})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"partition_type": self.partition_type})
        config.update({"node_indexing": self.node_indexing})
        config.update({"use_bias_cfconv": self.use_bias_cfconv})
        return config


class MEGnetBlock(ks.layers.Layer):
    """
    Megnet Block.

    Args:
        node_embed (list, optional): List of node embedding dimension. Defaults to [16,16,16].
        edge_embed (list, optional): List of edge embedding dimension. Defaults to [16,16,16].
        env_embed (list, optional): List of environment embedding dimension. Defaults to [16,16,16].
        activation (func, optional): Activation function. Defaults to 'selu'.
        use_bias (bool, optional): Use bias. Defaults to True.
        is_sorted (bool, optional): Edge index list is sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        partition_type (str): Partition type of the partition information. Default is row_length".
        node_indexing (str): Indexing information. Whether indices refer to per sample or per batch. Default is "batch".
        **kwargs
    """

    def __init__(self, node_embed=None,
                 edge_embed=None,
                 env_embed=None,
                 activation='selu',
                 use_bias=True,
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='batch',
                 **kwargs):
        """Initialize layer."""
        super(MEGnetBlock, self).__init__(**kwargs)
        if node_embed is None:
            node_embed = [16, 16, 16]
        if env_embed is None:
            env_embed = [16, 16, 16]
        if edge_embed is None:
            edge_embed = [16, 16, 16]
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.env_embed = env_embed
        self.activation = activation
        self.use_bias = use_bias
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.partition_type = partition_type
        self.node_indexing = node_indexing
        # Node
        self.lay_phi_n = ks.layers.Dense(self.node_embed[0], activation=self.activation, use_bias=self.use_bias)
        self.lay_phi_n_1 = ks.layers.Dense(self.node_embed[1], activation=self.activation, use_bias=self.use_bias)
        self.lay_phi_n_2 = ks.layers.Dense(self.node_embed[2], activation='linear', use_bias=self.use_bias)
        self.lay_esum = PoolingLocalEdges(is_sorted=self.is_sorted, has_unconnected=self.has_unconnected,
                                          partition_type=self.partition_type, node_indexing=self.node_indexing)
        self.lay_gather_un = GatherState(partition_type=self.partition_type)
        self.lay_conc_nu = ks.layers.Concatenate(axis=-1)
        # Edge
        self.lay_phi_e = ks.layers.Dense(self.edge_embed[0], activation=self.activation, use_bias=self.use_bias)
        self.lay_phi_e_1 = ks.layers.Dense(self.edge_embed[1], activation=self.activation, use_bias=self.use_bias)
        self.lay_phi_e_2 = ks.layers.Dense(self.edge_embed[2], activation='linear', use_bias=self.use_bias)
        self.lay_gather_n = GatherNodes(node_indexing=self.node_indexing, partition_type=self.partition_type)
        self.lay_gather_ue = GatherState(partition_type=self.partition_type)
        self.lay_conc_enu = ks.layers.Concatenate(axis=-1)
        # Environment
        self.lay_usum_e = PoolingGlobalEdges(partition_type=self.partition_type)
        self.lay_usum_n = PoolingNodes(partition_type=self.partition_type)
        self.lay_conc_u = ks.layers.Concatenate(axis=-1)
        self.lay_phi_u = ks.layers.Dense(self.env_embed[0], activation=self.activation, use_bias=self.use_bias)
        self.lay_phi_u_1 = ks.layers.Dense(self.env_embed[1], activation=self.activation, use_bias=self.use_bias)
        self.lay_phi_u_2 = ks.layers.Dense(self.env_embed[2], activation='linear', use_bias=self.use_bias)

    def build(self, input_shape):
        """Build layer."""
        super(MEGnetBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, edges, edge_index, env_input, node_partition, edge_partition]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.
            - graph_state (tf.tensor): Graph state input of shape (batch, F).
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)

        Returns:
            list: vp,ep,up
        """
        # Calculate edge Update
        node_input, edge_input, edge_index_input, env_input, len_node, len_edge = inputs
        e_n = self.lay_gather_n([node_input, len_node, edge_index_input, len_edge])
        e_u = self.lay_gather_ue([env_input, len_edge])
        ec = self.lay_conc_enu([e_n, edge_input, e_u])
        ep = self.lay_phi_e(ec)  # Learning of Update Functions
        ep = self.lay_phi_e_1(ep)  # Learning of Update Functions
        ep = self.lay_phi_e_2(ep)  # Learning of Update Functions
        # Calculate Node update
        vb = self.lay_esum([node_input, len_node, ep, len_edge, edge_index_input])  # Summing for each node connections
        v_u = self.lay_gather_un([env_input, len_node])
        vc = self.lay_conc_nu([vb, node_input, v_u])  # Concatenate node features with new edge updates
        vp = self.lay_phi_n(vc)  # Learning of Update Functions
        vp = self.lay_phi_n_1(vp)  # Learning of Update Functions
        vp = self.lay_phi_n_2(vp)  # Learning of Update Functions
        # Calculate environment update
        es = self.lay_usum_e([ep, len_edge])
        vs = self.lay_usum_n([vp, len_node])
        ub = self.lay_conc_u([es, vs, env_input])
        up = self.lay_phi_u(ub)
        up = self.lay_phi_u_1(up)
        up = self.lay_phi_u_2(up)  # Learning of Update Functions
        return vp, ep, up

    def get_config(self):
        config = super(MEGnetBlock, self).get_config()
        config.update({"node_embed": self.node_embed,
                       "edge_embed": self.edge_embed,
                       "env_embed": self.env_embed,
                       "activation": self.activation,
                       "use_bias": self.use_bias,
                       "is_sorted}": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "node_indexing": self.node_indexing})
        return config
