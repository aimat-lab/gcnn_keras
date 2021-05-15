import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition


class GatherNodes(GraphBaseLayer):
    """Gather nodes by node indices.

    An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    """

    def __init__(self,
                 concat_nodes=True,
                 **kwargs):
        """Initialize layer."""
        super(GatherNodes, self).__init__(**kwargs)
        self.concat_nodes = concat_nodes

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, edge_index]

            - nodes (tf.ragged): Node embeddings of shape (batch, [N], F)
            - edge_index (tf.ragged): Node indices for edges of shape (batch, [M], 2)

        Returns:
            embeddings: Gathered node embeddings that match the number of edges.
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 2)
        # We cast to values here
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge_index, edge_part = dyn_inputs[1].values, dyn_inputs[1].row_lengths()

        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node="row_splits",
                                                                           partition_type_edge="row_length",
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        out = tf.gather(node, indexlist, axis=0)
        if self.concat_nodes:
            out = tf.keras.backend.concatenate([out[:, i] for i in range(edge_index.shape[-1])], axis=1)
        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index, batch_dims=1) # Works now
        # if self.concat_nodes:
        #   out = tf.keras.backend.concatenate([out[:, :, i] for i in range(edge_index.shape[-1])], axis=2)
        out = self._kgcnn_map_output_ragged([out, edge_part], "row_length", 1)
        return out

    def get_config(self):
        """Update config."""
        config = super(GatherNodes, self).get_config()
        config.update({"concat_nodes": self.concat_nodes})
        return config


class GatherNodesOutgoing(GraphBaseLayer):
    """Gather nodes by indices.

    For outgoing nodes, layer uses only index[1]. An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherNodesOutgoing, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesOutgoing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, edge_index]

            - nodes (tf.ragged): Node embeddings of shape (batch, [N], F)
            - edge_index (tf.ragged): Node indices for edges of shape (batch, [M], 2)

        Returns:
            embeddings: Gathered node embeddings that match the number of edges.
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 2)

        # We cast to values here
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge_index, edge_part = dyn_inputs[1].values, dyn_inputs[1].row_lengths()

        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node="row_splits",
                                                                           partition_type_edge="row_length",
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index[:, :, 1], batch_dims=1)
        out = tf.gather(node, indexlist[:, 1], axis=0)

        out = self._kgcnn_map_output_ragged([out, edge_part], "row_length", 1)
        return out

    def get_config(self):
        """Update config."""
        config = super(GatherNodesOutgoing, self).get_config()
        return config


class GatherNodesIngoing(GraphBaseLayer):
    """Gather nodes by edge edge_indices.
    
    For ingoing nodes, layer uses only index[0]. An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherNodesIngoing, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesIngoing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, edge_index]

            - nodes (tf.ragged): Node embeddings of shape (batch, [N], F)
            - edge_index (tf.ragged): Node indices for edges of shape (batch, [M], 2)

        Returns:
            embeddings: Gathered node embeddings that match the number of edges.
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 2)

        # We cast to values here
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge_index, edge_part = dyn_inputs[1].values, dyn_inputs[1].row_lengths()

        # We cast to values here
        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node="row_splits",
                                                                           partition_type_edge="row_length",
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        out = tf.gather(node, indexlist[:, 0], axis=0)
        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index[:, :, 0], batch_dims=1)
        out = self._kgcnn_map_output_ragged([out, edge_part], "row_length", 1)
        return out

    def get_config(self):
        """Update config."""
        config = super(GatherNodesIngoing, self).get_config()
        return config


class GatherState(GraphBaseLayer):
    """Layer to repeat environment or global state for node or edge lists.
    
    To repeat the correct environment for each sample, a tensor with the target length/partition is required.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [state, target]

            - state (tf.tensor): Graph specific embedding tensor. This is tensor of shape (batch, F)
            - target (tf.ragged): Target to collect state for, of shape (batch, [N], F)

        Returns:
            state: Graph embedding with repeated single state for each graph of shape (batch, [N], F).
        """
        dyn_inputs, = self._kgcnn_map_input_ragged([inputs[1]], 1)

        # We cast to values here
        env = inputs[0]
        target_len = dyn_inputs.row_lengths()

        out = tf.repeat(env, target_len, axis=0)
        out = self._kgcnn_map_output_ragged([out, target_len], "row_length", 0)
        return out

    def get_config(self):
        """Update config."""
        config = super(GatherState, self).get_config()
        return config
