from keras_core.layers import Layer
from keras_core import ops


class CastBatchGraphListToPyGDisjoint(Layer):
    """Cast batched node and edge tensors to a (single) disjoint graph representation of Pytorch Geometric (PyG).
    For PyG a batch of graphs is represented by single graph which contains disjoint sub-graphs,
    and the batch information is passed as batch ID tensor: `batch` .

    Keras layers can pass unstacked tensors without batch dimension, however, for model input and output
    batched tensors are preferred.
    """

    def __init__(self, reverse_indices: bool = True, **kwargs):
        super(CastBatchGraphListToPyGDisjoint, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices

    def build(self, input_shape):
        return super(CastBatchGraphListToPyGDisjoint, self).build(input_shape)

    def call(self, inputs: list, **kwargs):
        """Changes node and edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[nodes, edges, edge_indices, nodes_in_batch, edges_in_batch]` ,

                - nodes (Tensor): Node features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                    where N denotes the number of nodes.
                - edges (Tensor): edge features are represented by a keras tensor of shape `(batch, M, F, ...)` ,
                    where M denotes the number of edges.
                - edge_indices (Tensor): Edge indices have shape `(batch, M, 2)` with the indices of directed
                    edges at last axis for each edge corresponding to `edges` .
                - nodes_in_batch (Tensor):
                - edges_in_batch (Tensor):


        Returns:
            list: List of graph tensors in PyG format that is `[node_attr, edge_attr, edge_index, batch]` .

                - node_attr (Tensor): Represents node attributes or coordinates of shape `(batch*N, F, ...)` ,
                - edge_attr (Tensor): Represents edge attributes of shape `(batch*M, F, ...)` and
                - edge_index (Tensor): Represents the index table of shape `(2, batch*M)` for directed edges.
                - batch (Tensor): The ID-tensor to assign each node to its respective batch of shape `(batch*N)` .
        """
        all_tensor = all([ops.is_tensor(x) for x in inputs])

        # Case 1: Padded node and edges tensors but with batch dimension at axis 0.
        if all_tensor and len(inputs) == 5:
            nodes, edges, edge_indices, node_len, edge_len = inputs
            node_mask = ops.repeat(ops.expand_dims(ops.arange(
                ops.shape(nodes[1])), axis=0), ops.shape(node_len)[0], axis=0) < ops.expand_dims(node_len, axis=-1)
            edge_mask = ops.repeat(ops.expand_dims(ops.arange(
                ops.shape(nodes[1])), axis=0), ops.shape(node_len)[0], axis=0) < ops.expand_dims(node_len, axis=-1)
            edge_indices_flatten = edge_indices[ops.cast(edge_mask, dtype="bool")]
            nodes_flatten = nodes[ops.cast(node_mask, dtype="bool")]
            edges_flatten = edges[ops.cast(edge_mask, dtype="bool")]

        # Case 2: Ragged Tensor input.
        # As soon as ragged tensors are supported by Keras-Core.

        # Unknown input raises an error.
        else:
            raise NotImplementedError("Inputs type to layer '%s' not supported or wrong format." % self.name)

        # Shift indices and make batch tensor.
        node_splits = ops.pad(ops.cumsum(node_len), [[1, 0]])
        offset_edge_indices = ops.expand_dims(ops.repeat(node_splits[:-1], edge_len), axis=-1)
        offset_edge_indices = ops.broadcast_to(offset_edge_indices, ops.shape(edge_indices_flatten))
        batch = ops.repeat(ops.arange(ops.shape(node_len)[0]), node_len)
        disjoint_indices = edge_indices_flatten + ops.cast(offset_edge_indices, edge_indices_flatten.dtype)
        disjoint_indices = ops.transpose(disjoint_indices)

        if self.reverse_indices:
            disjoint_indices = ops.flip(disjoint_indices, axis=0)

        return [nodes_flatten, edges_flatten, disjoint_indices, batch]

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(CastBatchGraphListToPyGDisjoint, self).get_config()
        config.update({"reverse_indices": self.reverse_indices})
        return config
