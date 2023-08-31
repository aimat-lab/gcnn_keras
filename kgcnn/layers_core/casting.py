from keras_core.layers import Layer
from keras_core import ops


class CastBatchGraphListToPyGDisjoint(Layer):
    """Cast batched node and edge tensors to a (single) disjoint graph representation of Pytorch Geometric (PyG).
    For PyG a batch of graphs is represented by single graph which contains disjoint sub-graphs,
    and the batch information is passed as batch ID tensor: `batch` .

    Keras layers can pass unstacked tensors without batch dimension, however, for model input and output
    batched tensors are preferred.
    """

    def __init__(self, reverse_indices: bool = True, batch_dtype: str = "int64",
                 batch_info: str = "lengths", **kwargs):
        super(CastBatchGraphListToPyGDisjoint, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.batch_dtype = batch_dtype
        self.batch_info = batch_info
        assert batch_info in ["lengths", "mask"], "Wrong format for batch information tensor to expect in call()."

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
            if self.batch_info == "lengths":
                nodes, edges, edge_indices, node_len, edge_len = inputs
                node_len = ops.cast(node_len, dtype=self.batch_dtype)
                edge_len = ops.cast(edge_len, dtype=self.batch_dtype)
                node_mask = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=self.batch_dtype), axis=0),
                                       ops.shape(node_len)[0], axis=0) < ops.expand_dims(node_len, axis=-1)
                edge_mask = ops.repeat(ops.expand_dims(ops.arange(ops.shape(edges)[1], dtype=self.batch_dtype), axis=0),
                                       ops.shape(edge_len)[0], axis=0) < ops.expand_dims(edge_len, axis=-1)
                edge_indices_flatten = edge_indices[edge_mask]
                nodes_flatten = nodes[node_mask]
                edges_flatten = edges[edge_mask]
            elif self.batch_info == "mask":
                nodes, edges, edge_indices, node_mask, edge_mask = inputs
                edge_indices_flatten = edge_indices[ops.cast(edge_mask, dtype="bool")]
                nodes_flatten = nodes[ops.cast(node_mask, dtype="bool")]
                edges_flatten = edges[ops.cast(edge_mask, dtype="bool")]
                node_len = ops.sum(ops.cast(node_mask, dtype=self.batch_dtype), axis=1)
                edge_len = ops.sum(ops.cast(edge_mask, dtype=self.batch_dtype), axis=1)
            else:
                raise NotImplementedError("Unknown batch information '%s'." % b)

        # Case 2: Fixed sized graphs without batch information.
        elif all_tensor and len(inputs) == 3:
            nodes, edges, edge_indices = inputs
            n_shape, e_shape, ei_shape = ops.shape(nodes), ops.shape(edges), ops.shape(edge_indices)
            nodes_flatten = ops.reshape(nodes, ops.concatenate([[n_shape[0] * n_shape[1]], n_shape[2:]]))
            edges_flatten = ops.reshape(edges, ops.concatenate([[e_shape[0] * e_shape[1]], e_shape[2:]]))
            edge_indices_flatten = ops.reshape(
                edge_indices, ops.concatenate([[ei_shape[0] * ei_shape[1]], ei_shape[2:]]))
            node_len = ops.repeat(ops.cast([n_shape[1]], dtype=self.batch_dtype), n_shape[0])
            edge_len = ops.repeat(ops.cast([ei_shape[1]], dtype=self.batch_dtype), ei_shape[0])

        # Case: Ragged Tensor input.
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
