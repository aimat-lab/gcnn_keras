from keras_core.layers import Layer
from keras_core import ops
from keras_core.backend import backend


class CastBatchedGraphIndicesToPyGDisjoint(Layer):
    """Cast batched node and edge tensors to a (single) disjoint graph representation of Pytorch Geometric (PyG).
    For PyG a batch of graphs is represented by single graph which contains disjoint sub-graphs,
    and the batch information is passed as batch ID tensor: `batch` .

    Keras layers can pass unstacked tensors without batch dimension, however, for model input and output
    batched tensors are preferred.
    """

    def __init__(self, reverse_indices: bool = True, dtype_batch: str = "int64", dtype_index=None, **kwargs):
        super(CastBatchedGraphIndicesToPyGDisjoint, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.dtype_index = dtype_index
        self.dtype_batch = dtype_batch
        # self.supports_jit = supports_jit
        
    def build(self, input_shape):
        super(CastBatchedGraphIndicesToPyGDisjoint, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [tuple([None] + list(input_shape[0][2:])),
                tuple(list(reversed(input_shape[1][2:])) + [None]),
                (None,), (None,), (None,)]

    def call(self, inputs: list, **kwargs):
        """Changes node and edge indices into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[nodes, edges, edge_indices, nodes_in_batch, edges_in_batch]` ,

                - nodes (Tensor): Node features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes.
                - edge_indices (Tensor): Edge index list have shape `(batch, M, 2)` with the indices of M directed
                  edges at last axis for each edge corresponding to `edges` .
                - nodes_in_batch (Tensor):
                - edges_in_batch (Tensor):


        Returns:
            list: `[node_attr, edge_attr, edge_index, batch, counts]` .

                - node_attr (Tensor): Represents node attributes or coordinates of shape `(batch*N, F, ...)` ,
                - edge_index (Tensor): Represents the index table of shape `(2, batch*M)` for directed edges.
                - batch (Tensor): The ID-tensor to assign each node to its respective batch of shape `(batch*N)` .
                - nodes_in_batch (Tensor): Tensor of lengths for each graph of shape `(batch, )` .
        """
        all_tensor = all([ops.is_tensor(x) for x in inputs])

        if all_tensor:
            nodes, edge_indices, node_len, edge_len = inputs
            node_len = ops.cast(node_len, dtype=self.dtype_batch)
            edge_len = ops.cast(edge_len, dtype=self.dtype_batch)
            node_mask = ops.repeat(
                ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=self.dtype_batch), axis=0),
                ops.shape(node_len)[0], axis=0) < ops.expand_dims(node_len, axis=-1)
            edge_mask = ops.repeat(
                ops.expand_dims(ops.arange(ops.shape(edge_indices)[1], dtype=self.dtype_batch), axis=0),
                ops.shape(edge_len)[0], axis=0) < ops.expand_dims(edge_len, axis=-1)
            edge_indices_flatten = edge_indices[edge_mask]
            nodes_flatten = nodes[node_mask]

        # nodes_flatten = ops.reshape(nodes, [-1] + list(ops.shape(nodes)[2:]))
        # edge_indices_flatten = ops.reshape(edge_indices, [-1] + list(ops.shape(edge_indices)[2:]))
        # node_len = ops.repeat(ops.cast([ops.shape(nodes)[1]], dtype=self.dtype_batch), ops.shape(nodes)[0])
        # edge_len = ops.repeat(ops.cast([ops.shape(edge_indices)[1]], dtype=self.dtype_batch),
        #                       ops.shape(edge_indices)[0])

        # Case: Ragged Tensor input.
        # As soon as ragged tensors are supported by Keras-Core.

        # Unknown input raises an error.
        else:
            raise NotImplementedError("Inputs type to layer '%s' not supported or wrong format." % self.name)

        # Shift indices and make batch tensor.
        node_splits = ops.pad(ops.cumsum(node_len), [[1, 0]])
        offset_edge_indices = ops.expand_dims(ops.repeat(node_splits[:-1], edge_len), axis=-1)
        offset_edge_indices = ops.broadcast_to(offset_edge_indices, ops.shape(edge_indices_flatten))
        batch = ops.repeat(ops.arange(ops.shape(node_len)[0], dtype=self.dtype_batch), node_len)
        if self.dtype_index is not None:
            edge_indices_flatten = ops.cast(edge_indices_flatten, dtype=self.dtype_index)
        disjoint_indices = edge_indices_flatten + ops.cast(offset_edge_indices, edge_indices_flatten.dtype)
        disjoint_indices = ops.transpose(disjoint_indices)

        if self.reverse_indices:
            disjoint_indices = ops.flip(disjoint_indices, axis=0)

        return [nodes_flatten, disjoint_indices, batch, node_len, edge_len]

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(CastBatchedGraphIndicesToPyGDisjoint, self).get_config()
        config.update({"reverse_indices": self.reverse_indices, "dtype_batch": self.dtype_batch,
                       "dtype_index": self.dtype_index})
        return config


class CastBatchedGraphAttributesToPyGDisjoint(Layer):

    def __init__(self, reverse_indices: bool = True, dtype_batch: str = "int64", **kwargs):
        super(CastBatchedGraphAttributesToPyGDisjoint, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.dtype_batch = dtype_batch
        self.supports_jit = False

    def build(self, input_shape):
        super(CastBatchedGraphAttributesToPyGDisjoint, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [tuple([None] + list(input_shape[0][2:])), (None,)]

    def call(self, inputs: list, **kwargs):
        """Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[attr, counts_in_batch]` ,

                - attr (Tensor): Features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes or edges.
                - counts_in_batch (Tensor, optional):

        Returns:
            list: `[node_attr, counts]` .

                - node_attr (Tensor): Represents attributes or coordinates of shape `(batch*N, F, ...)` ,
                - counts_in_batch (Tensor): Tensor of lengths for each graph of shape `(batch, )` .
        """
        # Case 1: Padded node and edges tensors but with batch dimension at axis 0.
        if all([ops.is_tensor(x) for x in inputs]):
            nodes, node_len = inputs
            node_len = ops.cast(node_len, dtype=self.dtype_batch)
            node_mask = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=self.dtype_batch), axis=0),
                                   ops.shape(node_len)[0], axis=0) < ops.expand_dims(node_len, axis=-1)
            nodes_flatten = nodes[node_mask]

        # Case: Ragged Tensor input.
        # As soon as ragged tensors are supported by Keras-Core.

        # Unknown input raises an error.
        else:
            raise NotImplementedError("Inputs type to layer '%s' not supported or wrong format." % self.name)

        return [nodes_flatten, node_len]

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(CastBatchedGraphAttributesToPyGDisjoint, self).get_config()
        config.update({"dtype_batch": self.dtype_batch})
        return config
