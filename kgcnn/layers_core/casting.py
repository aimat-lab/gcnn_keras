from keras_core.layers import Layer
from keras_core import ops
# from keras_core.backend import backend


def pad_left(t):
    return ops.pad(t, [[1, 0]] + [[0, 0] for _ in range(len(ops.shape(t)) - 1)])


def cat_one(t):
    return ops.concatenate([ops.convert_to_tensor([1], dtype=t.dtype), t], axis=0)


class CastBatchedGraphIndicesToDisjoint(Layer):
    """Cast batched node and edge tensors to a (single) disjoint graph representation of Pytorch Geometric (PyG).
    For PyG a batch of graphs is represented by single graph which contains disjoint sub-graphs,
    and the batch information is passed as batch ID tensor: `nodes_id` or `edge_id` .


    Keras layers can pass unstacked tensors without batch dimension, however, for model input and output
    batched tensors is currently built in the framework.
    """

    def __init__(self, reverse_indices: bool = True, dtype_batch: str = "int64", dtype_index=None,
                 padded_disjoint: bool = False, ragged: bool = False, **kwargs):
        super(CastBatchedGraphIndicesToDisjoint, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.dtype_index = dtype_index
        self.dtype_batch = dtype_batch
        self._has_ragged_input = ragged
        self.padded_disjoint = padded_disjoint
        self.supports_jit = padded_disjoint
        
    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        out_shape = [tuple([None] + list(input_shape[0][2:])), tuple(list(reversed(input_shape[1][2:])) + [None]),
            (None, ), (None, ), (None, ), (None, )]
        if len(input_shape) == 5:
            out_shape = out_shape + [tuple([None] + list(input_shape[4][2:]))]
        return out_shape

    def call(self, inputs: list, **kwargs):
        """Changes node and edge indices into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[nodes, edges, edge_indices, nodes_in_batch, edges_in_batch]` ,

                - nodes (Tensor): Node features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes.
                - edge_indices (Tensor): Edge index list have shape `(batch, M, 2)` with the indices of M directed
                  edges at last axis for each edge corresponding to `edges` .
                - total_nodes (Tensor):
                - total_edges (Tensor):


        Returns:
            list: `[node_attr, edge_attr, edge_index, batch, counts]` .

                - node_attr (Tensor): Represents node attributes or coordinates of shape `([N], F, ...)` ,
                - edge_index (Tensor): Represents the index table of shape `(2, [M])` for directed edges.
                - nodes_id (Tensor): The ID-tensor to assign each node to its respective batch of shape `([N], )` .
                - edges_id (Tensor): The ID-tensor to assign each edge to its respective batch of shape `([M], )` .
                - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .
        """
        all_tensor = all([ops.is_tensor(x) for x in inputs])

        # Case: Ragged Tensor input.
        # As soon as ragged tensors are supported by Keras-Core. We will add this here to simply extract the disjoint
        # graph representation.
        if not all_tensor:
            raise NotImplementedError("Ragged or sparse input is not supported yet for '%s'." % self.name)

        if len(inputs) == 4:
            nodes, edge_indices, node_len, edge_len = inputs
            edges = None
        elif len(inputs) == 5:
            nodes, edge_indices, node_len, edge_len, edges = inputs
        else:
            raise ValueError("Wrong number of inputs to layer '%s'. " % self.name)

        node_len = ops.cast(node_len, dtype=self.dtype_batch)
        edge_len = ops.cast(edge_len, dtype=self.dtype_batch)

        node_mask = ops.repeat(
            ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=self.dtype_batch), axis=0),
            ops.shape(node_len)[0], axis=0) < ops.expand_dims(node_len, axis=-1)
        edge_mask = ops.repeat(
            ops.expand_dims(ops.arange(ops.shape(edge_indices)[1], dtype=self.dtype_batch), axis=0),
            ops.shape(edge_len)[0], axis=0) < ops.expand_dims(edge_len, axis=-1)

        if not self.padded_disjoint:
            edge_indices_flatten = edge_indices[edge_mask]
            nodes_flatten = nodes[node_mask]
            edges_flatten = edges[edge_mask] if edges is not None else None
            node_mask_flatten, edge_mask_flatten = None, None
        else:
            nodes_flatten = ops.reshape(nodes, [-1] + list(ops.shape(nodes)[2:]))
            edge_indices_flatten = ops.reshape(edge_indices, [-1] + list(ops.shape(edge_indices)[2:]))
            edges_flatten = ops.reshape(edges, [-1] + list(ops.shape(edges)[2:])) if edges is not None else None

            node_len = ops.repeat(ops.cast([ops.shape(nodes)[1]], dtype=self.dtype_batch), ops.shape(nodes)[0])
            edge_len = ops.repeat(ops.cast([ops.shape(edge_indices)[1]], dtype=self.dtype_batch),
                                  ops.shape(edge_indices)[0])

            edge_mask_flatten = ops.reshape(edge_mask, [-1])
            node_mask_flatten = ops.reshape(node_mask, [-1])

            nodes_flatten = pad_left(nodes_flatten)
            edge_indices_flatten = pad_left(edge_indices_flatten)
            node_len = cat_one(node_len)
            edge_len = cat_one(edge_len)
            node_mask_flatten = pad_left(node_mask_flatten)
            edge_mask_flatten = pad_left(edge_mask_flatten)
            edges_flatten = pad_left(edges_flatten) if edges is not None else None

        if self.dtype_index is not None:
            edge_indices_flatten = ops.cast(edge_indices_flatten, dtype=self.dtype_index)

        nodes_id = ops.repeat(ops.arange(ops.shape(node_len)[0], dtype=self.dtype_batch), node_len)
        edges_id = ops.repeat(ops.arange(ops.shape(edge_len)[0], dtype=self.dtype_batch), edge_len)

        if self.padded_disjoint:
            nodes_id = ops.where(node_mask_flatten, nodes_id, ops.convert_to_tensor(0, dtype=self.dtype_batch))
            edges_id = ops.where(edge_mask_flatten, edges_id, ops.convert_to_tensor(0, dtype=self.dtype_batch))

        node_splits = ops.pad(ops.cumsum(node_len), [[1, 0]])
        offset_edge_indices = ops.expand_dims(ops.repeat(node_splits[:-1], edge_len), axis=-1)
        offset_edge_indices = ops.broadcast_to(offset_edge_indices, ops.shape(edge_indices_flatten))

        disjoint_indices = edge_indices_flatten + ops.cast(offset_edge_indices, edge_indices_flatten.dtype)

        if self.padded_disjoint:
            disjoint_indices = ops.where(
                ops.expand_dims(edge_mask_flatten, axis=-1), disjoint_indices, 0)

        disjoint_indices = ops.transpose(disjoint_indices)
        if self.reverse_indices:
            disjoint_indices = ops.flip(disjoint_indices, axis=0)

        out = [nodes_flatten, disjoint_indices, nodes_id, edges_id, node_len, edge_len]
        if edges is not None:
            out = out + [edges_flatten]
        return out

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(CastBatchedGraphIndicesToDisjoint, self).get_config()
        config.update({"reverse_indices": self.reverse_indices, "dtype_batch": self.dtype_batch,
                       "dtype_index": self.dtype_index})
        return config


class CastBatchedGraphAttributesToDisjoint(Layer):

    def __init__(self, reverse_indices: bool = True, dtype_batch: str = "int64", dtype_index=None,
                 padded_disjoint: bool = False, **kwargs):
        super(CastBatchedGraphAttributesToDisjoint, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.dtype_batch = dtype_batch
        self.padded_disjoint = padded_disjoint
        self.supports_jit = padded_disjoint
        self.dtype_index = dtype_index

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return [tuple([None] + list(input_shape[0][2:])), (None,)]

    def call(self, inputs: list, **kwargs):
        """Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[attr, counts_in_batch]` ,

                - attr (Tensor): Features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes or edges.
                - counts_in_batch (Tensor):

        Returns:
            list: `[node_attr, counts]` .

                - node_attr (Tensor): Represents attributes or coordinates of shape `([N], F, ...)` ,
                - counts_in_batch (Tensor): Tensor of lengths for each graph of shape `(batch, )` .
        """
        all_tensor = all([ops.is_tensor(x) for x in inputs])

        # Case: Ragged Tensor input.
        # As soon as ragged tensors are supported by Keras-Core. We will add this here to simply extract the disjoint
        # graph representation.
        if not all_tensor:
            raise NotImplementedError("Ragged or sparse input is not supported yet for '%s'." % self.name)

        nodes, node_len = inputs
        node_len = ops.cast(node_len, dtype=self.dtype_batch)
        node_mask = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=self.dtype_batch), axis=0),
            ops.shape(node_len)[0], axis=0) < ops.expand_dims(node_len, axis=-1)

        if not self.padded_disjoint:
            nodes_flatten = nodes[node_mask]
            node_mask_flatten = None
        else:
            nodes_flatten = ops.reshape(nodes, [-1] + list(ops.shape(nodes)[2:]))
            node_len = ops.repeat(ops.cast([ops.shape(nodes)[1]], dtype=self.dtype_batch), ops.shape(nodes)[0])
            node_mask_flatten = ops.reshape(node_mask, [-1])
            nodes_flatten = pad_left(nodes_flatten)
            node_len = cat_one(node_len)

        nodes_id = ops.repeat(ops.arange(ops.shape(node_len)[0], dtype=self.dtype_batch), node_len)

        if self.padded_disjoint:
            nodes_id = ops.where(node_mask_flatten, nodes_id, ops.convert_to_tensor(0, dtype=self.dtype_batch))

        return [nodes_flatten, nodes_id, node_len]

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(CastBatchedGraphAttributesToDisjoint, self).get_config()
        config.update({"dtype_batch": self.dtype_batch})
        return config


class CastDisjointToGraphLabels(Layer):

    def __init__(self, reverse_indices: bool = True, dtype_batch: str = "int64", dtype_index=None,
                 padded_disjoint: bool = False, **kwargs):
        super(CastDisjointToGraphLabels, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.dtype_batch = dtype_batch
        self.dtype_index = dtype_index
        self.supports_jit = True
        self.padded_disjoint = padded_disjoint

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return [tuple([None] + list(input_shape[0][2:])), (None,)]

    def call(self, inputs: list, **kwargs):
        """Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[attr, counts_in_batch]` ,

                - attr (Tensor): Features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes or edges.
                - counts_in_batch (Tensor):

        Returns:
            Tensor: Graph labels of shape `(batch, ...)` .
        """
        if self.padded_disjoint:
            return inputs[1:]
        return inputs

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(CastDisjointToGraphLabels, self).get_config()
        config.update({"dtype_batch": self.dtype_batch})
        return config
