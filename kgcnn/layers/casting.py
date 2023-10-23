import keras_core as ks
from keras_core.layers import Layer
from keras_core import ops
from kgcnn.ops.core import repeat_static_length
from kgcnn.ops.scatter import scatter_reduce_sum
from kgcnn import __indices_axis__ as global_axis_indices


def pad_left(t):
    return ops.pad(t, [[1, 0]] + [[0, 0] for _ in range(len(ops.shape(t)) - 1)])


def cat_one(t):
    return ops.concatenate([ops.convert_to_tensor([1], dtype=t.dtype), t], axis=0)


class _CastDisjointBase(Layer):

    def __init__(self, reverse_indices: bool = False, dtype_batch: str = "int64", dtype_index=None,
                 padded_disjoint: bool = False, ragged: bool = False, **kwargs):
        r"""Initialize layer.

        Args:
            reverse_indices (bool): Whether to reverse index order. Default is False.
            dtype_batch (str): Dtype for batch ID tensor. Default is 'int64'.
            dtype_index (str): Dtype for index tensor. Default is None.
            padded_disjoint (bool): Whether to keep padding in disjoint representation. Default is False.
            ragged (bool): Whether input is ragged. Default is False.
        """
        super(_CastDisjointBase, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.dtype_index = dtype_index
        self.dtype_batch = dtype_batch
        self._has_ragged_input = ragged
        self.padded_disjoint = padded_disjoint
        self.supports_jit = padded_disjoint

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(_CastDisjointBase, self).get_config()
        config.update({"reverse_indices": self.reverse_indices, "dtype_batch": self.dtype_batch,
                       "dtype_index": self.dtype_index, "padded_disjoint": self.padded_disjoint,
                       "ragged": self._has_ragged_input})
        return config


class CastBatchedIndicesToDisjoint(_CastDisjointBase):
    r"""Cast batched node and edge indices to a (single) disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .
    For PyG a batch of graphs is represented by single graph which contains disjoint sub-graphs,
    and the batch information is passed as batch ID tensor: `graph_id_node` and `graph_id_edge` .

    Keras layers can pass unstacked tensors without batch dimension, however, for model input and output
    batched tensors is most natural to the framework. Therefore, this layer can cast to disjoint from ragged or padded
    input and also keep padding in disjoint representation for jax.

    For padded disjoint all padded nodes are assigned to a padded first empty graph, with single node and at least
    a single self-loop. This graph therefore does not interact with the actual graphs in the message passing.

    .. warning::

        However, for special operations such as :obj:`GraphBatchNormalization` the information of :obj:`padded_disjoint`
        must be separately provided, otherwise this will lead to unwanted behaviour.
    """

    def __init__(self,  **kwargs):
        super(CastBatchedIndicesToDisjoint, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        # Not variables or sub-layers. Nothing to build.
        self.built = True

    def compute_output_spec(self, inputs_spec):
        """Compute output spec as possible."""
        output_shape = self.compute_output_shape([x.shape for x in inputs_spec])
        dtype_batch = inputs_spec[2].dtype if self.dtype_batch is None else self.dtype_batch
        dtype_index = inputs_spec[1].dtype if self.dtype_index is None else self.dtype_index
        output_dtypes = [inputs_spec[0].dtype, dtype_index, dtype_batch, dtype_batch, dtype_batch, dtype_batch,
                         dtype_batch, dtype_batch]
        output_spec = [ks.KerasTensor(s, dtype=d) for s, d in zip(output_shape, output_dtypes)]
        return output_spec

    def compute_output_shape(self, input_shape):
        """Compute output shape as possible."""
        in_n, in_i, in_size_n, in_size_e = input_shape

        if not self.padded_disjoint:
            out_n = tuple([None] + list(in_n[2:]))
            out_i = tuple(list(reversed(in_i[2:])) + [None])
            out_gn, out_ge, out_id_n, out_id_e = (None, ), (None, ), (None, ), (None, )
            out_size_n, out_size_e = in_size_n, in_size_e
        else:
            out_n = tuple([in_n[0]*in_n[1]+1 if in_n[0] is not None and in_n[1] is not None else None] + list(in_n[2:]))
            out_i = tuple(list(reversed(in_i[2:])) + [
                in_i[0]*in_i[1]+1 if in_i[0] is not None and in_i[1] is not None else None])
            out_gn = (None, ) if out_n[0] is None else out_n[:1]
            out_ge = (None, ) if out_i[-1] is None else tuple([out_i[-1]])
            out_id_n = (None, ) if out_n[0] is None else out_n[:1]
            out_id_e = (None, ) if out_i[-1] is None else tuple([out_i[-1]])
            out_size_n = (in_size_n[0]+1, ) if in_size_n[0] is not None else (None, )
            out_size_e = (in_size_e[0]+1, ) if in_size_e[0] is not None else (None, )

        return out_n, out_i, out_gn, out_ge, out_id_n, out_id_e, out_size_n, out_size_e

    def call(self, inputs: list, **kwargs):
        r"""Changes node and edge indices into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[nodes, edge_indices, nodes_in_batch, edges_in_batch]` ,

                - nodes (Tensor): Node features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes.
                - edge_indices (Tensor): Edge index list have shape `(batch, M, 2)` with the indices of M directed
                  edges at last axis for each edge.
                - total_nodes (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - total_edges (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .


        Returns:
            list: `[node_attr, edge_index, graph_id_node, graph_id_edge, node_id, edge_id, nodes_count, edges_count]`

                - node_attr (Tensor): Represents node attributes or coordinates of shape `([N], F, ...)` ,
                - edge_index (Tensor): Represents the index table of shape `(2, [M])` for directed edges.
                - graph_id_node (Tensor): ID tensor of graph assignment in disjoint graph of shape `([N], )` .
                - graph_id_edge (Tensor): ID tensor of graph assignment in disjoint graph of shape `([M], )` .
                - nodes_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
                - edges_id (Tensor): The ID-tensor to assign each edge to its respective batch of shape `([M], )` .
                - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .
        """
        nodes, edge_indices, node_len, edge_len = inputs

        # Case: Ragged Tensor input.
        # As soon as ragged tensors are supported by Keras-Core. We will add this here to simply extract the disjoint
        # graph representation.
        if self._has_ragged_input:
            raise NotImplementedError("Ragged or sparse input is not supported yet for '%s'." % self.name)

        if self.dtype_batch is None:
            dtype_batch = node_len.dtype
        else:
            dtype_batch = self.dtype_batch
            node_len = ops.cast(node_len, dtype=dtype_batch)
            edge_len = ops.cast(edge_len, dtype=dtype_batch)

        if self.dtype_index is not None:
            edge_indices = ops.cast(edge_indices, dtype=self.dtype_index)

        node_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=dtype_batch), axis=0),
                             ops.shape(node_len)[0], axis=0)
        edge_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(edge_indices)[1], dtype=dtype_batch), axis=0),
                             ops.shape(edge_len)[0], axis=0)

        # def make_mask_flatten(len_per_dim, target_shape):
        #     mask = ops.reshape(ops.repeat(
        #         ops.convert_to_tensor([[True, False]], dtype="bool"), ops.shape(len_per_dim)[0], axis=0), (-1,))
        #     mask = ops.repeat(mask, ops.reshape(ops.concatenate([ops.expand_dims(len_per_dim, axis=-1),
        #            ops.expand_dims(target_shape[1] - len_per_dim, axis=-1)], axis=-1), [-1]), axis=0)
        #     return mask

        node_mask = node_id < ops.expand_dims(node_len, axis=-1)
        edge_mask = edge_id < ops.expand_dims(edge_len, axis=-1)

        if not self.padded_disjoint:
            edge_indices_flatten = edge_indices[edge_mask]
            nodes_flatten = nodes[node_mask]
            node_id = node_id[node_mask]
            edge_id = edge_id[edge_mask]
            node_splits = ops.pad(ops.cumsum(node_len), [[1, 0]])
            graph_id_node = ops.repeat(ops.arange(ops.shape(node_len)[0], dtype=dtype_batch), node_len)
            graph_id_edge = ops.repeat(ops.arange(ops.shape(edge_len)[0], dtype=dtype_batch), edge_len)
            offset_edge_indices = ops.expand_dims(ops.repeat(node_splits[:-1], edge_len), axis=-1)
            offset_edge_indices = ops.broadcast_to(offset_edge_indices, ops.shape(edge_indices_flatten))
            disjoint_indices = edge_indices_flatten + ops.cast(offset_edge_indices, edge_indices_flatten.dtype)
        else:
            nodes_flatten = ops.reshape(nodes, [-1] + list(ops.shape(nodes)[2:]))
            edge_indices_flatten = ops.reshape(edge_indices, [-1] + list(ops.shape(edge_indices)[2:]))
            node_len_flat = ops.repeat(ops.cast([ops.shape(nodes)[1]], dtype=dtype_batch), ops.shape(nodes)[0])
            edge_len_flat = ops.repeat(ops.cast([ops.shape(edge_indices)[1]], dtype=dtype_batch),
                                       ops.shape(edge_indices)[0])
            edge_mask_flatten = ops.reshape(edge_mask, [-1])
            node_mask_flatten = ops.reshape(node_mask, [-1])
            node_id = ops.reshape(node_id, [-1])
            edge_id = ops.reshape(edge_id, [-1])

            nodes_flatten = pad_left(nodes_flatten)
            edge_indices_flatten = pad_left(edge_indices_flatten)
            node_id = pad_left(node_id)
            edge_id = pad_left(edge_id)
            node_len_flat = cat_one(node_len_flat)
            edge_len_flat = cat_one(edge_len_flat)
            node_mask_flatten = pad_left(node_mask_flatten)
            edge_mask_flatten = pad_left(edge_mask_flatten)

            graph_id_node = repeat_static_length(
                ops.arange(ops.shape(node_len_flat)[0], dtype=dtype_batch), node_len_flat,
                total_repeat_length=ops.shape(nodes_flatten)[0])
            graph_id_edge = repeat_static_length(
                ops.arange(ops.shape(edge_len_flat)[0], dtype=dtype_batch), edge_len_flat,
                total_repeat_length=ops.shape(edge_indices_flatten)[0])
            graph_id_node = ops.where(node_mask_flatten, graph_id_node, 0)
            graph_id_edge = ops.where(edge_mask_flatten, graph_id_edge, 0)

            node_id = ops.where(node_mask_flatten, node_id, 0)
            edge_id = ops.where(edge_mask_flatten, edge_id, 0)

            node_splits = ops.pad(ops.cumsum(node_len_flat), [[1, 0]])
            offset_edge_indices = repeat_static_length(
                node_splits[:-1], edge_len_flat, total_repeat_length=ops.shape(edge_indices_flatten)[0])
            offset_edge_indices = ops.expand_dims(offset_edge_indices, axis=-1)
            offset_edge_indices = ops.broadcast_to(offset_edge_indices, ops.shape(edge_indices_flatten))
            disjoint_indices = edge_indices_flatten + ops.cast(offset_edge_indices, edge_indices_flatten.dtype)
            edge_mask_flatten = ops.expand_dims(edge_mask_flatten, axis=-1)
            disjoint_indices = ops.where(edge_mask_flatten, disjoint_indices, 0)
            node_len = ops.concatenate([ops.sum(node_len_flat[1:] - node_len, axis=0, keepdims=True), node_len], axis=0)
            edge_len = ops.concatenate([ops.sum(edge_len_flat[1:] - edge_len, axis=0, keepdims=True), edge_len], axis=0)

        # Transpose edge indices.
        if global_axis_indices == 0:
            disjoint_indices = ops.transpose(disjoint_indices)

        if self.reverse_indices:
            disjoint_indices = ops.flip(disjoint_indices, axis=global_axis_indices)

        return [nodes_flatten, disjoint_indices, graph_id_node, graph_id_edge, node_id, edge_id, node_len, edge_len]


class CastBatchedAttributesToDisjoint(_CastDisjointBase):
    r"""Cast batched node and edge attributes to a (single) disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .

    Only applies a casting of attribute tensors similar to :obj:`CastBatchedIndicesToDisjoint` but without any
    index adjustment. Produces the batch-ID tensor assignment.

    For padded disjoint all padded nodes are assigned to a padded first empty graph, with single node and at least
    a single self-loop. This graph therefore does not interact with the actual graphs in the message passing.

    .. warning::

        However, for special operations such as :obj:`GraphBatchNormalization` the information of :obj:`padded_disjoint`
        must be separately provided, otherwise this will lead to unwanted behaviour.
    """

    def __init__(self, **kwargs):
        super(CastBatchedAttributesToDisjoint, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.built = True

    def compute_output_spec(self, inputs_spec):
        """Compute output spec as possible."""
        output_shape = self.compute_output_shape([x.shape for x in inputs_spec])
        dtype_batch = inputs_spec[1].dtype if self.dtype_batch is None else self.dtype_batch
        output_dtypes = [inputs_spec[0].dtype, dtype_batch, dtype_batch, dtype_batch]
        output_spec = [ks.KerasTensor(s, dtype=d) for s, d in zip(output_shape, output_dtypes)]
        return output_spec

    def compute_output_shape(self, input_shape):
        """Compute output shape as possible."""
        in_n, in_size_n = input_shape
        if not self.padded_disjoint:
            out_n = tuple([None] + list(in_n[2:]))
            out_gn, out_id_n = (None,), (None,)
            out_size_n = in_size_n
        else:
            out_n = tuple(
                [in_n[0] * in_n[1] + 1 if in_n[0] is not None and in_n[1] is not None else None] + list(in_n[2:]))
            out_gn = (None,) if out_n[0] is None else out_n[:1]
            out_id_n = (None,) if out_n[0] is None else out_n[:1]
            out_size_n = (in_size_n[0] + 1,) if in_size_n[0] is not None else (None,)
        return out_n, out_gn, out_id_n, out_size_n

    def call(self, inputs: list, **kwargs):
        """Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[attr, total_attr]` ,

                - attr (Tensor): Features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes or edges.
                - total_attr (Tensor): Tensor of lengths for each graph of shape `(batch, )` .

        Returns:
            list: `[attr, graph_id, item_id, item_counts]` .

                - attr (Tensor): Represents attributes or coordinates of shape `([N], F, ...)`
                - graph_id (Tensor): ID tensor of graph assignment in disjoint graph of shape `([N], )` .
                - item_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
                - item_counts (Tensor): Tensor of lengths for each graph of shape `(batch, )` .
        """
        # Case: Ragged Tensor input.
        # As soon as ragged tensors are supported by Keras-Core. We will add this here to simply extract the disjoint
        # graph representation.
        if self._has_ragged_input:
            raise NotImplementedError("Ragged or sparse input is not supported yet for '%s'." % self.name)

        nodes, node_len = inputs

        if self.dtype_batch is None:
            dtype_batch = node_len.dtype
        else:
            dtype_batch = self.dtype_batch
            node_len = ops.cast(node_len, dtype=dtype_batch)

        node_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=dtype_batch), axis=0),
                             ops.shape(node_len)[0], axis=0)
        node_mask = node_id < ops.expand_dims(node_len, axis=-1)

        if not self.padded_disjoint:
            nodes_flatten = nodes[node_mask]
            graph_id_node = ops.repeat(ops.arange(ops.shape(node_len)[0], dtype=dtype_batch), node_len)
            node_id = node_id[node_mask]
        else:
            nodes_flatten = ops.reshape(nodes, [-1] + list(ops.shape(nodes)[2:]))
            node_len_flat = ops.repeat(ops.cast([ops.shape(nodes)[1]], dtype=dtype_batch), ops.shape(nodes)[0])
            node_mask_flatten = ops.reshape(node_mask, [-1])
            node_id = ops.reshape(node_id, [-1])
            nodes_flatten = pad_left(nodes_flatten)
            node_id = pad_left(node_id)
            node_len_flat = cat_one(node_len_flat)
            node_mask_flatten = pad_left(node_mask_flatten)
            graph_id = repeat_static_length(
                ops.arange(ops.shape(node_len_flat)[0], dtype=self.dtype_batch), node_len_flat,
                total_repeat_length=ops.shape(nodes_flatten)[0])
            graph_id_node = ops.where(node_mask_flatten, graph_id, 0)
            node_id = ops.where(node_mask_flatten, node_id, 0)
            node_len = ops.concatenate([ops.sum(node_len_flat[1:] - node_len, axis=0, keepdims=True), node_len], axis=0)

        return [nodes_flatten, graph_id_node, node_id, node_len]


class CastDisjointToGraphState(_CastDisjointBase):
    r"""Cast graph property tensor from disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .

    The graph state is usually kept as batched tensor, except for padded disjoint representation, an empty zero valued
    graph is removed that represents all padded nodes.
    """

    def __init__(self, **kwargs):
        super(CastDisjointToGraphState, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.padded_disjoint:
            if input_shape[0] is not None:
                return tuple([input_shape[0] - 1] + list(input_shape[1:]))
        return input_shape

    def call(self, inputs: list, **kwargs):
        """Changes graph tensor from disjoint representation.

        Args:
            inputs (Tensor): Graph labels from a disjoint representation of shape `(batch, ...)` or
                `(batch+1, ...)` for padded disjoint.

        Returns:
            Tensor: Graph labels of shape `(batch, ...)` .
        """
        # Simply remove first graph.
        if self.padded_disjoint:
            return inputs[1:]
        return inputs


class CastDisjointToBatchedAttributes(_CastDisjointBase):
    r"""Cast batched node and edge attributes from a (single) disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .

    Reconstructs batched tensor with the help of ID tensor information.
    """

    def __init__(self, **kwargs):
        super(CastDisjointToBatchedAttributes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs: list, **kwargs):
        """Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[target, attr, graph_id_attr, attr_id]` ,

                - target (Tensor): Target tensor of shape equal to desired output of shape `(batch, N, F, ...)` .
                  Only first dimension is required for making new tensor e.g. of shape  `(batch,)` .
                - attr (Tensor): Features are represented by a keras tensor of shape `([N], F, ...)` ,
                  where N denotes the number of nodes or edges.
                - graph_id_attr (Tensor): ID tensor of graph assignment in disjoint graph of shape `([N], )` .
                - attr_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .

        Returns:
            Tensor: Batched output tensor of node or edge attributes of shape `(batch, N, F, ...)` .
        """
        target, attr, graph_id_attr, attr_id = inputs

        if not self.padded_disjoint:
            output_shape = [ops.shape(target)[0]*ops.shape(target)[1]] + list(ops.shape(attr)[1:])
            indices = graph_id_attr*ops.convert_to_tensor(ops.shape(target)[1], dtype=graph_id_attr.dtype) + ops.cast(
                attr_id, dtype=graph_id_attr.dtype)
            out = scatter_reduce_sum(indices, attr, output_shape)
            out = ops.reshape(out, list(ops.shape(target)[:2]) + list(ops.shape(attr)[1:]))
        else:
            output_shape = [(ops.shape(target)[0]+1)*ops.shape(target)[1]] + list(ops.shape(attr)[1:])
            indices = graph_id_attr * ops.convert_to_tensor(ops.shape(target)[1], dtype=graph_id_attr.dtype) + ops.cast(
                attr_id, dtype=graph_id_attr.dtype)
            out = scatter_reduce_sum(indices, attr, output_shape)
            out = out[ops.shape(target)[1]:]
            out = ops.reshape(out, list(ops.shape(target)[:2]) + list(ops.shape(attr)[1:]))
        return out


class CastGraphStateToDisjoint(_CastDisjointBase):
    r"""Cast graph property tensor to disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .

    The graph state is usually kept as batched tensor, except for padded disjoint representation, an empty zero valued
    graph is added to represent all padded nodes.
    """

    def __init__(self, **kwargs):
        super(CastGraphStateToDisjoint, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.padded_disjoint:
            if input_shape[0] is not None:
                return tuple([input_shape[0] + 1] + list(input_shape[1:]))
        return input_shape

    def call(self, inputs: list, **kwargs):
        """Changes graph tensor from disjoint representation.

        Args:
            inputs (Tensor): Graph labels from a disjoint representation of shape `(batch, ...)` .

        Returns:
            Tensor: Graph labels of shape `(batch, ...)` or `(batch+1, ...)` for padded disjoint.
        """
        if self.padded_disjoint:
            return pad_left(inputs)
        return inputs
