import keras as ks
from keras.layers import Layer
from keras import ops
from kgcnn.ops.core import repeat_static_length, decompose_ragged_tensor
from kgcnn.ops.scatter import scatter_reduce_sum
from kgcnn import __indices_axis__ as global_axis_indices


def _pad_left(t):
    return ops.pad(t, [[1, 0]] + [[0, 0] for _ in range(len(ops.shape(t)) - 1)])


def _cat_one(t):
    return ops.concatenate([ops.convert_to_tensor([1], dtype=t.dtype), t], axis=0)


class _CastBatchedDisjointBase(Layer):

    def __init__(self,
                 reverse_indices: bool = False,
                 dtype_batch: str = "int64",
                 dtype_index=None,
                 padded_disjoint: bool = False,
                 uses_mask: bool = False,
                 static_batched_node_output_shape: tuple = None,
                 static_batched_edge_output_shape: tuple = None,
                 remove_padded_disjoint_from_batched_output: bool = True,
                 **kwargs):
        r"""Initialize layer.

        Args:
            reverse_indices (bool): Whether to reverse index order. Default is False.
            dtype_batch (str): Dtype for batch ID tensor. Default is 'int64'.
            dtype_index (str): Dtype for index tensor. Default is None.
            padded_disjoint (bool): Whether to keep padding in disjoint representation. Default is False.
                Not used for ragged arguments.
            uses_mask (bool): Whether the padding is marked by a boolean mask or by a length tensor, counting the
                non-padded nodes from index 0. Default is False.
                Not used for ragged arguments.
            static_batched_node_output_shape (tuple): Statical output shape of nodes. Default is None.
                Not used for ragged arguments.
            static_batched_edge_output_shape (tuple): Statical output shape of edges. Default is None.
                Not used for ragged arguments.
            remove_padded_disjoint_from_batched_output (bool): Whether to remove the first element on batched output
                in case of padding.
                Not used for ragged arguments.
        """
        super(_CastBatchedDisjointBase, self).__init__(**kwargs)
        self.reverse_indices = reverse_indices
        self.dtype_index = dtype_index
        self.dtype_batch = dtype_batch
        self.uses_mask = uses_mask
        self.padded_disjoint = padded_disjoint
        if padded_disjoint:
            self.supports_jit = True
        self.static_batched_node_output_shape = static_batched_node_output_shape
        self.static_batched_edge_output_shape = static_batched_edge_output_shape
        self.remove_padded_disjoint_from_batched_output = remove_padded_disjoint_from_batched_output

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(_CastBatchedDisjointBase, self).get_config()
        config.update({"reverse_indices": self.reverse_indices, "dtype_batch": self.dtype_batch,
                       "dtype_index": self.dtype_index, "padded_disjoint": self.padded_disjoint,
                       "uses_mask": self.uses_mask,
                       "static_batched_node_output_shape": self.static_batched_node_output_shape,
                       "static_batched_edge_output_shape": self.static_batched_edge_output_shape,
                       "remove_padded_disjoint_from_batched_output": self.remove_padded_disjoint_from_batched_output
                       })
        return config


class CastBatchedIndicesToDisjoint(_CastBatchedDisjointBase):
    r"""Cast batched node and edge indices to a (single) disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .
    For PyG a batch of graphs is represented by single graph which contains disjoint sub-graphs,
    and the batch information is passed as batch ID tensor: `graph_id_node` and `graph_id_edge` .

    Keras layers can pass unstacked tensors without batch dimension, however, for model input and output
    batched tensors is most natural to the framework. Therefore, this layer can cast to disjoint from padded
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
        dtype_batch = inputs_spec[2].dtype if self.dtype_batch is None and not self.uses_mask else self.dtype_batch
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
        else:
            out_n = tuple([in_n[0]*in_n[1]+1 if in_n[0] is not None and in_n[1] is not None else None] + list(in_n[2:]))

        out_gn = (None, ) if out_n[0] is None else out_n[:1]
        out_id_n = (None, ) if out_n[0] is None else out_n[:1]

        if not self.padded_disjoint:
            out_i = tuple([None] + list(in_i[2:]))
        else:
            out_i = tuple(
                [in_i[0] * in_i[1] + 1 if in_i[0] is not None and in_i[1] is not None else None] + list(in_i[2:]))

        out_ge = (None,) if out_i[0] is None else tuple([out_i[0]])
        out_id_e = (None,) if out_i[0] is None else tuple([out_i[0]])

        if global_axis_indices == 0:
            out_i = tuple(reversed(list(out_i)))

        batch_dim_n = in_size_n[0] if not self.uses_mask else in_n[0]
        batch_dim_e = in_size_e[0] if not self.uses_mask else in_i[0]
        if self.padded_disjoint:
            out_size_n = (batch_dim_n+1, ) if batch_dim_n is not None else (None, )
            out_size_e = (batch_dim_e+1, ) if batch_dim_e is not None else (None, )
        else:
            out_size_n, out_size_e = (batch_dim_n, ), (batch_dim_e, )

        return out_n, out_i, out_gn, out_ge, out_id_n, out_id_e, out_size_n, out_size_e

    def call(self, inputs: list, **kwargs):
        r"""Changes node and edge indices into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[nodes, edge_indices, nodes_in_batch/node_mask, edges_in_batch/edge_mask]` ,

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
                - graph_id_node (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - graph_id_edge (Tensor): ID tensor of batch assignment in disjoint graph of shape `([M], )` .
                - nodes_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
                - edges_id (Tensor): The ID-tensor to assign each edge to its respective graph of shape `([M], )` .
                - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .
        """
        nodes, edge_indices, node_pad, edge_pad = inputs

        # def make_mask_flatten(len_per_dim, target_shape):
        #     mask = ops.reshape(ops.repeat(
        #         ops.convert_to_tensor([[True, False]], dtype="bool"), ops.shape(len_per_dim)[0], axis=0), (-1,))
        #     mask = ops.repeat(mask, ops.reshape(ops.concatenate([ops.expand_dims(len_per_dim, axis=-1),
        #            ops.expand_dims(target_shape[1] - len_per_dim, axis=-1)], axis=-1), [-1]), axis=0)
        #     return mask

        if self.dtype_index is not None:
            edge_indices = ops.cast(edge_indices, dtype=self.dtype_index)

        if self.dtype_batch is None:
            if self.uses_mask:
                raise ValueError("Require `dtype_batch` for batch ID tensor when using boolean mask.")
            dtype_batch = node_pad.dtype
        else:
            dtype_batch = self.dtype_batch

        if not self.uses_mask:
            node_len = ops.cast(node_pad, dtype=dtype_batch)
            edge_len = ops.cast(edge_pad, dtype=dtype_batch)
            node_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=dtype_batch), axis=0),
                                 ops.shape(node_len)[0], axis=0)
            edge_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(edge_indices)[1], dtype=dtype_batch), axis=0),
                                 ops.shape(edge_len)[0], axis=0)
            node_mask = node_id < ops.expand_dims(node_len, axis=-1)
            edge_mask = edge_id < ops.expand_dims(edge_len, axis=-1)
        else:
            node_mask = node_pad
            edge_mask = edge_pad
            node_len = ops.sum(ops.cast(node_mask, dtype=dtype_batch), axis=1)
            edge_len = ops.sum(ops.cast(edge_mask, dtype=dtype_batch), axis=1)
            node_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=dtype_batch), axis=0),
                                 ops.shape(node_len)[0], axis=0)
            edge_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(edge_indices)[1], dtype=dtype_batch), axis=0),
                                 ops.shape(edge_len)[0], axis=0)

        if not self.padded_disjoint:
            edge_indices_flatten = edge_indices[edge_mask]
            nodes_flatten = nodes[node_mask]
            node_id = node_id[node_mask]
            edge_id = edge_id[edge_mask]
            node_splits = ops.pad(ops.cumsum(node_len), [[1, 0]])
            graph_id_node = ops.repeat(ops.arange(ops.shape(node_len)[0], dtype=dtype_batch), node_len)
            graph_id_edge = ops.repeat(ops.arange(ops.shape(edge_len)[0], dtype=dtype_batch), edge_len)
            # offset_edge_indices = ops.expand_dims(ops.repeat(node_splits[:-1], edge_len), axis=-1)
            offset_edge_indices = ops.expand_dims(ops.take(node_splits, graph_id_edge, axis=0), axis=-1)
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

            nodes_flatten = _pad_left(nodes_flatten)
            edge_indices_flatten = _pad_left(edge_indices_flatten)
            node_id = _pad_left(node_id)
            edge_id = _pad_left(edge_id)
            node_len_flat = _cat_one(node_len_flat)
            edge_len_flat = _cat_one(edge_len_flat)
            node_mask_flatten = _pad_left(node_mask_flatten)
            edge_mask_flatten = _pad_left(edge_mask_flatten)

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
            # offset_edge_indices = repeat_static_length(node_splits[:-1], edge_len_flat, total_repeat_length=ops.shape(edge_indices_flatten)[0])
            offset_edge_indices = ops.take(node_splits, graph_id_edge, axis=0)
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


CastBatchedIndicesToDisjoint.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__


class CastBatchedAttributesToDisjoint(_CastBatchedDisjointBase):
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
        dtype_batch = inputs_spec[1].dtype if self.dtype_batch is None and not self.uses_mask else self.dtype_batch
        output_dtypes = [inputs_spec[0].dtype, dtype_batch, dtype_batch, dtype_batch]
        output_spec = [ks.KerasTensor(s, dtype=d) for s, d in zip(output_shape, output_dtypes)]
        return output_spec

    def compute_output_shape(self, input_shape):
        """Compute output shape as possible."""
        in_n, in_size_n = input_shape
        if not self.padded_disjoint:
            out_n = tuple([None] + list(in_n[2:]))
        else:
            out_n = tuple(
                [in_n[0] * in_n[1] + 1 if in_n[0] is not None and in_n[1] is not None else None] + list(in_n[2:]))
        out_gn = (None,) if out_n[0] is None else out_n[:1]
        out_id_n = (None,) if out_n[0] is None else out_n[:1]

        batch_dim_n = in_size_n[0] if not self.uses_mask else in_n[0]
        if self.padded_disjoint:
            out_size_n = (batch_dim_n + 1,) if batch_dim_n is not None else (None,)
        else:
            out_size_n = (batch_dim_n,)
        return out_n, out_gn, out_id_n, out_size_n

    def call(self, inputs: list, **kwargs):
        r"""Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[attr, total_attr/mask_attr]` ,

                - attr (Tensor): Features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes or edges.
                - total_attr (Tensor): Tensor of lengths for each graph of shape `(batch, )` .

        Returns:
            list: `[attr, graph_id, item_id, item_counts]` .

                - attr (Tensor): Represents attributes or coordinates of shape `([N], F, ...)`
                - graph_id (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - item_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
                - item_counts (Tensor): Tensor of lengths for each graph of shape `(batch, )` .
        """
        nodes, node_pad = inputs

        if self.dtype_batch is None:
            if self.uses_mask:
                raise ValueError("Require `dtype_batch` for batch ID tensor when using boolean mask.")
            dtype_batch = node_pad.dtype
        else:
            dtype_batch = self.dtype_batch

        if not self.uses_mask:
            node_len = ops.cast(node_pad, dtype=dtype_batch)
            node_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=dtype_batch), axis=0),
                             ops.shape(node_len)[0], axis=0)
            node_mask = node_id < ops.expand_dims(node_len, axis=-1)
        else:
            node_mask = node_pad
            node_len = ops.sum(ops.cast(node_mask, dtype=dtype_batch), axis=1)
            node_id = ops.repeat(ops.expand_dims(ops.arange(ops.shape(nodes)[1], dtype=dtype_batch), axis=0),
                                 ops.shape(node_len)[0], axis=0)

        if not self.padded_disjoint:
            nodes_flatten = nodes[node_mask]
            graph_id_node = ops.repeat(ops.arange(ops.shape(node_len)[0], dtype=dtype_batch), node_len)
            node_id = node_id[node_mask]
        else:
            nodes_flatten = ops.reshape(nodes, [-1] + list(ops.shape(nodes)[2:]))
            node_len_flat = ops.repeat(ops.cast([ops.shape(nodes)[1]], dtype=dtype_batch), ops.shape(nodes)[0])
            node_mask_flatten = ops.reshape(node_mask, [-1])
            node_id = ops.reshape(node_id, [-1])
            nodes_flatten = _pad_left(nodes_flatten)
            node_id = _pad_left(node_id)
            node_len_flat = _cat_one(node_len_flat)
            node_mask_flatten = _pad_left(node_mask_flatten)
            graph_id = repeat_static_length(
                ops.arange(ops.shape(node_len_flat)[0], dtype=self.dtype_batch), node_len_flat,
                total_repeat_length=ops.shape(nodes_flatten)[0])
            graph_id_node = ops.where(node_mask_flatten, graph_id, 0)
            node_id = ops.where(node_mask_flatten, node_id, 0)
            node_len = ops.concatenate([ops.sum(node_len_flat[1:] - node_len, axis=0, keepdims=True), node_len], axis=0)

        return [nodes_flatten, graph_id_node, node_id, node_len]


CastBatchedAttributesToDisjoint.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__


class CastDisjointToBatchedAttributes(_CastBatchedDisjointBase):
    r"""Cast batched node and edge attributes from a (single) disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .

    Reconstructs batched tensor with the help of ID tensor information.
    """

    def __init__(self, static_output_shape: tuple = None, return_mask: bool = False, **kwargs):
        super(CastDisjointToBatchedAttributes, self).__init__(**kwargs)
        self.static_output_shape = static_output_shape
        self.return_mask = return_mask

    def build(self, input_shape):
        self.built = True

    def call(self, inputs: list, **kwargs):
        r"""Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[attr, graph_id_attr, (attr_id), attr_counts]` ,

                - attr (Tensor): Features are represented by a keras tensor of shape `([N], F, ...)` ,
                  where N denotes the number of nodes or edges.
                - graph_id_attr (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - attr_id (Tensor, optional): The ID-tensor to assign each node to its respective graph
                  of shape `([N], )` . For padded disjoint graphs this is required.
                - attr_counts (Tensor): Tensor of lengths for each graph of shape `(batch, )` .

        Returns:
            Tensor: Batched output tensor of node or edge attributes of shape `(batch, N, F, ...)` .
        """
        if len(inputs) == 4:
            attr, graph_id_attr, attr_id, attr_len = inputs
        else:
            attr, graph_id_attr, attr_len = inputs
            attr_id = None

        if self.static_output_shape is not None:
            target_shape = (ops.shape(attr_len)[0], self.static_output_shape[0])
        else:
            target_shape = (ops.shape(attr_len)[0], ops.cast(ops.amax(attr_len), dtype="int32"))
        out_mask = None

        if not self.padded_disjoint:
            if attr_id is None:
                attr_id = ops.arange(0, ops.shape(graph_id_attr)[0], dtype=graph_id_attr.dtype)
                attr_splits = ops.pad(ops.cumsum(attr_len), [[1, 0]])
                attr_id = attr_id - repeat_static_length(attr_splits[:1], attr_len, ops.shape(graph_id_attr)[0])
        else:
            if attr_id is None:
                # Required because padded graphs in the general case can have padded nodes inbetween batches.
                raise ValueError("Require sub-graph IDs in addition to batch IDs for padded disjoint graphs.")

        output_shape = tuple([target_shape[0] * target_shape[1]] + list(ops.shape(attr)[1:]))
        indices = graph_id_attr * ops.convert_to_tensor(target_shape[1], dtype=graph_id_attr.dtype) + ops.cast(
            attr_id, dtype=graph_id_attr.dtype)
        out = scatter_reduce_sum(indices, attr, output_shape)
        out = ops.reshape(out, list(target_shape[:2]) + list(ops.shape(attr)[1:]))
        if self.return_mask:
            output_mask_shape = output_shape[:1]
            out_mask = scatter_reduce_sum(indices, ops.ones(ops.shape(attr)[0], dtype="bool"), output_mask_shape)
            out_mask = ops.reshape(out_mask, list(target_shape[:2]))

        if self.padded_disjoint and self.remove_padded_disjoint_from_batched_output:
            out = out[1:]
            if self.return_mask:
                out_mask = out_mask[1:]

        if self.return_mask:
            return out, out_mask
        return out

    def get_config(self):
        """Get config dictionary for this layer."""
        config = super(_CastBatchedDisjointBase, self).get_config()
        config.update({"static_output_shape": self.static_output_shape, "return_mask": self.return_mask})
        return config


CastDisjointToBatchedAttributes.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__


class CastDisjointToBatchedGraphState(_CastBatchedDisjointBase):
    r"""Cast graph property tensor from disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .

    The graph state is usually kept as batched tensor, except for padded disjoint representation, an empty zero valued
    graph is removed that represents all padded nodes.
    """

    def __init__(self, **kwargs):
        super(CastDisjointToBatchedGraphState, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.padded_disjoint and self.remove_padded_disjoint_from_batched_output:
            if input_shape[0] is not None:
                return tuple([input_shape[0] - 1] + list(input_shape[1:]))
        return input_shape

    def call(self, inputs: list, **kwargs):
        r"""Changes graph tensor from disjoint representation.

        Args:
            inputs (Tensor): Graph labels from a disjoint representation of shape `(batch, ...)` or
                `(batch+1, ...)` for padded disjoint.

        Returns:
            Tensor: Graph labels of shape `(batch, ...)` .
        """
        # Simply remove first graph.
        if self.padded_disjoint and self.remove_padded_disjoint_from_batched_output:
            return inputs[1:]
        return inputs


CastDisjointToBatchedGraphState.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__


class CastBatchedGraphStateToDisjoint(_CastBatchedDisjointBase):
    r"""Cast graph property tensor to disjoint graph representation of
    `Pytorch Geometric (PyG) <https://github.com/pyg-team/pytorch_geometric>`__ .

    The graph state is usually kept as batched tensor, except for padded disjoint representation, an empty zero valued
    graph is added to represent all padded nodes.
    """

    def __init__(self, **kwargs):
        super(CastBatchedGraphStateToDisjoint, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.padded_disjoint:
            if input_shape[0] is not None:
                return tuple([input_shape[0] + 1] + list(input_shape[1:]))
        return input_shape

    def compute_output_spec(self, input_spec):
        return ks.KerasTensor(self.compute_output_shape(input_spec.shape), input_spec.dtype)

    def call(self, inputs: list, **kwargs):
        r"""Changes graph tensor from disjoint representation.

        Args:
            inputs (Tensor): Graph labels from a disjoint representation of shape `(batch, ...)` .

        Returns:
            Tensor: Graph labels of shape `(batch, ...)` or `(batch+1, ...)` for padded disjoint.
        """
        if self.padded_disjoint:
            return _pad_left(inputs)
        return inputs


CastBatchedGraphStateToDisjoint.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__


class CastRaggedAttributesToDisjoint(_CastBatchedDisjointBase):

    def __init__(self, **kwargs):
        super(CastRaggedAttributesToDisjoint, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        out_n = input_shape[1:]
        out_gn, out_id_n, out_size_n = out_n[:1], out_n[:1], input_shape[:1]
        return out_n, out_gn, out_id_n, out_size_n

    def compute_output_spec(self, inputs_spec):
        """Compute output spec as possible."""
        output_shape = self.compute_output_shape(inputs_spec.shape)
        dtype_batch = self.dtype_batch
        output_dtypes = [inputs_spec[0].dtype, dtype_batch, dtype_batch, dtype_batch]
        output_spec = [ks.KerasTensor(s, dtype=d) for s, d in zip(output_shape, output_dtypes)]
        return output_spec

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (RaggedTensor): Attributes of shape `(batch, [None], F, ...)`

        Returns:
            list: `[attr, graph_id, item_id, item_counts]` .

                - attr (Tensor): Represents attributes or coordinates of shape `([N], F, ...)`
                - graph_id (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - item_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
                - item_counts (Tensor): Tensor of lengths for each graph of shape `(batch, )` .
        """
        return decompose_ragged_tensor(inputs, batch_dtype=self.dtype_batch)


CastRaggedAttributesToDisjoint.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__


class CastRaggedIndicesToDisjoint(_CastBatchedDisjointBase):

    def __init__(self, **kwargs):
        super(CastRaggedIndicesToDisjoint, self).__init__(**kwargs)

    def compute_output_spec(self, inputs_spec):
        """Compute output spec as possible."""
        output_shape = self.compute_output_shape([x.shape for x in inputs_spec])
        dtype_batch = self.dtype_batch
        dtype_index = inputs_spec[1].dtype if self.dtype_index is None else self.dtype_index
        output_dtypes = [inputs_spec[0].dtype, dtype_index, dtype_batch, dtype_batch, dtype_batch, dtype_batch,
                         dtype_batch, dtype_batch]
        output_spec = [ks.KerasTensor(s, dtype=d) for s, d in zip(output_shape, output_dtypes)]
        return output_spec

    def compute_output_shape(self, input_shape):
        """Compute output shape as possible."""
        in_n, in_i = input_shape

        out_n = tuple([None] + list(in_n[2:]))
        out_gn = (None, ) if out_n[0] is None else out_n[:1]
        out_id_n = (None, ) if out_n[0] is None else out_n[:1]

        out_i = tuple([None] + list(in_i[2:]))
        out_ge = (None,) if out_i[0] is None else tuple([out_i[0]])
        out_id_e = (None,) if out_i[0] is None else tuple([out_i[0]])

        if global_axis_indices == 0:
            out_i = tuple(reversed(list(out_i)))

        batch_dim_n = in_n[0]
        batch_dim_e = in_i[0]
        out_size_n, out_size_e = (batch_dim_n, ), (batch_dim_e, )

        return out_n, out_i, out_gn, out_ge, out_id_n, out_id_e, out_size_n, out_size_e

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Changes node and edge indices into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            inputs (list): List of `[nodes, edge_indices]` ,

                - nodes (Tensor): Node features are represented by a keras tensor of shape `(batch, N, F, ...)` ,
                  where N denotes the number of nodes.
                - edge_indices (Tensor): Edge index list have shape `(batch, M, 2)` with the indices of M directed
                  edges at last axis for each edge.

        Returns:
            list: `[node_attr, edge_index, graph_id_node, graph_id_edge, node_id, edge_id, nodes_count, edges_count]`

                - node_attr (Tensor): Represents node attributes or coordinates of shape `([N], F, ...)` ,
                - edge_index (Tensor): Represents the index table of shape `(2, [M])` for directed edges.
                - graph_id_node (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - graph_id_edge (Tensor): ID tensor of batch assignment in disjoint graph of shape `([M], )` .
                - nodes_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
                - edges_id (Tensor): The ID-tensor to assign each edge to its respective graph of shape `([M], )` .
                - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .
        """
        nodes, edge_indices = inputs
        nodes_flatten, graph_id_node, node_id, node_len = decompose_ragged_tensor(nodes, batch_dtype=self.dtype_batch)
        edge_indices_flatten, graph_id_edge, edge_id, edge_len = decompose_ragged_tensor(
            edge_indices, batch_dtype=self.dtype_batch)

        if self.dtype_index is not None:
            edge_indices_flatten = ops.cast(edge_indices_flatten, dtype=self.dtype_index)

        node_splits = ops.pad(ops.cumsum(node_len), [[1, 0]])
        offset_edge_indices = ops.expand_dims(ops.repeat(node_splits[:-1], edge_len), axis=-1)
        offset_edge_indices = ops.broadcast_to(offset_edge_indices, ops.shape(edge_indices_flatten))
        disjoint_indices = edge_indices_flatten + ops.cast(offset_edge_indices, edge_indices_flatten.dtype)

        # Transpose edge indices.
        if global_axis_indices == 0:
            disjoint_indices = ops.transpose(disjoint_indices)
        if self.reverse_indices:
            disjoint_indices = ops.flip(disjoint_indices, axis=global_axis_indices)

        return [nodes_flatten, disjoint_indices, graph_id_node, graph_id_edge, node_id, edge_id, node_len, edge_len]


CastRaggedIndicesToDisjoint.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__


class CastDisjointToRaggedAttributes(_CastBatchedDisjointBase):

    def __init__(self, **kwargs):
        super(CastDisjointToRaggedAttributes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Changes node or edge tensors into a Pytorch Geometric (PyG) compatible tensor format.

        Args:
            list: `[attr, graph_id, item_id, item_counts]` .

                - attr (Tensor): Represents attributes or coordinates of shape `([N], F, ...)`
                - graph_id (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - item_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
                - item_counts (Tensor): Tensor of lengths for each graph of shape `(batch, )` .

        Returns:
            Tensor: Ragged or Jagged tensor of attributes.
        """
        raise NotImplementedError()


CastDisjointToRaggedAttributes.__init__.__doc__ = _CastBatchedDisjointBase.__init__.__doc__