import numpy as np
import tensorflow as tf
from typing import Union
from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.ragged import partition_from_ragged_tensor_by_name
from kgcnn.layers.base import GraphBaseLayer

# import tensorflow.keras as ks
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='ChangeTensorType')
class ChangeTensorType(GraphBaseLayer):
    r"""Layer to change the ragged tensor representation into tensor type information.

    The tensor representation of :obj:`tf.RaggedTensor` is cast into similar :obj:`tf.Tensor` formats.
    For example, the :obj:`RaggedTensor` has shape `(batch, None, F)`.
    The dense tensor in case of equal sized graphs or zero padded graphs will have shape `(batch, N, F)`, with `N`
    being the (maximum) number of nodes, or given shape otherwise.

    For disjoint representation (one big graph with disconnected sub-graphs) the :obj:`tf.RaggedTensor` can be
    split into a flattened value :obj:`tf.Tensor` of shape `(batch*[None], F)` and a partition :obj:`tf.Tensor` of
    either 'row_length', 'row_splits' or 'value_rowids'.
    This requires the :obj:`tf.RaggedTensor` to have a ragged rank of one.

    """

    def __init__(self,
                 input_tensor_type: str = "RaggedTensor",
                 output_tensor_type: str = "RaggedTensor",
                 partition_type: str = "row_length",
                 shape: Union[list, tuple, None] = None,
                 default_value: Union[float, None, list] = None,
                 boolean_mask: bool = False,
                 **kwargs):
        r"""Initialize layer.

        Args:
            input_tensor_type (str): Input type of the tensors for :obj:`call`. Default is "RaggedTensor".
            output_tensor_type (str): Output type of the tensors for :obj:`call`. Default is "RaggedTensor".
            partition_type (str): Partition tensor type. Default is "row_length".
            shape (list, tuple): Defining shape for conversion to tensor. Default is None.
            default_value (float, list): Default value for padding. Must broadcast. Default is None.
            boolean_mask (bool): Whether mask for padded tensor should be boolean or the same type as tensor.

        """
        super(ChangeTensorType, self).__init__(**kwargs)
        self.partition_type = partition_type
        self.input_tensor_type = str(input_tensor_type)
        self.output_tensor_type = str(output_tensor_type)
        self.shape = shape
        self.default_value = default_value
        self.boolean_mask = boolean_mask

        self._str_type_ragged = ["ragged", "RaggedTensor"]
        self._str_type_tensor = ["Tensor", "tensor"]
        self._str_type_mask = ["padded", "masked", "mask"]
        self._str_type_partition = ["disjoint", "row_partition", "values_partition", "values"]

    def build(self, input_shape):
        """Build layer."""
        super(ChangeTensorType, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (tf.RaggedTensor): Ragged tensor with `ragged_rank` of 1.

        Returns:
            tensor: Changed tensor type.
        """
        if self.input_tensor_type in self._str_type_ragged:

            if self.output_tensor_type in self._str_type_ragged:
                return inputs  # Nothing to do here.
            if self.output_tensor_type in self._str_type_tensor:
                return inputs.to_tensor(shape=self.shape, default_value=self.default_value)
            elif self.output_tensor_type in self._str_type_mask:
                padded = inputs.to_tensor(shape=self.shape, default_value=self.default_value)
                mask_values = tf.ones_like(inputs.flat_values) if not self.boolean_mask else tf.ones(
                    tf.shape(inputs.flat_values), dtype="bool")
                mask = inputs.with_flat_values(mask_values).to_tensor(shape=self.shape)
                return padded, mask
            elif self.output_tensor_type in self._str_type_partition:
                inputs = self.assert_ragged_input_rank(inputs, ragged_rank=1)
                return partition_from_ragged_tensor_by_name(inputs, self.partition_type)

        # Unsupported type conversion.
        raise NotImplementedError(
            "Unsupported conversion from '%s' to '%s'." % (self.input_tensor_type, self.output_tensor_type))

    def get_config(self):
        """Update layer config."""
        config = super(ChangeTensorType, self).get_config()
        config.update({
            "partition_type": self.partition_type,
            "input_tensor_type": self.input_tensor_type,
            "output_tensor_type": self.output_tensor_type,
            "shape": self.shape, "default_value": self.default_value, "boolean_mask": self.boolean_mask
        })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='CastEdgeIndicesToDenseAdjacency')
class CastEdgeIndicesToDenseAdjacency(GraphBaseLayer):
    r"""Layer to change the ragged tensor representation of edges of graphs into (dense) tensor type information of the
    adjacency matrix.

    In addition to the (featured) adjacency matrix, a mask can be returned. Note that the adjacency matrix is padded
    and has possible extra dimensions to include edge feature information.

    """

    def __init__(self, n_max: int = None, return_mask: bool = True, use_node_tensor: bool = True, **kwargs):
        r"""Initialize layer.

        Args:
            n_max (int): Defining maximum shape for padded adjacency matrix. Default is None.
            default_value (float, list): Default value for padding. Must broadcast. Default is None.
        """
        super(CastEdgeIndicesToDenseAdjacency, self).__init__(**kwargs)
        self.n_max = int(n_max) if n_max else None
        self.return_mask = bool(return_mask)
        self.use_node_tensor = use_node_tensor

    def build(self, input_shape):
        """Build layer."""
        super(CastEdgeIndicesToDenseAdjacency, self).build(input_shape)

    # @tf.function
    def call(self, inputs, **kwargs):
        r"""Forward pass. The additional node information is optional but recommended for auto shape.

        Args:
            inputs (list): [nodes, edges, indices]

                - nodes (tf.RaggedTensor, tf.Tensor): Edge features of shape `(batch, [N], ...)`
                - edges (tf.RaggedTensor, tf.Tensor): Edge features of shape `(batch, [N], F)`
                - indices (tf.RaggedTensor, tf.Tensor): Edge indices referring to nodes of shape `(batch, [N], 2)`.

        Returns:
            tuple: Padded (batch) adjacency matrix of shape `(batch, N_max, N_max, F)` plus mask of shape
                `(batch, N_max, N_max)`.
        """
        if self.use_node_tensor:
            nodes, edges, indices = self.assert_ragged_input_rank(inputs, ragged_rank=1)
        else:
            edges, indices = self.assert_ragged_input_rank(inputs, ragged_rank=1)
            nodes = None

        indices_flatten = indices.values
        edges_flatten = edges.values
        indices_batch = tf.expand_dims(tf.cast(indices.value_rowids(), indices.values.dtype), axis=-1)
        feature_shape_edges_static = edges.shape[2:]
        feature_shape_edges = tf.cast(feature_shape_edges_static, dtype=indices.values.dtype)

        if self.n_max:
            n_max = self.n_max
            indices_okay = tf.math.reduce_all(indices.values < self.n_max, axis=-1)
            indices_flatten = indices_flatten[indices_okay]
            indices_batch = indices_batch[indices_okay]
            edges_flatten = edges_flatten[indices_okay]
            n_max_shape = tf.constant([n_max, n_max], dtype=indices.values.dtype)
        else:
            if self.use_node_tensor:
                n_max = tf.math.reduce_max(nodes.row_lengths())
            else:
                n_max = tf.math.reduce_max(indices.values) + 1
            n_max_shape = tf.cast(tf.repeat(n_max, 2), dtype=indices.values.dtype)  # Shape of adjacency matrix

        scatter_indices = tf.concat([indices_batch, indices_flatten], axis=-1)

        # Determine shape of output adjacency matrix.
        batch_shape = tf.cast(tf.expand_dims(tf.shape(indices)[0], axis=-1), dtype=scatter_indices.dtype)
        if len(feature_shape_edges_static) > 0:
            shape_adj = tf.concat([batch_shape, n_max_shape, feature_shape_edges], axis=0)
        else:
            shape_adj = tf.concat([batch_shape, n_max_shape], axis=0)

        adj = tf.scatter_nd(scatter_indices, edges_flatten, shape=shape_adj)

        if self.return_mask:
            mask_values = tf.ones(tf.shape(edges_flatten)[0], dtype=edges.dtype)
            shape_mask = tf.concat([batch_shape, n_max_shape], axis=0)
            mask = tf.scatter_nd(scatter_indices, mask_values, shape=shape_mask)
        else:
            mask = None

        return adj, mask

    def get_config(self):
        """Update layer config."""
        config = super(CastEdgeIndicesToDenseAdjacency, self).get_config()
        config.update({"n_max": self.n_max, "return_mask": self.return_mask, "use_node_tensor": self.use_node_tensor})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='CastEdgeIndicesToDisjointSparseAdjacency')
class CastEdgeIndicesToDisjointSparseAdjacency(GraphBaseLayer):
    r"""Helper layer to cast a set of RaggedTensors forming a graph representation into a single SparseTensor, which
    then can be regarded to be in disjoint representation. This means that the batch is represented as one big
    adjacency matrix with disjoint sub-blocks.

    This includes edge indices and adjacency matrix entries. The Sparse tensor is simply the adjacency matrix.
    """

    def __init__(self, is_sorted: bool = False, **kwargs):
        """Initialize layer.

        Args:
            is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        """
        super(CastEdgeIndicesToDisjointSparseAdjacency, self).__init__(**kwargs)
        self.node_indexing = "sample"
        self.is_sorted = is_sorted

    def build(self, input_shape):
        """Build layer."""
        super(CastEdgeIndicesToDisjointSparseAdjacency, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node feature tensor of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge feature ragged tensor of shape (batch, [M], 1)
                - edge_index (tf.RaggedTensor): Ragged edge_indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.SparseTensor: Sparse disjoint matrix of shape (batch*[N],batch*[N])
        """
        inputs = self.assert_ragged_input_rank(inputs)
        nod, node_len = inputs[0].values, inputs[0].row_lengths()
        edge, _ = inputs[1].values, inputs[1].row_lengths()
        edge_index, edge_len = inputs[2].values, inputs[2].row_lengths()

        # batch-wise indexing
        edge_index = partition_row_indexing(
            edge_index,
            node_len, edge_len,
            partition_type_target="row_length",
            partition_type_index="row_length",
            from_indexing=self.node_indexing,
            to_indexing="batch"
        )
        indexlist = edge_index
        valuelist = edge

        if not self.is_sorted:
            # Sort per outgoing
            batch_order = tf.argsort(indexlist[:, 1], axis=0, direction='ASCENDING')
            indexlist = tf.gather(indexlist, batch_order, axis=0)
            valuelist = tf.gather(valuelist, batch_order, axis=0)
            # Sort per ingoing node
            node_order = tf.argsort(indexlist[:, 0], axis=0, direction='ASCENDING', stable=True)
            indexlist = tf.gather(indexlist, node_order, axis=0)
            valuelist = tf.gather(valuelist, node_order, axis=0)

        indexlist = tf.cast(indexlist, dtype=tf.int64)
        dense_shape = tf.concat([tf.shape(nod)[0:1], tf.shape(nod)[0:1]], axis=0)
        dense_shape = tf.cast(dense_shape, dtype=tf.int64)
        out = tf.sparse.SparseTensor(indexlist, valuelist[:, 0], dense_shape)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastEdgeIndicesToDisjointSparseAdjacency, self).get_config()
        config.update({"is_sorted": self.is_sorted})
        return config
