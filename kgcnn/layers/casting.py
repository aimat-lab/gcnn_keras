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
    split into a flattened value :obj:`tf.Tensor` of shape `(batch*[None], F)` and a partition
    :obj:`tf.Tensor` of either 'row_length', 'row_splits' or 'value_rowids'.
    This requires the :obj:`tf.RaggedTensor` to have a ragged rank of one.

    """

    def __init__(self,
                 input_tensor_type: str = "RaggedTensor",
                 output_tensor_type: str = "RaggedTensor",
                 partition_type: str = "row_length",
                 shape: Union[list, tuple, None] = None,
                 default_value: Union[float, None, list] = None,
                 **kwargs):
        r"""Initialize layer.

        Args:
            input_tensor_type (str): Input type of the tensors for :obj:`call`. Default is "RaggedTensor".
            output_tensor_type (str): Output type of the tensors for :obj:`call`. Default is "RaggedTensor".
            partition_type (str): Partition tensor type. Default is "row_length".
            shape (list, tuple): Defining shape for conversion to tensor. Default is None.
            default_value (float, list): Default value for padding. Must broadcast. Default is None.
        """
        super(ChangeTensorType, self).__init__(**kwargs)
        self.partition_type = partition_type
        self.input_tensor_type = str(input_tensor_type)
        self.output_tensor_type = str(output_tensor_type)
        self.shape = shape
        self.default_value = default_value

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
                return inputs.to_tensor(shape=self.shape, default_value=self.default_value), inputs.with_flat_values(
                    tf.ones_like(inputs.flat_values)).to_tensor(shape=self.shape, default_value=0.0)
            elif self.output_tensor_type in self._str_type_partition:
                self.assert_ragged_input_rank(inputs, ragged_rank=1)
                return partition_from_ragged_tensor_by_name(inputs, self.partition_type)

        # Unsupported type conversion.
        raise NotImplementedError(
            "Unsupported conversion from %s to %s" % (self.input_tensor_type, self.output_tensor_type))

    def get_config(self):
        """Update layer config."""
        config = super(ChangeTensorType, self).get_config()
        config.update({"partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "output_tensor_type": self.output_tensor_type,
                       "shape": self.shape, "default_value": self.default_value
                       })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='CastEdgeIndicesToDenseAdjacency')
class CastEdgeIndicesToDenseAdjacency(GraphBaseLayer):
    r"""Layer to change the ragged tensor representation of edges of graphs into (dense) tensor type information of the
    adjacency matrix.

    In addition to the (feature) adjacency matrix, a mask can be returned. Note that the adjacency matrix is padded
    and has an extra dimension to include edge feature information.

    """

    def __init__(self, n_max: int = None, return_mask: bool = True, **kwargs):
        r"""Initialize layer.

        Args:
            n_max (int): Defining maximum shape for padded adjacency matrix. Default is None.
            default_value (float, list): Default value for padding. Must broadcast. Default is None.
        """
        super(CastEdgeIndicesToDenseAdjacency, self).__init__(**kwargs)
        self.n_max = int(n_max) if n_max else None
        self.return_mask = bool(return_mask)

    def build(self, input_shape):
        """Build layer."""
        super(CastEdgeIndicesToDenseAdjacency, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [edges, indices]

                - edges (tf.RaggedTensor): Edge features of shape `(batch, [N], F)`
                - indices (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [N], 2)`.

        Returns:
            tuple: Padded (batch) adjacency matrix of shape `(batch, N_max, N_max, F)` plus mask of shape
                `(batch, N_max, N_max)`.
        """
        edges, indices = self.assert_ragged_input_rank(inputs, ragged_rank=1)

        indices_flatten = indices.values
        edges_flatten = edges.values
        indices_batch = tf.expand_dims(tf.cast(indices.value_rowids(), indices.values.dtype), axis=-1)
        feature_shape_edges = edges.shape[2:]

        if self.n_max:
            n_max = self.n_max
            indices_okay = tf.math.reduce_all(indices.values < self.n_max, axis=-1)
            indices_flatten = indices_flatten[indices_okay]
            indices_batch = indices_batch[indices_okay]
            edges_flatten = edges_flatten[indices_okay]
        else:
            n_max = tf.math.reduce_max(indices.values) + 1

        scatter_indices = tf.concat([indices_batch, indices_flatten], axis=-1)

        # Determine shape of output adjacency matrix.
        shape_adj = tf.TensorShape([tf.shape(indices)[0], n_max, n_max] + feature_shape_edges)
        adj = tf.scatter_nd(scatter_indices, edges_flatten, shape=shape_adj)

        if self.return_mask:
            mask_values = tf.ones(tf.shape(edges_flatten)[0], dtype=edges.dtype)
            shape_mask = tf.TensorShape([tf.shape(indices)[0], n_max, n_max])
            mask = tf.scatter_nd(scatter_indices, mask_values, shape=shape_mask)
        else:
            mask = None

        return adj, mask

    def get_config(self):
        """Update layer config."""
        config = super(CastEdgeIndicesToDenseAdjacency, self).get_config()
        config.update({"n_max": self.n_max, "return_mask": self.return_mask})
        return config
