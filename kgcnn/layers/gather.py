import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing
from typing import Union
# from kgcnn.ops.axis import get_positive_axis
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='GatherEmbedding')
class GatherEmbedding(GraphBaseLayer):
    r"""Gather node or edge embedding from an index list.

    The embeddings are gather from a ragged index tensor. An edge is defined by index tuple :math:`(i ,j)`.
    In the default definition, index :math:`i` is expected to be the receiving or target node.
    Effectively, the layer simply does:

    .. code-block:: python

        tf.gather(embedding, indices, batch_dims=1, axis=1)

    Additionally, the gathered embeddings can be split or concatenated along the index dimension after gather,
    by setting :obj:`split_axis` or :obj:`concat_axis` if index shape is known during build.

    .. note:

        Default of this layer is concatenation with :obj:`concat_axis=2`.

    Example of usage for :obj:`GatherEmbedding`:

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.gather import GatherEmbedding
        nodes = tf.ragged.constant([[[0.0],[1.0]],[[2.0],[3.0],[4.0]]], ragged_rank=1)
        edge_idx = tf.ragged.constant([[[0,1],[1,0]],[[0,2],[1,2]]], ragged_rank=1)
        print(GatherEmbedding()([nodes, edge_idx]))

    """

    def __init__(self,
                 axis: int = 1,
                 split_axis: Union[int, None] = 2,
                 split_indices: list = None,
                 concat_axis: Union[int, None] = 2,
                 allow_disjoint_implementation: bool = True,
                 **kwargs):
        r"""Initialize layer.

        Args:
            axis (int): The axis to gather embeddings from. Default is 1.
            split_axis (int): The axis to split indices to gather embeddings. Default is None.
            split_indices (list): List of indices to split from gathered tensor. Default is None.
            concat_axis (int): The axis which concatenates embeddings. Default is 2.
            allow_disjoint_implementation (bool): Whether to allow (preferred) disjoint implementation.
        """
        super(GatherEmbedding, self).__init__(**kwargs)
        self.concat_axis = concat_axis
        self.axis = axis
        self.split_axis = split_axis
        self.split_indices = split_indices
        self.node_indexing = "sample"
        self.allow_disjoint_implementation = allow_disjoint_implementation

        if self.concat_axis is not None and self.split_axis is None:
            raise ValueError("Can only concat `list` of gathered tensors. Require `split_axis` not None.")

    def build(self, input_shape):
        super(GatherEmbedding, self).build(input_shape)

        if len(input_shape) != 2:
            print("Number of inputs for layer '%s' must be 2: `[nodes, indices]` ." % self.name)

    def _is_disjoint_possible(self, inputs, **kwargs):
        # The primary case for aggregation of nodes from node feature list. Case from doc-string.
        # Possibly faster implementation via values and indices shifted by row-partition.
        # Equal to disjoint implementation. Only works for ragged_rank=1 and specific axis.
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
            is_rank_one = all([x.ragged_rank == 1 for x in inputs])
            if is_rank_one and self.axis == 1 and self.concat_axis in [None, 2] and self.split_axis in [None, 2]:
                return True
        return False

    def _disjoint_implementation(self, inputs, **kwargs):
        node, node_part = inputs[0].values, inputs[0].row_splits
        edge_index, edge_part = inputs[1].values, inputs[1].row_lengths()
        disjoint_list = partition_row_indexing(
            edge_index, node_part, edge_part, partition_type_target="row_splits", partition_type_index="row_length",
            to_indexing='batch', from_indexing=self.node_indexing
        )
        if self.split_axis == 2:
            if self.split_indices is None and edge_index.shape[1] is None:
                raise ValueError("Cannot infer split indices, please specify statically in `split_indices` .")
            split_indices = self.split_indices if self.split_indices else range(edge_index.shape[1])
            indices_list = [tf.gather(disjoint_list, i, axis=1) for i in split_indices]
            out = [tf.gather(node, ix, axis=0) for ix in indices_list]
            if self.concat_axis == 2:
                out = tf.concat(out, axis=1)
                return tf.RaggedTensor.from_row_lengths(out, edge_part, validate=self.ragged_validate)
            return [tf.RaggedTensor.from_row_lengths(x, edge_part, validate=self.ragged_validate) for x in out]
        else:
            out = tf.gather(node, disjoint_list, axis=0)
            return tf.RaggedTensor.from_row_lengths(out, edge_part, validate=self.ragged_validate)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [embeddings, tensor_index]

                - embeddings (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [M], 2)`

        Returns:
            tf.RaggedTensor: Gathered node embeddings that match the number of edges of shape `(batch, [M], 2*F)`
        """
        # Old disjoint implementation that could be faster.
        if self._is_disjoint_possible(inputs, **kwargs) and self.allow_disjoint_implementation:
            return self._disjoint_implementation(inputs, **kwargs)

        # For arbitrary gather from ragged tensor use tf.gather with batch_dims=1.
        # Works in tf.__version__ >= 2.4 !
        # Option: Split features.
        nodes, indices = inputs
        if self.split_axis is not None:
            if self.split_indices is None and indices.shape[self.split_axis] is None:
                raise ValueError("Cannot infer split indices, please specify statically in `split_indices` .")
            split_indices = self.split_indices if self.split_indices else range(indices.shape[self.split_axis])
            indices_list = [tf.gather(nodes, i, axis=self.split_axis) for i in split_indices]
            out = [tf.gather(nodes, ix, batch_dims=1, axis=self.axis) for ix in indices_list]
            # Option: Concat features.
            if self.concat_axis is not None:
                out = tf.concat(out, axis=self.concat_axis)
        else:
            out = tf.gather(nodes, indices, batch_dims=1, axis=self.axis)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(GatherEmbedding, self).get_config()
        config.update({
            "concat_axis": self.concat_axis, "axis": self.axis, "split_axis": self.split_axis,
            "split_indices": self.split_indices, "allow_disjoint_implementation": self.allow_disjoint_implementation
        })
        return config


GatherNodes = GatherEmbedding


@ks.utils.register_keras_serializable(package='kgcnn', name='GatherEmbeddingSelection')
class GatherEmbeddingSelection(GatherEmbedding):
    r"""Gather node or edge embedding for a defined index in the index list.

    The embeddings are gather from a ragged index tensor for a list of specific indices which are given by
    :obj:`selection_index`. This can be used for ingoing or outgoing nodes or angles.
    Returns a list of embeddings for each :obj:`selection_index`. An edge is defined by index tuple :math:`(i, j)`.
    In the default definition, index :math:`i` is expected to be the receiving or target node.
    Effectively, the layer simply does:

    .. code-block:: python

        tf.gather(embedding, indices[:, :, selection_index], batch_dims=1, axis=1)

    Additionally, the axis for gather can be specified for target and index tensor via :obj:`axis` and
    :obj:`axis_indices`.
    This layer always returns a list of embeddings even if :obj:`selection_index` is of type :obj:`int`.

    Example of usage for :obj`GatherEmbeddingSelection`:

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.gather import GatherEmbeddingSelection
        nodes = tf.ragged.constant([[[0.0],[1.0]],[[2.0],[3.0],[4.0]]], ragged_rank=1)
        edge_idx = tf.ragged.constant([[[0,1],[1,0]],[[0,2],[1,2]]], ragged_rank=1)
        print(GatherEmbeddingSelection([0, 1])([nodes, edge_idx]))

    """

    def __init__(self, selection_index, axis: int = 1, axis_indices: int = 2, **kwargs):
        r"""Initialize layer.

        Args:
            selection_index (list, int): Which indices to gather embeddings for.
            axis (int): Axis to gather embeddings from. Default is 1.
            axis_indices (int): From which axis to take the indices for gather. Default is 2.
        """
        if not isinstance(selection_index, (list, tuple, int)):
            raise ValueError("Indices for selection must be list or tuple for layer `GatherEmbeddingSelection`.")

        if isinstance(selection_index, int):
            selection_index = [selection_index]
        else:
            selection_index = list(selection_index)
        self.selection_index = selection_index
        self.axis_indices = axis_indices
        # Different names as in parent class.
        super(GatherEmbeddingSelection, self).__init__(
            axis=axis, concat_axis=None, split_axis=axis_indices, split_indices=self.selection_index, **kwargs)

    def get_config(self):
        config = super(GatherEmbeddingSelection, self).get_config()
        # Different names as in parent class.
        for x in ["split_indices", "split_axis", "concat_axis"]:
            config.pop(x)
        config.update({"axis_indices": self.axis_indices, "selection_index": self.selection_index})
        return config


GatherNodesSelection = GatherEmbeddingSelection


@ks.utils.register_keras_serializable(package='kgcnn', name='GatherNodesIngoing')
class GatherNodesIngoing(GatherEmbeddingSelection):
    r"""Gather receiving or ingoing nodes of edges with index :math:`i`.

    An edge is defined by index tuple :math:`(i, j)`.
    In the default definition, index :math:`i` is expected to be the receiving or target node.
    The layer inherits from :obj:`GatherEmbeddingSelection` and effectively does:

    .. code-block:: python

        GatherEmbeddingSelection(selection_index=0)(inputs)[0]

    """

    def __init__(self, selection_index: int = 0, **kwargs):
        r"""Initialize layer.

        Args:
            selection_index (list, int): Which index to gather embeddings for. Default is 0.
        """
        super(GatherNodesIngoing, self).__init__(selection_index=selection_index, **kwargs)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [nodes, tensor_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [M], 2)`

        Returns:
            tf.RaggedTensor: Gathered node embeddings for ingoing nodes of edges of shape `(batch, [M], F)`
        """
        return super(GatherNodesIngoing, self).call(inputs, **kwargs)[0]


@ks.utils.register_keras_serializable(package='kgcnn', name='GatherNodesOutgoing')
class GatherNodesOutgoing(GatherEmbeddingSelection):
    r"""Gather sending or outgoing nodes of edges with index :math:`j`.

    An edge is defined by index tuple :math:`(i, j)`.
    In the default definition, index :math:`j` is expected to be the sending or source node.
    The layer inherits from :obj:`GatherEmbeddingSelection` and effectively does:

    .. code-block:: python

        GatherEmbeddingSelection(selection_index=1)(inputs)[0]

    """

    def __init__(self, selection_index: int = 1, **kwargs):
        r"""Initialize layer.

        Args:
            selection_index (list, int): Which index to gather embeddings for. Default is 1.
        """
        super(GatherNodesOutgoing, self).__init__(selection_index=selection_index, **kwargs)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [nodes, tensor_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [M], 2)`

        Returns:
            tf.RaggedTensor: Gathered node embeddings for outgoing nodes of edges of shape `(batch, [M], F)`
        """
        return super(GatherNodesOutgoing, self).call(inputs, **kwargs)[0]


@ks.utils.register_keras_serializable(package='kgcnn', name='GatherState')
class GatherState(GraphBaseLayer):
    r"""Layer to repeat environment or global state for a specific embeddings (ragged) tensor like node or edge lists.
    
    To repeat the correct global state (like an environment feature vector) for each sample,
    a tensor with the target shape or length, partition in case of ragged target tensors is required.

    Mostly used to concatenate a global state :math:`\mathbf{s}` with node embeddings :math:`\mathbf{h}_i`
    like for example:

    .. math::

        \mathbf{h}_i = \mathbf{h}_i \oplus \mathbf{s}

    where this layer only repeats :math:`\mathbf{s}` to match a ragged embedding tensor :math:`\mathbf{h}_i`.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [state, target]

                - state (tf.Tensor): Graph specific embedding tensor. This is tensor of shape `(batch, F)`
                - target (tf.RaggedTensor): Target to collect state for [N] of shape `(batch, [N], F)`

        Returns:
            tf.RaggedTensor: Graph embedding with repeated single state for each graph of shape `(batch, [N], F)`.
        """
        env = inputs[0]
        dyn_inputs = inputs[1]

        if isinstance(dyn_inputs, tf.RaggedTensor):
            target_len = dyn_inputs.row_lengths()
        else:
            target_len = tf.repeat(tf.shape(dyn_inputs)[1], tf.shape(dyn_inputs)[0])

        out = tf.repeat(env, target_len, axis=0)
        out = tf.RaggedTensor.from_row_lengths(out, target_len, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(GatherState, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GatherEdgesPairs')
class GatherEdgesPairs(GraphBaseLayer):
    """Gather edge pairs that also works for invalid indices given a certain pair, i.e. if an edge does not have its
    reverse counterpart in the edge indices list.

    This class is used in `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`__ .

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherEdgesPairs, self).__init__(**kwargs)
        self.gather_layer = GatherNodesIngoing()

    def build(self, input_shape):
        """Build layer."""
        super(GatherEdgesPairs, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [edges, pair_index]

                - edges (tf.RaggedTensor): Node embeddings of shape (batch, [M], F)
                - pair_index (tf.RaggedTensor): Edge indices referring to edges of shape (batch, [M], 1)

        Returns:
            list: Gathered edge embeddings that match the reverse edges of shape (batch, [M], F) for selection_index.
        """
        self.assert_ragged_input_rank(inputs)
        edges, pair_index = inputs
        index_corrected = tf.RaggedTensor.from_row_splits(
            tf.where(pair_index.values >= 0, pair_index.values, tf.zeros_like(pair_index.values)),
            pair_index.row_splits, validate=self.ragged_validate)
        edges_paired = self.gather_layer([edges, index_corrected], **kwargs)
        edges_corrected = tf.RaggedTensor.from_row_splits(
            tf.where(pair_index.values >= 0, edges_paired.values, tf.zeros_like(edges_paired.values)),
            edges_paired.row_splits, validate=self.ragged_validate)
        return edges_corrected
