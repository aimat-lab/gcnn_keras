import keras as ks
from typing import Union
from keras.layers import Layer, Concatenate
from keras import ops
from kgcnn import __indices_axis__ as global_axis_indices
from kgcnn import __index_send__ as global_index_send
from kgcnn import __index_receive__ as global_index_receive


class GatherNodes(Layer):
    r"""Gather node or edge embedding from an index list.

    The embeddings are gather from an index tensor. An edge is defined by index tuple :math:`(i ,j)` .
    In the default definition, index :math:`i` is expected to be the receiving or target node.
    Effectively, the layer simply does:

    .. code-block:: python

        ops.take(nodes, index[x], axis=0) for x in split_indices

    Additionally, the gathered embeddings can be concatenated along the index dimension,
    by setting :obj:`concat_axis` if index shape is known during build.

    .. note:

        Default of this layer is concatenation with :obj:`concat_axis=1` .

    Example of usage for :obj:`GatherNodes` :

    .. code-block:: python

        from keras import ops
        from kgcnn.layers.gather import GatherNodes
        nodes = ops.convert_to_tensor([[0.0],[1.0],[2.0],[3.0],[4.0]], dtype="float32")
        edge_idx = ops.convert_to_tensor([[0,0,1,2], [1,2,0,1]], dtype="int32")
        print(GatherNodes()([nodes, edge_idx]))

    """

    def __init__(self, split_indices=(0, 1),
                 concat_axis: Union[int, None] = 1,
                 axis_indices: int = global_axis_indices, **kwargs):
        """Initialize layer.

        Args:
            split_indices (list): List of indices to split and take values for. Default is (0, 1).
            concat_axis (int): The axis which concatenates embeddings. Default is 1.
            axis_indices (int): Axis on which to split indices from. Default is 0.
        """
        super(GatherNodes, self).__init__(**kwargs)
        self.split_indices = split_indices
        self.concat_axis = concat_axis
        self.axis_indices = axis_indices
        if self.concat_axis is not None:
            self._concat = Concatenate(axis=concat_axis)

    def _compute_gathered_shape(self, input_shape):
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]), list(input_shape[1])
        xs = []
        indices_shape.pop(self.axis_indices)
        for _ in self.split_indices:
            xs.append(indices_shape + x_shape[1:])
        return xs

    def build(self, input_shape):
        """Build layer."""
        # We could call build on concatenate layer.
        xs = self._compute_gathered_shape(input_shape)
        if self.concat_axis is not None:
            self._concat.build(xs)
        self.built = True

    def compute_output_shape(self, input_shape):
        """Compute output shape of this layer."""
        xs = self._compute_gathered_shape(input_shape)
        if self.concat_axis is not None:
            xs = self._concat.compute_output_shape(xs)
        return xs

    def compute_output_spec(self, inputs_spec):
        output_shape = self.compute_output_shape([x.shape for x in inputs_spec])
        if self.concat_axis is not None:
            return ks.KerasTensor(output_shape, dtype=inputs_spec[0].dtype)
        return [ks.KerasTensor(s, dtype=inputs_spec[0].dtype) for s in output_shape]

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [nodes, index]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - index (Tensor): Edge indices referring to nodes of shape `(2, [M])`

        Returns:
            Tensor: Gathered node embeddings that match the number of edges of shape `([M], 2*F)` or list of single
                node embeddings of shape [`([M], F)` , `([M], F)` , ...].
        """
        x, index = inputs
        gathered = []
        for i in self.split_indices:
            indices_take = ops.take(index, i, axis=self.axis_indices)
            x_i = ops.take(x, indices_take, axis=0)
            gathered.append(x_i)
        if self.concat_axis is not None:
            gathered = self._concat(gathered)
        return gathered

    def get_config(self):
        """Get config for this layer."""
        conf = super(GatherNodes, self).get_config()
        conf.update({"split_indices": self.split_indices, "concat_axis": self.concat_axis,
                     "axis_indices": self.axis_indices})
        return conf


class GatherNodesOutgoing(Layer):
    r"""Gather sending or outgoing nodes of edges with index :math:`j` .

    An edge is defined by index tuple :math:`(i, j)`.
    In the default definition, index :math:`j` is expected to be the sending or source node.
    """

    def __init__(self, selection_index: int = global_index_send, axis_indices: int = global_axis_indices, **kwargs):
        """Initialize layer.

        Args:
            selection_index (int): Index of sending nodes. Default is 1.
            axis_indices (int): Axis of node indices in index Tensor. Default is 0.
        """
        super(GatherNodesOutgoing, self).__init__(**kwargs)
        self.selection_index = selection_index
        self.axis_indices = axis_indices

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesOutgoing, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute output shape of this layer."""
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]),  list(input_shape[1])
        indices_shape.pop(self.axis_indices)
        return tuple(indices_shape + x_shape[1:])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [nodes, index]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - index (Tensor): Edge indices referring to nodes of shape `(2, [M])`

        Returns:
            Tensor: Gathered node embeddings that match the number of edges of shape `([M], F)` .
        """
        x, index = inputs
        indices_take = ops.take(index, self.selection_index, axis=self.axis_indices)
        return ops.take(x, indices_take, axis=0)

    def get_config(self):
        """Get config for this layer."""
        conf = super(GatherNodesOutgoing, self).get_config()
        conf.update({"selection_index": self.selection_index, "axis_indices": self.axis_indices})
        return conf


class GatherNodesIngoing(Layer):
    r"""Gather receiving or ingoing nodes of edges with index :math:`i` .

    An edge is defined by index tuple :math:`(i, j)`.
    In the default definition, index :math:`i` is expected to be the receiving or target node.
    """

    def __init__(self, selection_index: int = global_index_receive, axis_indices: int = global_axis_indices, **kwargs):
        """Initialize layer.

        Args:
            selection_index (int): Index of receiving nodes. Default is 0.
            axis_indices (int): Axis of node indices in index Tensor. Default is 0.
        """
        super(GatherNodesIngoing, self).__init__(**kwargs)
        self.selection_index = selection_index
        self.axis_indices = axis_indices

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesIngoing, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute output shape of this layer."""
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]),  list(input_shape[1])
        indices_shape.pop(self.axis_indices)
        return tuple(indices_shape + x_shape[1:])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [nodes, index]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - index (Tensor): Edge indices referring to nodes of shape `(2, [M])`

        Returns:
            Tensor: Gathered node embeddings that match the number of edges of shape `([M], F)` .
        """
        x, index = inputs
        indices_take = ops.take(index, self.selection_index, axis=self.axis_indices)
        return ops.take(x, indices_take, axis=0)

    def get_config(self):
        """Get config for this layer."""
        conf = super(GatherNodesIngoing, self).get_config()
        conf.update({"selection_index": self.selection_index, "axis_indices": self.axis_indices})
        return conf


class GatherState(Layer):
    r"""Layer to repeat environment or global state for a specific embeddings tensor like node or edge lists.

    To repeat the correct global state (like an environment feature vector) for each sub graph,
    a tensor with the target shape and batch ID is required.

    Mostly used to concatenate a global state :math:`\mathbf{s}` with node embeddings :math:`\mathbf{h}_i`
    like for example:

    .. math::

        \mathbf{h}_i = \mathbf{h}_i \oplus \mathbf{s}

    where this layer only repeats :math:`\mathbf{s}` to match an embedding tensor :math:`\mathbf{h}_i`.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute output shape of this layer."""
        assert len(input_shape) == 2
        state_shape, id_shape = list(input_shape[0]),  list(input_shape[1])
        return tuple(id_shape + state_shape[1:])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [state, batch_id]

                - state (Tensor): Graph specific embedding tensor. This is tensor of shape `(batch, F)`
                - batch_id (Tensor): Tensor of batch IDs for each sub-graph of shape `([N], )` .

        Returns:
            Tensor: Graph embedding with repeated single state for each sub-graph of shape `([N], F)`.
        """
        env, batch_id = inputs
        out = ops.take(env, batch_id, axis=0)
        return out


class GatherEdgesPairs(Layer):
    """Gather edge pairs that also works for invalid indices given a certain pair, i.e. if an edge does not have its
    reverse counterpart in the edge indices list.

    This class is used in e.g. `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`__ .
    """

    def __init__(self, axis_indices: int = global_axis_indices, **kwargs):
        """Initialize layer.

        Args:
            axis_indices (int): Axis of indices. Default is 0.
        """
        super(GatherEdgesPairs, self).__init__(**kwargs)
        self.axis_indices = axis_indices

    def build(self, input_shape):
        """Build this layer."""
        self.built = True

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [edges, pair_index]

                - edges (Tensor): Edge embeddings of shape ([M], F)
                - pair_index (Tensor): Edge indices referring to edges of shape (1, [M])

        Returns:
            Tensor: Gathered edge embeddings that match the reverse edges of shape ([M], F) for index.
        """
        edges, pair_index = inputs
        indices_take = ops.take(pair_index, 0, self.axis_indices)
        index_corrected = ops.where(indices_take >= 0, indices_take, ops.zeros_like(indices_take))
        edges_paired = ops.take(edges, index_corrected, axis=0)
        edges_corrected = ops.where(
            ops.expand_dims(indices_take, axis=-1) >= 0, edges_paired, ops.zeros_like(edges_paired))
        return edges_corrected

    def get_config(self):
        """Get layer config."""
        conf = super(GatherEdgesPairs, self).get_config()
        conf.update({"axis_indices": self.axis_indices})
        return conf