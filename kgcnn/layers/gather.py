from typing import Union
from keras_core.layers import Layer, Concatenate
from keras_core import ops
from kgcnn import __indices_axis__ as global_axis_indices
from kgcnn import __index_send__ as global_index_send
from kgcnn import __index_receive__ as global_index_receive


class GatherNodes(Layer):

    def __init__(self, split_indices=(0, 1),
                 concat_axis: Union[int, None] = 1,
                 axis_indices: int = global_axis_indices, **kwargs):
        super(GatherNodes, self).__init__(**kwargs)
        self.split_indices = split_indices
        self.concat_axis = concat_axis
        self.axis_indices = axis_indices
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
        # We could call build on concatenate layer.
        xs = self._compute_gathered_shape(input_shape)
        if self.concat_axis is not None:
            self._concat.build(xs)
        self.built = True

    def compute_output_shape(self, input_shape):
        xs = self._compute_gathered_shape(input_shape)
        if self.concat_axis is not None:
            xs = self._concat.compute_output_shape(xs)
        return xs

    def call(self, inputs, **kwargs):
        x, index = inputs
        gathered = []
        for i in self.split_indices:
            indices_take = ops.take(index, i, axis=self.axis_indices)
            x_i = ops.take(x, indices_take, axis=0)
            gathered.append(x_i)
        if self.concat_axis is not None:
            gathered = self._concat(gathered)
        return gathered


class GatherNodesOutgoing(Layer):

    def __init__(self, selection_index: int = global_index_send, axis_indices: int = global_axis_indices, **kwargs):
        super(GatherNodesOutgoing, self).__init__(**kwargs)
        self.selection_index = selection_index
        self.axis_indices = axis_indices

    def build(self, input_shape):
        super(GatherNodesOutgoing, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]),  list(input_shape[1])
        indices_shape.pop(self.axis_indices)
        return tuple(indices_shape + x_shape[1:])

    def call(self, inputs, **kwargs):
        x, index = inputs
        indices_take = ops.take(index, self.selection_index, axis=self.axis_indices)
        return ops.take(x, indices_take, axis=0)


class GatherNodesIngoing(Layer):

    def __init__(self, selection_index: int = global_index_receive, axis_indices: int = global_axis_indices, **kwargs):
        super(GatherNodesIngoing, self).__init__(**kwargs)
        self.selection_index = selection_index
        self.axis_indices = axis_indices

    def build(self, input_shape):
        super(GatherNodesIngoing, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]),  list(input_shape[1])
        indices_shape.pop(self.axis_indices)
        return tuple(indices_shape + x_shape[1:])

    def call(self, inputs, **kwargs):
        x, index = inputs
        indices_take = ops.take(index, self.selection_index, axis=self.axis_indices)
        return ops.take(x, indices_take, axis=0)


class GatherState(Layer):

    def __init__(self, **kwargs):
        super(GatherState, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GatherState, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        state_shape, id_shape = list(input_shape[0]),  list(input_shape[1])
        return tuple(id_shape + state_shape[1:])

    def call(self, inputs, **kwargs):
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
