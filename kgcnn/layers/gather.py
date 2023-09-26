from typing import Union
from keras_core.layers import Layer, Concatenate
from keras_core import ops
from kgcnn import __indices_first__ as global_indices_first


class GatherNodes(Layer):

    def __init__(self, split_indices=(0, 1), concat_axis: Union[int, None] = 1, **kwargs):
        super(GatherNodes, self).__init__(**kwargs)
        self.split_indices = split_indices
        self.concat_axis = concat_axis
        self._concat = Concatenate(axis=concat_axis)

    def _compute_gathered_shape(self, input_shape):
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]), list(input_shape[1])
        xs = []
        indices_length = indices_shape[1:] if global_indices_first else indices_shape[:1]
        for _ in self.split_indices:
            xs.append(indices_length + x_shape[1:])
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
            indices_take = index[i] if global_indices_first else index[:, i]
            x_i = ops.take(x, indices_take, axis=0)
            gathered.append(x_i)
        if self.concat_axis is not None:
            gathered = self._concat(gathered)
        return gathered


class GatherNodesOutgoing(Layer):

    def __init__(self, selection_index: int = 0, **kwargs):
        super(GatherNodesOutgoing, self).__init__(**kwargs)
        self.selection_index = selection_index

    def build(self, input_shape):
        super(GatherNodesOutgoing, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]),  list(input_shape[1])
        indices_length = indices_shape[1:] if global_indices_first else indices_shape[:1]
        return tuple(indices_length + x_shape[1:])

    def call(self, inputs, **kwargs):
        x, index = inputs
        indices_take = index[self.selection_index] if global_indices_first else index[:, self.selection_index]
        return ops.take(x, indices_take, axis=0)


class GatherNodesIngoing(Layer):

    def __init__(self, selection_index: int = 1, **kwargs):
        super(GatherNodesIngoing, self).__init__(**kwargs)
        self.selection_index = selection_index

    def build(self, input_shape):
        super(GatherNodesIngoing, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]),  list(input_shape[1])
        indices_length = indices_shape[1:] if global_indices_first else indices_shape[:1]
        return tuple(indices_length + x_shape[1:])

    def call(self, inputs, **kwargs):
        x, index = inputs
        indices_take = index[self.selection_index] if global_indices_first else index[:, self.selection_index]
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

    This class is used in `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`__ .
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherEdgesPairs, self).__init__(**kwargs)
        self.gather_layer = GatherNodesOutgoing()

    def build(self, input_shape):
        self.gather_layer.build(input_shape)
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
        index_corrected = ops.where(pair_index >= 0, pair_index, ops.zeros_like(pair_index))
        edges_paired = self.gather_layer([edges, index_corrected], **kwargs)
        edges_corrected = ops.where(ops.transpose(pair_index) >= 0, edges_paired, ops.zeros_like(edges_paired))
        return edges_corrected
