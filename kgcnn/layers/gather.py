from typing import Union
from keras_core.layers import Layer, Concatenate
from keras_core import ops


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
        for _ in self.split_indices:
            xs.append(indices_shape[1:] + x_shape[1:])
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
            x_i = ops.take(x, index[i], axis=0)
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
        return tuple(indices_shape[1:] + x_shape[1:])

    def call(self, inputs, **kwargs):
        x, index = inputs
        return ops.take(x, index[self.selection_index], axis=0)


class GatherNodesIngoing(Layer):

    def __init__(self, selection_index: int = 1, **kwargs):
        super(GatherNodesIngoing, self).__init__(**kwargs)
        self.selection_index = selection_index

    def build(self, input_shape):
        super(GatherNodesIngoing, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        x_shape, indices_shape = list(input_shape[0]),  list(input_shape[1])
        return tuple(indices_shape[1:] + x_shape[1:])

    def call(self, inputs, **kwargs):
        x, index = inputs
        return ops.take(x, index[self.selection_index], axis=0)


class GatherState(Layer):

    def __init__(self, selection_index: int = 1, **kwargs):
        super(GatherState, self).__init__(**kwargs)
        self.selection_index = selection_index

    def build(self, input_shape):
        super(GatherState, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        state_shape, _ = list(input_shape[0]),  list(input_shape[1])
        return tuple([None] + state_shape[1:])

    def call(self, inputs, **kwargs):
        env, target_len = inputs
        out = ops.repeat(env, target_len, axis=0)
        return out
