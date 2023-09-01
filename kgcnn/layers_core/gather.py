from typing import Union
from keras_core.layers import Layer
from keras_core import ops


class GatherNodes(Layer):

    def __init__(self, split_indices=(0, 1), concat_axis: Union[int, None] = 1, **kwargs):
        super(GatherNodes, self).__init__(**kwargs)
        self.split_indices = split_indices
        self.concat_axis = concat_axis

    def call(self, inputs, **kwargs):
        x, index = inputs
        gathered = []
        for i in self.split_indices:
            x_i = ops.take(x, index[i], axis=0)
            gathered.append(x_i)
        if self.concat_axis is not None:
            gathered = ops.concatenate(gathered, axis=self.concat_axis)
        return gathered


class GatherNodesOutgoing(Layer):

    def __init__(self, selection_index: int = 1, **kwargs):
        super(GatherNodesOutgoing, self).__init__(**kwargs)
        self.selection_index = selection_index

    def call(self, inputs, **kwargs):
        x, index = inputs
        return ops.take(x, index[self.selection_index], axis=0)