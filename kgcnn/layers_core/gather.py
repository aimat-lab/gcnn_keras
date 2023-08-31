from keras_core.layers import Layer
from keras_core import ops


class GatherNodes(Layer):

    def __init__(self, split_indices=(0, 1), concat_axis=1, **kwargs):
        super(GatherNodes, self).__init__(**kwargs)
        self.split_indices = split_indices
        self.concat_axis = concat_axis

    def call(self, inputs, **kwargs):
        x, index = inputs
        gathered = [x[i] for i in self.split_indices]
        if self.concat_axis is not None:
            gathered = ops.concatenate(gathered, axis=self.concat_axis)
        return gathered


class GatherNodesOutgoing(GatherNodes):

    def __init__(self, **kwargs):
        super(GatherNodes, self).__init__(split_indices=1, concat_axis=None, **kwargs)

    def call(self, inputs, **kwargs):
        return super(GatherNodesOutgoing, self).call(inputs, **kwargs)[0]

