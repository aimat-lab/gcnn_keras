from keras import Layer
from keras import ops


class ExplanationSparsityRegularization(Layer):

    def __init__(self,
                 factor: float = 1.0,
                 **kwargs):
        super(ExplanationSparsityRegularization, self).__init__(**kwargs)
        self.factor = factor

    def build(self, input_shape):
        super(ExplanationSparsityRegularization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Computes a loss from importance scores.

        Args:
            inputs: Importance tensor of shape ([batch], [N], K) .

        Returns:
            None.
        """
        # importances: ([batch], [N], K)
        importances = inputs

        loss = ops.mean(ops.abs(importances))
        loss = loss * self.factor
        self.add_loss(loss)
        return loss