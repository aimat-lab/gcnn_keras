import typing as t

import tensorflow as tf


class AbstractExplanationMixin:

    def explain(self, x):
        raise NotImplementedError()


class ImportanceExplanationMixin:

    def explain(self, x, **kwargs):
        return self.explain_importances(x, **kwargs)

    # Returns a tuple of ragged tensors (node_importances, edge_importances)
    def explain_importances(self,
                            x: t.Sequence[tf.Tensor],
                            **kwargs
                            ) -> t.Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        raise NotImplementedError
