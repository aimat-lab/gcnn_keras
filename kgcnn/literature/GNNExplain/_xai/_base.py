import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.data.utils import ragged_tensor_from_nested_numpy


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


class AbstractExplanationMethod:

    def __call__(self, model, x, y):
        raise NotImplementedError


class ImportanceExplanationMethod(AbstractExplanationMethod):

    def __init__(self,
                 channels: int):
        self.channels = channels

    def __call__(self,
                model: ks.models.Model,
                x: tf.Tensor,
                y: tf.Tensor
                ) -> t.Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError


class MockImportanceExplanationMethod(ImportanceExplanationMethod):
    """
    This is a mock implementation of "ImportanceExplanationMethod". It is purely for testing purposes.
    Using this method will result in randomly generated importance values for nodes and edges.
    """
    def __init__(self, channels):
        super(MockImportanceExplanationMethod, self).__init__(channels=channels)

    def __call__(self,
                 model: ks.models.Model,
                 x: t.Tuple[tf.Tensor],
                 y: t.Tuple[tf.Tensor],
                 ) -> t.Tuple[tf.Tensor, tf.Tensor]:
        node_input, edge_input, _ = x

        # Im sure you could probably do this in tensorflow directly, but I am just going to go the numpy
        # route here because that's just easier.
        node_input = node_input.numpy()
        edge_input = edge_input.numpy()

        node_importances = [np.random.uniform(0, 1, size=(v.shape[0], self.channels))
                            for v in node_input]
        edge_importances = [np.random.uniform(0, 1, size=(v.shape[0], self.channels))
                            for v in edge_input]

        return (
            ragged_tensor_from_nested_numpy(node_importances),
            ragged_tensor_from_nested_numpy(edge_importances)
        )

