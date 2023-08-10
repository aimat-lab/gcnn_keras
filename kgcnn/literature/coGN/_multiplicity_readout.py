import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.segment import segment_ops_by_name
from kgcnn.layers.modules import LazyMultiply
from kgcnn.layers.pooling import PoolingEmbedding


class MultiplicityReadout(PoolingEmbedding):
    """MultiplicityReadout layer for asymmetric unit crystal graph representations."""

    def __init__(self, pooling_method="mean", **kwargs):

        super().__init__(pooling_method=pooling_method, **kwargs)
        self.lazy_multiply = LazyMultiply()

        multiplicity_invariant_pooling_operations = [
            "segment_min",
            "sum",
            "reduce_min",
            "segment_max",
            "max",
            "reduce_max",
        ]
        no_normalization_pooling_operations = ["segment_sum", "sum", "reduce_sum"]

        if pooling_method in multiplicity_invariant_pooling_operations:
            self.use_multiplicities = False
        else:
            self.use_multiplicities = True

        if (
            self.use_multiplicities
            and pooling_method in no_normalization_pooling_operations
        ):
            self.multiplicity_normalization = False
        else:
            self.multiplicity_normalization = True

    def get_multiplicity_normalization_factor(self, multiplicity):
        numerator = tf.cast(multiplicity.row_lengths(), multiplicity.dtype)
        denominator = segment_ops_by_name(
            "sum", multiplicity.values, multiplicity.value_rowids()
        )[:, 0]
        normalization_factor = numerator / denominator
        return tf.expand_dims(normalization_factor, -1)

    def call(self, inputs, **kwargs):

        features, multiplicity = inputs[0], inputs[1]

        if self.use_multiplicities:
            features_weighted = self.lazy_multiply([features, multiplicity])
            aggregated_features = super().call(features_weighted, **kwargs)
            if self.multiplicity_normalization:
                normalization_factor = self.get_multiplicity_normalization_factor(
                    multiplicity
                )
                aggregated_features = aggregated_features * normalization_factor
            return aggregated_features
        else:
            return super().call(features, **kwargs)
