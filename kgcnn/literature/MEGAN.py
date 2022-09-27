from typing import List, Tuple, Dict, Optional, Any

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.conv.attention import AttentionHeadGATV2
from kgcnn.layers.base import GraphBaseLayer


class MEGAN(ks.models.Model):

    def __init__(self,
                 # convolutional network related arguments
                 units: List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 use_edge_features: bool = True,
                 # node/edge importance related arguments
                 importance_units: List[int] = [],
                 importance_channels: int = 2,
                 importance_dropout_rate: float = 0.0,
                 importance_factor: float = 0.0,
                 importance_multiplier: float = 1.0,
                 sparsity_factor: float = 0.0,
                 # mlp tail end related arguments
                 final_units: List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 regression_limits: Optional[Tuple[float, float]] = None,
                 regression_reference: Optional[float] = None,
                 **kwargs):
        super(MEGAN, self).__init__(self, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_edge_features = use_edge_features
        self.importance_units = importance_units
        self.importance_channels = importance_channels
        self.importance_dropout_rate = importance_dropout_rate
        self.importance_factor = importance_factor
        self.importance_multiplier = importance_multiplier
        self.sparsity_factor = sparsity_factor
        self.final_units = final_units
        self.final_dropout_rate = final_dropout_rate
        self.final_activation = final_activation
        self.regression_limits = regression_limits
        self.regression_reference = regression_reference

        self.attention_layers: List[GraphBaseLayer] = []
        for unit in self.units:
            # lay = MultiHeadGatv2Layer(
            #     units=units,
            #     num_heads=self.importance_channels,
            #     use_edge_features=self.use_edge_features,
            #     activation=self.activation,
            #     use_bias=self.use_bias,
            #     has_self_loops=True,
            #     concat_heads=True
            # )
            # self.attention_layers.append(lay)
            pass

