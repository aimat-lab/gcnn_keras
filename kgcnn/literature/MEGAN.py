from typing import List, Tuple, Dict, Optional, Any

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.modules import ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.modules import LazyConcatenate, LazyAverage
from kgcnn.layers.conv.gat_conv import MultiHeadGATV2Layer
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.pooling import PoolingWeightedNodes, PoolingNodes


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
                 importance_activation: str = 'sigmoid',
                 importance_dropout_rate: float = 0.0,
                 importance_factor: float = 0.0,
                 importance_multiplier: float = 1.0,
                 sparsity_factor: float = 0.0,
                 # mlp tail end related arguments
                 final_units: List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 final_pooling: str = 'sum',
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
        self.importance_activation = importance_activation
        self.importance_dropout_rate = importance_dropout_rate
        self.importance_factor = importance_factor
        self.importance_multiplier = importance_multiplier
        self.sparsity_factor = sparsity_factor
        self.final_units = final_units
        self.final_dropout_rate = final_dropout_rate
        self.final_activation = final_activation
        self.final_pooling = final_pooling
        self.regression_limits = regression_limits
        self.regression_reference = regression_reference

        # ~ MAIN CONVOLUTIONAL / ATTENTION LAYERS
        self.attention_layers: List[GraphBaseLayer] = []
        for u in self.units:
            lay = MultiHeadGATV2Layer(
                units=u,
                num_heads=self.importance_channels,
                use_edge_features=self.use_edge_features,
                activation=self.activation,
                use_bias=self.use_bias,
                has_self_loops=True,
                concat_heads=True
            )
            self.attention_layers.append(lay)

        self.lay_dropout = DropoutEmbedding(rate=self.dropout_rate)

        # ~ EDGE IMPORTANCES
        self.lay_act_importance = ActivationEmbedding(activation=self.importance_activation)
        self.lay_concat_alphas = LazyConcatenate(axis=-1)

        self.lay_pool_edges_in = PoolingLocalEdges(pooling_method='mean', pooling_index=0)
        self.lay_pool_edges_out = PoolingLocalEdges(pooling_method='mean', pooling_index=1)
        self.lay_average = LazyAverage()

        # ~ NODE IMPORTANCES
        self.node_importance_units = importance_units + [self.importance_channels]
        self.node_importance_acts = ['relu' for _ in importance_units] + ['linear']
        self.node_importance_layers = []
        for u, act in zip(self.node_importance_units, self.node_importance_acts):
            lay = DenseEmbedding(
                units=u,
                activation=act,
                use_bias=use_bias
            )
            self.node_importance_layers.append(lay)

        # ~ OUTPUT / MLP TAIL END
        self.lay_pool_out = PoolingNodes(pooling_method=self.final_pooling)
        self.lay_concat_out = LazyConcatenate(axis=-1)
        self.lay_final_dropout = DropoutEmbedding(rate=self.final_dropout_rate)

        self.final_acts = ['relu' for _ in self.final_units]
        self.final_acts[-1] = self.final_activation
        self.final_layers = []
        for u, act in zip(self.final_units, self.final_acts):
            lay = DenseEmbedding(
                units=u,
                activation=act,
                use_bias=use_bias
            )
            self.final_layers.append(lay)

        # ~ EXPLANATION ONLY TRAIN STEP

    def call(self,
             inputs):

        node_input, edge_input, edge_index_input = inputs

        alphas = []
        x = node_input
        for lay in self.attention_layers:
            x, alpha = lay([x, edge_input, edge_index_input])
            x = self.lay_dropout(x)

            alphas.append(alpha)

        alphas = self.lay_concat_alphas(alphas)
        edge_importances = tf.reduce_sum(alphas, axis=-1, keepdims=False)
        edge_importances = self.lay_act_importance(edge_importances)

        pooled_edges_in = self.lay_pool_edges_in([node_input, edge_importances, edge_index_input])
        pooled_edges_out = self.lay_pool_edges_out([node_input, edge_importances, edge_index_input])
        pooled_edges = self.lay_average([pooled_edges_out, pooled_edges_in])

        node_importances_tilde = x
        for lay in self.node_importance_layers:
            node_importances_tilde = lay(node_importances_tilde)

        node_importances_tilde = self.lay_act_importance(node_importances_tilde)

        node_importances = node_importances_tilde * pooled_edges

        outs = []
        for k in range(self.importance_channels):
            node_importance_slice = tf.expand_dims(node_importances[:, :, k], axis=-1)
            out = self.lay_pool_out(x * node_importance_slice)

            outs.append(out)

        out = self.lay_concat_out(outs)

        for lay in self.final_layers:
            out = lay(out)
            self.lay_final_dropout(out)

        return out, node_importances, edge_importances


