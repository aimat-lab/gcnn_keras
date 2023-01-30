import random
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.pooling import PoolingGlobalEdges
from kgcnn.data.utils import ragged_tensor_from_nested_numpy


# This is a very simple mock implementation, because to test the explanation methods we need some sort
# of a model as basis and this model will act as such.
class Model(ks.models.Model):

    def __init__(self,
                 num_targets: int = 1):
        super(Model, self).__init__()
        self.conv_layers = [
            AttentionHeadGATV2(units=64, use_edge_features=True, use_bias=True),
        ]
        self.lay_pooling = PoolingGlobalEdges(pooling_method='sum')
        self.lay_dense = DenseEmbedding(units=num_targets, activation='linear')

    def call(self, inputs, training=False):
        node_input, edge_input, edge_index_input = inputs
        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_index_input])

        pooled = self.lay_pooling(x)
        out = self.lay_dense(pooled)
        return out


class MockContext:

    def __init__(self,
                 num_elements: int = 10,
                 num_targets: int = 1,
                 epochs: int = 10,
                 batch_size: int = 2):
        self.num_elements = num_elements
        self.num_targets = num_targets
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Model(num_targets=num_targets)
        self.x = None
        self.y = None

    def generate_graph(self,
                       num_nodes: int,
                       num_node_attributes: int = 3,
                       num_edge_attributes: int = 1):
        remaining = list(range(num_nodes))
        random.shuffle(remaining)
        inserted = [remaining.pop(0)]
        node_attributes = [[random.random() for _ in range(num_node_attributes)] for _ in range(num_nodes)]
        edge_indices = []
        edge_attributes = []
        while len(remaining) != 0:
            i = remaining.pop(0)
            j = random.choice(inserted)
            inserted.append(i)

            edge_indices += [[i, j], [j, i]]
            edge_attribute = [1 for _ in range(num_edge_attributes)]
            edge_attributes += [edge_attribute, edge_attribute]

        return (
            np.array(node_attributes, dtype=float),
            np.array(edge_attributes, dtype=float),
            np.array(edge_indices, dtype=int)
        )

    def generate_data(self):
        node_attributes_list = []
        edge_attributes_list = []
        edge_indices_list = []
        targets_list = []
        for i in range(self.num_elements):
            num_nodes = random.randint(5, 20)
            node_attributes, edge_attributes, edge_indices = self.generate_graph(num_nodes)
            node_attributes_list.append(node_attributes)
            edge_attributes_list.append(edge_attributes)
            edge_indices_list.append(edge_indices)

            # The target value we will actually determine deterministically here so that our network
            # actually has a chance to learn anything
            target = np.sum(node_attributes)
            targets = [target for _ in range(self.num_targets)]
            targets_list.append(targets)

        self.x = (
            ragged_tensor_from_nested_numpy(node_attributes_list),
            ragged_tensor_from_nested_numpy(edge_attributes_list),
            ragged_tensor_from_nested_numpy(edge_indices_list)
        )

        self.y = (
            np.array(targets_list, dtype=float)
        )

    def __enter__(self):
        # This method will generate random input and output data and thus populate the internal attributes
        # self.x and self.y
        self.generate_data()

        # Using these we will train our mock model for a few very brief epochs.
        self.model.compile(
            loss=ks.losses.mean_squared_error,
            metrics=ks.metrics.mean_squared_error,
            run_eagerly=False,
            optimizer=ks.optimizers.Nadam(learning_rate=0.01),
        )
        hist = self.model.fit(
            self.x, self.y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
        )
        self.history = hist.history

        return self

    def __exit__(self, *args, **kwargs):
        pass
