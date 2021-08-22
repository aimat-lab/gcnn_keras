import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesIngoing
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pool.pooling import PoolingLocalEdges


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MessagePassingBase')
class MessagePassingBase(GraphBaseLayer):
    """Base Layer for Message passing.

    """

    def __init__(self, pooling_method="sum", **kwargs):
        super(MessagePassingBase, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.lay_gather_in = GatherNodesIngoing(**self._kgcnn_info)
        self.lay_gather_out = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_pool_default = PoolingLocalEdges(pooling_method=self.pooling_method, **self._kgcnn_info)

    def message_function(self, inputs, **kwargs):
        n_in, n_out, edges = inputs
        raise NotImplementedError("A method to generate messages must be implemented in sub-class of `MessagePassingBase`.")

    def aggregate_message(self, inputs, **kwargs):
        return self.lay_pool_default(inputs)

    def update_nodes(self, inputs, **kwargs):
        nodes, nodes_update = inputs
        raise NotImplementedError("A method to update nodes must be implemented in sub-class of `MessagePassingBase`.")

    def call(self, inputs, **kwargs):
        """Standard message passing scheme using `update_nodes`, `aggregate_message` and `message_function`."""
        nodes, edges, edge_index = inputs
        n_in = self.lay_gather_in([nodes, edge_index])
        n_out = self.lay_gather_out([nodes, edge_index])
        msg = self.message_function([n_in, n_out, edges])
        pool_n = self.aggregate_message([nodes, msg, edge_index])
        n_new = self.update_nodes([nodes, pool_n])
        return n_new

    def get_config(self):
        """Update config."""
        config = super(MessagePassingBase, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config
