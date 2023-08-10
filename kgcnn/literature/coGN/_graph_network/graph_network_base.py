from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherEmbeddingSelection
from kgcnn.layers.aggr import (
    AggregateLocalEdges,
    AggregateLocalEdgesAttention,
)
from kgcnn.layers.pooling import PoolingNodes, PoolingGlobalEdges, PoolingEmbeddingAttention
from copy import copy


def map_dict(f, x):
    if isinstance(x, dict):
        return {k: f(v) for k, v in x.items()}
    else:
        return f(x)


class GraphNetworkBase(GraphBaseLayer):
    """Graph Network according to "Relational inductive biases, deep learning, and graph networks" by Battaglia et al.
    (<https://arxiv.org/abs/1806.01261>)"""

    def __init__(
        self,
        aggregate_edges_local="sum",
        aggregate_edges_global="sum",
        aggregate_nodes="sum",
        return_updated_edges=True,
        return_updated_nodes=True,
        return_updated_globals=True,
        **kwargs
    ):
        """Initialize the GraphNetworkBase layer.

        Args:
            aggregate_edges_local (str, optional): Identifier for the local edge aggregation function.
                Defaults to "sum".
            aggregate_edges_global (str, optional): Identifier for the global edge aggregation function.
                Defaults to "sum".
            aggregate_nodes (str, optional): Identifier for the node aggregation function.
                Defaults to "sum".
            return_updated_edges (bool, optional): Whether to return updated node features.
                May be set to False, if edge features are only used as messages and not updated between GN layers.
                Defaults to True.
            return_updated_nodes (bool, optional): Whether to return updated edges features. Defaults to True.
            return_updated_globals (bool, optional): Whether to return updated global graph features. Defaults to True.
        """
        super().__init__(**kwargs)
        self.aggregate_edges_local_ = aggregate_edges_local
        self.aggregate_edges_global_ = aggregate_edges_global
        self.aggregate_nodes_ = aggregate_nodes
        self.lay_gather = GatherEmbeddingSelection([0, 1])

        self.attention_strings = ["attention"]

        if isinstance(self.aggregate_edges_local_, str):
            if self.aggregate_edges_local_ in self.attention_strings:
                self.lay_pool_edges_local = AggregateLocalEdgesAttention()
            else:
                self.lay_pool_edges_local = AggregateLocalEdges(
                    pooling_method=self.aggregate_edges_local_
                )

        if isinstance(self.aggregate_edges_global_, str):
            if self.aggregate_edges_global_ in self.attention_strings:
                self.lay_pool_edges_global = PoolingEmbeddingAttention()
            else:
                self.lay_pool_edges_global = PoolingGlobalEdges(
                    pooling_method=self.aggregate_edges_global_
                )

        if isinstance(aggregate_nodes, str):
            if self.aggregate_nodes_ in self.attention_strings:
                self.lay_pool_nodes = PoolingEmbeddingAttention()
            else:
                self.lay_pool_nodes = PoolingNodes(pooling_method=self.aggregate_nodes_)

        self.return_updated_edges = return_updated_edges
        self.return_updated_nodes = return_updated_nodes
        self.return_updated_globals = return_updated_globals

    @staticmethod
    def get_features(x):
        """Getter for edge/node/graph features.

        If the argument is a Tensor it is returned as it is.
        If the argument is a dict the value for the "features" key is returned.
        """
        if isinstance(x, dict):
            assert "features" in x.keys()
            return x["features"]
        else:
            return x

    @staticmethod
    def update_features(x, v):
        """Setter for edge/node/graph features.

        Args:
            x: Tensor/dict to update
            v: New feature value.

        Returns:
            Updated Tensor or dict.
        """
        if isinstance(x, dict):
            x_ = copy(x)
            x_["features"] = v
            return x_
        else:
            return v

    @staticmethod
    def get_attribute(x, k):
        if isinstance(x, dict):
            assert k in x.keys()
            return x[k]
        else:
            raise ValueError()

    @staticmethod
    def update_attribute(x, k, v):
        if isinstance(x, dict):
            x_ = copy(x)
        else:
            x_ = {"features": x}
        x_[k] = v
        return x_

    def aggregate_edges_local(self, nodes, edges, edge_indices):
        """Function that aggregates local edges (a.k.a. message passing aggregation).

        In the GN framework this is notaded as `ρ_{E -> V}`.

        Args:
            nodes: Node attributes/features for the graphs as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [N], F_V).
            edges: Edge attributes/features for the graphs as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [M], F_E).
            edge_indices: Edge indices of the graph topologies as as RaggedTensor.
                Shape: (batch, [M], 2)

        Returns:
            Aggregated edges for each node as RaggedTensor(s) of shape: (batch, [N], F_V).
        """

        if self.aggregate_edges_local_ is None:
            return edges

        node_features = self.get_features(nodes)
        if self.aggregate_edges_local_ in self.attention_strings:
            edge_attention_local = self.get_attribute(edges, "attention_local")
            edge_features = self.get_features(edges)
            aggregated_edges = self.lay_pool_edges_local(
                [node_features, edge_features, edge_attention_local, edge_indices]
            )
        else:
            edge_features = self.get_features(edges)
            aggregated_edges = self.lay_pool_edges_local(
                [node_features, edge_features, edge_indices]
            )

        return aggregated_edges

    def aggregate_edges_global(self, edges):
        """Function that aggregates global edges (a.k.a. global edge readout).

        In the GN framework this is notaded as `ρ_{E -> G}`.

        Args:
            edges: Edge attributes/features for the graphs as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [M], F_E).

        Returns:
            Aggregated edges for each the graphs as RaggedTensor(s) of shape: (batch, F_G).
        """

        if self.aggregate_edges_global_ is None:
            return edges

        if self.aggregate_edges_global_ in self.attention_strings:
            edge_attention_global = self.get_attribute(edges, "attention_global")
            edge_features = self.get_features(edges)
            aggregated_edges = self.lay_pool_edges_global(
                [edge_features, edge_attention_global]
            )
        else:
            edge_features = self.get_features(edges)
            aggregated_edges = self.lay_pool_edges_global(edge_features)

        return aggregated_edges

    def aggregate_nodes(self, nodes):
        """Function that aggregates nodes (a.k.a. global (node) readout).

        In the GN framework this is notaded as `ρ_{V -> G}`.

        Args:
            nodes: Node attributes/features for the graphs as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [N], F_V).

        Returns:
            Aggregated nodes for each the graphs as RaggedTensor(s) of shape (batch, F_G).
        """

        if self.aggregate_nodes_ is None:
            return nodes

        if self.aggregate_nodes_ in self.attention_strings:
            node_attention = self.get_attribute(nodes, "attention")
            node_features = self.get_features(nodes)
            aggregated_nodes = self.lay_pool_nodes([node_features, node_attention])
        else:
            node_features = self.get_features(nodes)
            aggregated_nodes = self.lay_pool_nodes(node_features)

        return aggregated_nodes

    def update_edges(self, edges, nodes_in, nodes_out, globals_, **kwargs):
        """Update edges on basis of receiver, sender attributes/features,
        edge features/attributes and global features/attributes.
        
        In the GN framework this is notaded as `ɸ_E`.

        The default implementation leaves edge features/attributes unchanged.
        Use subclasses for other implementations.

        Args:
            edges: Edge attributes/features for the graphs as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [M], F_E).
            nodes_in: Receiver node attributes/features for for each edge as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [M], F_V).
            nodes_out (_type_): Sender node attributes/features for each edge as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [M], F_V).
            globals_: Global graph node attributes/features for each edge as Tensor(s).
                Can be a single Tensor (hidden node representations) or multiple Tensors in a dict.
                Tensor(s) must be of shape: (batch, G).

        Returns:
            Updated Edge attributes/features for the graphs as RaggedTensor(s).
            Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
            RaggedTensor(s) must be of shape: (batch, [M], F_E).
        """
        # edges = inputs[0] # shape: (batch, [M], E)
        # nodes_in = inputs[1]  # shape: (batch, [M], F)
        # nodes_out = inputs[2] # shape: (batch, [M], F)
        # globals_ = inputs[3] # shape: (batch, G)
        edge_new = edges  # Implement this part in sub class
        return edge_new  # shape: (batch, [M], E)

    def update_nodes(self, aggregated_edges, nodes, globals_, **kwargs):
        """Update nodes on basis of updated edge features/attributes and global features/attributes.
        
        In the GN framework this is notaded as `ɸ_V`.

        The default implementation leaves edge features/attributes unchanged.
        Use subclasses for other implementations.

        Args:
            aggregated_edges: Aggregated edge features for each node as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [M], F_V).
            nodes: Node attributes/features for the graphs as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, [N], F_V).
            globals_: Global graph node attributes/features for each edge as Tensor(s).
                Can be a single Tensor (hidden node representations) or multiple Tensors in a dict.
                Tensor(s) must be of shape: (batch, G).

        Returns:
            Updated node attributes/features for the graphs as RaggedTensor(s).
            Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
            RaggedTensor(s) must be of shape: (batch, [N], F_V).
        """
        # aggregated_edges shape: (batch, [N], E) or (batch, E) if aggregate_edges_local is None
        # nodes shape: (batch, [N], F)
        # globals_ shape: (batch, G)
        nodes_new = nodes  # Implement this part in sub class
        return nodes_new  # shape: (batch, [N], F)

    def update_globals(self, aggregated_edges, aggregated_nodes, globals_, **kwargs):
        """Update nodes on basis of aggregated edge features/attributes,
        aggregated node features/attributes and global features/attributes (a.k.a. global readout function).
        
        In the GN framework this is notaded as `ɸ_G`.

        The default implementation leaves edge features/attributes unchanged.
        Use subclasses for other implementations.

        Args:
            aggregated_edges: Aggregated edge features for each graph as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, G).
            aggregated_nodes: Aggregated node features for each graph as RaggedTensor(s).
                Can be a single RaggedTensor (hidden node representations) or multiple RaggedTensors in a dict.
                RaggedTensor(s) must be of shape: (batch, G).
            globals_: Global graph node attributes/features for each edge as Tensor(s).
                Can be a single Tensor (hidden node representations) or multiple Tensors in a dict.
                Tensor(s) must be of shape: (batch, G).

        Returns:
            Updated global graph node attributes/features for each edge as Tensor(s).
            Can be a single Tensor (hidden node representations) or multiple Tensors in a dict.
            Tensor(s) must be of shape: (batch, G).
        """
        # aggregated_edges shape: (batch, E) or (batch, [M], E) if aggregate_edges_global is None
        # aggregated_nodes shape: (batch, F) or (batch, [N], F) if aggregate_nodes is None
        # global_features shape: (batch, G)
        globals_new = globals_  # Implement this part in sub class
        return globals_new  # shape: (batch, G)

    def call(self, inputs, **kwargs):
        """Graph Network algorithm.

        Args:
            inputs: List of edge, node, global graph features and edge indices, which describe graph topologies.

        Returns:
            List of updated edge, node, global graph features and edge indices, which describe graph topologies.
        """
        edges, nodes, globals_, edge_indices = inputs

        def lay_gather_(x):
            return self.lay_gather([x, edge_indices])

        def first(x):
            if isinstance(x, list):
                return x[0]
            else:
                return x

        def second(x):
            if isinstance(x, list):
                return x[1]
            else:
                return x

        gathered = map_dict(lay_gather_, nodes)
        nodes_in = map_dict(first, gathered)
        nodes_out = map_dict(second, gathered)

        edges_new = self.update_edges(edges, nodes_in, nodes_out, globals_)
        aggregated_edges_local = self.aggregate_edges_local(
            nodes, edges_new, edge_indices
        )
        aggregated_edges_global = self.aggregate_edges_global(edges_new)
        nodes_new = self.update_nodes(aggregated_edges_local, nodes, globals_)
        aggregated_nodes = self.aggregate_nodes(nodes_new)
        globals_new = self.update_globals(
            aggregated_edges_global, aggregated_nodes, globals_
        )
        edges_new = edges_new if self.return_updated_edges else edges
        nodes_new = nodes_new if self.return_updated_nodes else nodes
        globals_new = globals_new if self.return_updated_globals else globals_

        return edges_new, nodes_new, globals_new, edge_indices

    def get_config(self):
        config = super(GraphNetworkBase, self).get_config()
        config.update(
            {
                "aggregate_edges_local": self.aggregate_edges_local_,
                "aggregate_edges_global": self.aggregate_edges_global_,
                "aggregate_nodes": self.aggregate_nodes_,
            }
        )
        return config
