import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import LazyMultiply, LazyAdd, LazyConcatenate
from kgcnn.layers.gather import GatherState
from kgcnn.ops.segment import segment_ops_by_name

from .graph_network_base import GraphNetworkBase

class GraphNetwork(GraphNetworkBase):
    """A basic concrete implementation of the GraphNetworkBase class.

    Update functions `ɸ_E`,`ɸ_V`,`ɸ_G` can be provided as parameters to the constructor.
    Aggregation functions `ρ_{E -> V}`,`ρ_{E -> G}`,`ρ_{V -> G}` can be selected via string identifiers
    (`'sum'`,`'mean'`,`'max'`,`'min'`,`'attention'`).
    It furthermore supports en/disabling gated updates, residual updates and which features are used for
    the update functions.
    
    Graph Network according to "Relational inductive biases, deep learning, and graph networks" by Battaglia et al.
    (<https://arxiv.org/abs/1806.01261>).

    """

    def __init__(self, edge_mlp, node_mlp, global_mlp,
                 aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
                 return_updated_edges=True, return_updated_nodes=True, return_updated_globals=True,
                 edge_attention_mlp_local=None, edge_attention_mlp_global=None, node_attention_mlp=None,
                 edge_gate=None, node_gate=None, global_gate=None,
                 residual_edge_update=True, residual_node_update=False, residual_global_update=False,
                 update_edges_input=[True, True, True, False], # [edges, nodes_in, nodes_out, globals_]
                 update_nodes_input=[True, False, False], # [aggregated_edges, nodes, globals_]
                 update_global_input=[False, True, False], # [aggregated_edges, aggregated_nodes, globals_]
                 **kwargs):
        """Instantiates the Graph Network block/layer.

        Args:
            edge_mlp (kgcnn.layers.mlp.MLP): Edge update function.
            node_mlp (kgcnn.layers.mlp.MLP): Node update function.
            global_mlp (kgcnn.layers.mlp.MLP): Global update function.
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
            edge_attention_mlp_local (kgcnn.layers.mlp.MLP, optional): MLP to calculate the attention value for
                attention local aggregation of edges.
                Defaults to None.
            edge_attention_mlp_global (kgcnn.layers.mlp.MLP, optional): MLP to calculate the attention value for
                attention global aggregation of edges.
                Defaults to None.
            node_attention_mlp (kgcnn.layers.mlp.MLP, optional): MLP to calculate the attention value for
                attention aggregation of nodes.
                Defaults to None.
            edge_gate (optional): Gating layer for edge updates (e.g. GRUCell). Defaults to None.
            node_gate (optional): Gating layer for node updates (e.g. GRUCell). Defaults to None.
            global_gate (optional): Gating layer for global updates (e.g. GRUCell). Defaults to None.
            residual_edge_update (bool, optional): Whether to do residual updates or not. Defaults to True.
            residual_node_update (bool, optional): Whether to do residual updates or not. Defaults to False.
            residual_global_update (bool, optional): Whether to do residual updates or not. Defaults to False.
            update_edges_input (list, optional): Whether to include [edges, nodes_in, nodes_out, global] features
                in the edge update function.
                Defaults to [True, True, True, False].
            update_node_input (list, optional): Whether to include [aggregated_edges, nodes, global] features
                in the node update.
                Defaults to [True, False, False].
            update_global_input (list, optional): Whether to include [aggregated_edges, aggregated_nodes, global] features
                in the global update.
                Defaults to [False, True, False].
        """
        super().__init__(
                aggregate_edges_local=aggregate_edges_local,
                aggregate_edges_global=aggregate_edges_global,
                aggregate_nodes=aggregate_nodes,
                return_updated_edges=return_updated_edges,
                return_updated_nodes=return_updated_nodes,
                return_updated_globals=return_updated_globals,
                **kwargs)
        self.edge_mlp = edge_mlp
        if self.aggregate_edges_local_ in self.attention_strings:
            assert edge_attention_mlp_local is not None
            self.edge_attention_mlp_local = edge_attention_mlp_local
        if self.aggregate_edges_global_ in self.attention_strings:
            assert edge_attention_mlp_global is not None
            self.edge_attention_mlp_local = edge_attention_mlp_global
        if self.aggregate_nodes_ in self.attention_strings:
            assert node_attention_mlp is not None
            self.node_attention_mlp = node_attention_mlp

        self.residual_node_update = residual_node_update
        self.residual_edge_update = residual_edge_update
        self.residual_global_update = residual_global_update

        self.update_edges_input = update_edges_input
        self.update_nodes_input = update_nodes_input
        self.update_global_input = update_global_input

        self.edge_gate = edge_gate
        self.node_gate = node_gate
        self.global_gate = global_gate
        self.node_mlp = node_mlp
        self.global_mlp = global_mlp
        self.lazy_add = LazyAdd()
        self.lazy_concat = LazyConcatenate(axis=-1)
        self.lazy_multiply = LazyMultiply()
        self.gather_state = GatherState()
            
    def update_edges(self, edges, nodes_in, nodes_out, globals_, **kwargs):

        if self.edge_mlp is None:
            return edges
        
        edge_features = self.get_features(edges)
        features_to_concat = []
        if self.update_edges_input[0]:
            features_to_concat.append(edge_features)
        if self.update_edges_input[1]:
            nodes_in_features = self.get_features(nodes_in)
            features_to_concat.append(nodes_in_features)
        if self.update_edges_input[2]:
            nodes_out_features = self.get_features(nodes_out)
            features_to_concat.append(nodes_out_features)
        if self.update_edges_input[3]:
            global_features = self.get_features(globals_)
            features_to_concat.append(self.gather_state([global_features, edge_features]))
        
        concat_features = self.lazy_concat(features_to_concat)
        messages = self.edge_mlp(concat_features)

        if self.edge_gate is not None:
            messages = tf.RaggedTensor.from_row_splits(
                    self.edge_gate(messages.values, edge_features.values)[1],
                    messages.row_splits)
        
        if self.residual_edge_update:
            messages = self.lazy_add([edge_features, messages])

        edges_new = self.update_features(edges, messages)
        
        if self.aggregate_edges_local_ in self.attention_strings:
            attention_local = self.edge_attention_mlp_local(messages)  
            edges_new = self.update_attribute(edges_new, 'attention_local', attention_local)
        elif self.aggregate_edges_global_ in self.attention_strings:
            attention_global = self.edge_attention_mlp_global(messages)  
            edges_new = self.update_attribute(edges_new, 'attention_global', attention_global)

        return edges_new

    def update_nodes(self, aggregated_edges, nodes, globals_, **kwargs):
        
        if self.node_mlp is None:
            return nodes

        aggregated_edge_features = self.get_features(aggregated_edges)
        node_features = self.get_features(nodes)

        features_to_concat = []
        if self.update_nodes_input[0]:
            features_to_concat.append(aggregated_edge_features)
        if self.update_nodes_input[1]:
            features_to_concat.append(node_features)
        if self.update_nodes_input[2]:
            global_features = self.get_features(globals_)
            features_to_concat.append(self.gather_state([global_features, node_features]))
        
        concat_features = self.lazy_concat(features_to_concat)
        node_features_new = self.node_mlp(concat_features)

        if self.node_gate is not None:
            node_features_new = tf.RaggedTensor.from_row_splits(
                    self.node_gate(node_features_new.values, node_features.values)[1],
                    node_features_new.row_splits, validate=self.ragged_validate)

        if self.residual_node_update:
            node_features_new = self.lazy_add([node_features, node_features_new])
        
        nodes_new = self.update_features(nodes, node_features_new)
        
        if self.aggregate_nodes_ in self.attention_strings:
            attention = self.node_attention_mlp(node_features_new)
            nodes_new = self.update_attribute(nodes_new, 'attention', attention)
        return nodes_new

    def update_globals(self, aggregated_edges, aggregated_nodes, globals_, **kwargs):
        
        if self.global_mlp is None:
            return globals_

        features_to_concat = []
        if self.update_global_input[0]:
            aggregated_edge_features = self.get_features(aggregated_edges)
            features_to_concat.append(aggregated_edge_features)
        if self.update_global_input[1]:
            aggregated_node_features = self.get_features(aggregated_nodes)
            features_to_concat.append(aggregated_node_features)
        if self.update_global_input[2]:
            global_features = self.get_features(globals_)
            features_to_concat.append(global_features)

        concat_features = self.lazy_concat(features_to_concat)
        global_features_new = self.global_mlp(concat_features)

        if self.global_gate is not None:
            global_features_new = tf.RaggedTensor.from_row_splits(
                    self.global_gate(global_features_new.values, global_features.values)[1],
                    global_features_new.row_splits, validate=self.ragged_validate)

        if self.residual_global_update:
            global_features = self.get_features(globals_)
            global_features_new = self.lazy_add([global_features, global_features_new])

        globals_new = self.update_features(globals_, global_features_new)

        return globals_new
    

class GraphNetworkMultiplicityReadout(GraphNetwork):
    """Same as a `GraphNetwork` but with multiplicity readout for asymmetric unit graphs.

    Multiplicity values must be attached to nodes with a `multiplicity` key, for multiplicity readout to work.
    """

    def __init__(self, edge_mlp, node_mlp, global_mlp,
                 aggregate_edges_local="sum", aggregate_edges_global=None, aggregate_nodes="sum",
                 return_updated_edges=True, return_updated_nodes=True, return_updated_globals=True,
                 edge_attention_mlp_local=None, edge_attention_mlp_global=None, node_attention_mlp=None,
                 **kwargs):
        super().__init__(edge_mlp, node_mlp, global_mlp,
                aggregate_edges_local=aggregate_edges_local,
                aggregate_edges_global=aggregate_edges_global,
                aggregate_nodes=aggregate_nodes,
                return_updated_edges=return_updated_edges,
                return_updated_nodes=return_updated_nodes,
                return_updated_globals=return_updated_globals,
                edge_attention_mlp_local=edge_attention_mlp_local,
                edge_attention_mlp_global=edge_attention_mlp_global,
                node_attention_mlp=node_attention_mlp, **kwargs)

        multiplicity_invariant_pooling_operations = ["segment_min", "min", "reduce_min",
                "segment_max", "max", "reduce_max"]
        no_normalization_pooling_operations = ["segment_sum", "sum", "reduce_sum"]
            
        if aggregate_edges_global is not None:
            if aggregate_edges_global in multiplicity_invariant_pooling_operations:
                self.use_edge_multiplicities = False
            else:
                self.use_edge_multiplicities = True
    
            if self.use_edge_multiplicities and aggregate_edges_global in no_normalization_pooling_operations:
                self.edge_multiplicity_normalization = False
            else:
                self.edge_multiplicity_normalization = True
        else:
            self.use_edge_multiplicities = False
            self.edge_multiplicity_normalization = False

        if aggregate_nodes is not None:
            if aggregate_nodes in multiplicity_invariant_pooling_operations:
                self.use_node_multiplicities = False
            else:
                self.use_node_multiplicities = True

            if self.use_node_multiplicities and aggregate_nodes in no_normalization_pooling_operations:
                self.node_multiplicity_normalization = False
            else:
                self.node_multiplicity_normalization = True
        else:
            self.use_node_multiplicities = False
            self.node_multiplicity_normalization = False

    def get_multiplicity_normalization_factor(self, multiplicity):
        numerator = tf.cast(multiplicity.row_lengths(), multiplicity.dtype)
        denominator = segment_ops_by_name('sum', multiplicity.values, multiplicity.value_rowids())
        normalization_factor = numerator / denominator
        return tf.expand_dims(normalization_factor, -1)

    def aggregate_edges_global(self, edges, **kwargs):

        if self.aggregate_edges_global_ is None:
            return edges

        if self.use_edge_multiplicities:
            multiplicity = self.get_attribute(edges, 'multiplicity')
            edge_features = self.get_features(edges)
            edge_features_weighted = self.lazy_multiply([edge_features, multiplicity])
            edge_features_weighted = self.update_features(edges, edge_features_weighted)
            aggregated_edge_features = super().aggregate_edges_global(edge_features_weighted, **kwargs)
            if self.edge_multiplicity_normalization:
                normalization_factor = self.get_multiplicity_normalization_factor(multiplicity)
                aggregated_edge_features_ = self.get_features(aggregated_node_features) * normalization_factor
                aggregated_edge_features = self.update_features(edges, aggregated_edge_features_)
            return aggregated_edge_features
        else:
            return super().aggregate_edges_global(edges, **kwargs)

    def aggregate_nodes(self, nodes, **kwargs):
        if self.aggregate_nodes_ is None:
            return nodes

        if self.use_node_multiplicities:
            multiplicity = self.get_attribute(nodes, 'multiplicity')
            node_features = self.get_features(nodes)
            node_features_weighted_ = self.lazy_multiply([node_features, multiplicity])
            node_features_weighted = self.update_features(nodes, node_features_weighted_)
            aggregated_node_features = super().aggregate_nodes(node_features_weighted, **kwargs)
            if self.node_multiplicity_normalization:
                normalization_factor = self.get_multiplicity_normalization_factor(multiplicity)
                aggregated_node_features_ = self.get_features(aggregated_node_features) * normalization_factor
                aggregated_node_features = self.update_features(nodes, aggregated_node_features_)
            return aggregated_node_features
        else:
            return super().aggregate_nodes(nodes, **kwargs)


class CrystalInputBlock(GraphNetworkBase):
    """Graph Network layer that embeds node and edges features of crystal graphs on the basis of atomic numbers (for nodes) and distances (for edges)."""

    def __init__(self,
            atom_embedding,
            edge_embedding,
            atom_mlp=None,
            edge_mlp=None,
            **kwargs):
        """Initialize crystal embedding layer.

        Args:
            atom_embedding (kgcnn.literature.coGN.embedding_layers.atom_embedding.AtomEmbedding):
                AtomEmbedding layer to use for graph nodes.
            edge_embedding (kgcnn.literature.coGN.embedding_layers.edge_embedding.EdgeEmbedding):
                EdgeEmbedding layer to use for graph edges.
            atom_mlp (kgcnn.layers.mlp.MLP, optional): Optional MLP layer that is applied to nodes after embedding.
                Defaults to None.
            edge_mlp (kgcnn.layers.mlp.MLP, optional): Optional MLP layer that is applied to edges after embedding.
                Defaults to None.
        """
        super().__init__(aggregate_edges_local=None, aggregate_edges_global=None, aggregate_nodes=None, **kwargs)
        self.atom_embedding = atom_embedding
        self.edge_embedding = edge_embedding
        self.atom_mlp = atom_mlp
        self.edge_mlp = edge_mlp

    def update_edges(self, edges, nodes_in, nodes_out, globals_, **kwargs):
        edge_features_new = self.edge_embedding(self.get_features(edges))
        if self.edge_mlp:
            edge_features_new = self.edge_mlp(edge_features_new)
        edges_new = self.update_features(edges, edge_features_new)
        return edges_new

    def update_nodes(self, aggregated_edge_features, nodes, global_features, **kwargs):
        node_features_new = self.atom_embedding(self.get_features(nodes))
        if self.atom_mlp:
            node_features_new = self.atom_mlp(node_features_new)
        nodes_new = self.update_features(nodes, node_features_new)
        return nodes_new


class SequentialGraphNetwork(GraphBaseLayer):
    """Layer to sequentially compose Graph Network Blocks."""

    def __init__(self, graph_network_blocks: list, update_edges=True, update_nodes=True, update_global=True, **kwargs):
        """Instantiates the sequence of GN blocks.

        Args:
            graph_network_blocks (list): List of GraphNetwork blocks.
            update_edges (bool, optional): Whether to use updated edge features between blocks. Defaults to True.
            update_nodes (bool, optional):  Whether to use updated node features between blocks. Defaults to True.
            update_global (bool, optional):  Whether to use updated global features between blocks. Defaults to True.
        """
        super().__init__(**kwargs)
        self.graph_network_blocks = graph_network_blocks
        self.update_edges = update_edges
        self.update_nodes = update_nodes
        self.update_global = update_global

    def call(self, inputs, **kwargs):
        edges, nodes, globals_, edge_indices = inputs
        for block in self.graph_network_blocks:
            out = block([edges, nodes, globals_, edge_indices])
            edges_new, nodes_new, globals_new, _ = out
            if self.update_edges:
                edges = edges_new
            if self.update_nodes:
                nodes = nodes_new
            if self.update_global:
                globals_ = globals_new
        return edges, nodes, globals_, edge_indices


class NestedGraphNetwork(GraphNetwork):
    """Nested Graph Network layer with a nested Graph Network in the edge update function."""

    def __init__(self, edge_mlp, node_mlp, global_mlp, nested_gn,
                 aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
                 return_updated_edges=True, return_updated_nodes=True, return_updated_globals=True,
                 edge_attention_mlp_local=None, edge_attention_mlp_global=None, node_attention_mlp=None,
                 **kwargs):
        """Nested Graph Network layer with a nested Graph Network in the edge update function.

        See `GraphNetwork` for documentation of all arguments, except for the `nested_gn` argument.
        The `nested_gn` argument specifies the nested Graph Network.

        Args:
            ...
            nested_gn (SequentialGraphNetwork): Nested Graph Network which operates on the line graph level.
            ....
        """
        super().__init__(edge_mlp, node_mlp, global_mlp,
                aggregate_edges_local=aggregate_edges_local,
                aggregate_edges_global=aggregate_edges_global,
                aggregate_nodes=aggregate_nodes,
                return_updated_edges=return_updated_edges,
                return_updated_nodes=return_updated_nodes,
                return_updated_globals=return_updated_globals,
                edge_attention_mlp_local=edge_attention_mlp_local,
                edge_attention_mlp_global=edge_attention_mlp_global,
                node_attention_mlp=node_attention_mlp,
                **kwargs)
        self.nested_gn = nested_gn 

    def update_edges(self, edges, nodes_in, nodes_out, globals_, **kwargs):

        if self.edge_mlp is None:
            return edges

        edge_features = self.get_features(edges)
        features_to_concat = []
        if self.update_edges_input[0]:
            features_to_concat.append(edge_features)
        if self.update_edges_input[1]:
            nodes_in_features = self.get_features(nodes_in)
            features_to_concat.append(nodes_in_features)
        if self.update_edges_input[2]:
            nodes_out_features = self.get_features(nodes_out)
            features_to_concat.append(nodes_out_features)
        if self.update_edges_input[3]:
            global_features = self.get_features(globals_)
            features_to_concat.append(self.gather_state([global_features, edge_features]))
        
        concat_features = self.lazy_concat(features_to_concat)
        messages = self.edge_mlp(concat_features)

        assert isinstance(globals_, dict)
        assert 'line_graph_edge_indices' in globals_.keys()
        line_graph_edge_indices = globals_['line_graph_edge_indices']

        if 'line_graph_edges' in globals_.keys():
            line_graph_edges = self.get_features(globals_['line_graph_edges'])
            _, messages, _, _ = self.nested_gn([line_graph_edges, messages, None, line_graph_edge_indices])
        else:
            _, messages, _, _ = self.nested_gn([None, messages, None, line_graph_edge_indices])
        messages = self.get_features(messages)

        if self.edge_gate is not None:
            messages = tf.RaggedTensor.from_row_splits(
                    self.edge_gate(messages.values, edge_features.values)[1],
                    messages.row_splits)
        
        if self.residual_edge_update:
            messages = self.lazy_add([edge_features, messages])

        edges_new = self.update_features(edges, messages)
        
        if self.aggregate_edges_local_ in self.attention_strings:
            attention_local = self.edge_attention_mlp_local(messages)  
            edges_new = self.update_attribute(edges_new, 'attention_local', attention_local)
        elif self.aggregate_edges_global_ in self.attention_strings:
            attention_global = self.edge_attention_mlp_global(messages)  
            edges_new = self.update_attribute(edges_new, 'attention_global', attention_global)

        return edges_new

