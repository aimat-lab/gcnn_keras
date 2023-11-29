from keras.layers import Concatenate
from kgcnn.layers.aggr import AggregateLocalEdgesLSTM, AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.pooling import PoolingNodes


def model_disjoint(
        inputs,
        use_node_embedding: bool = None,
        use_edge_embedding: bool = None,
        input_node_embedding: dict = None,
        input_edge_embedding: dict = None,
        node_mlp_args: dict = None,
        edge_mlp_args: dict = None,
        pooling_args: dict = None,
        pooling_nodes_args: dict = None,
        gather_args: dict = None,
        concat_args: dict = None,
        use_edge_features: bool = None,
        depth: int = None,
        output_embedding: str = None,
        output_mlp: dict = None,
):
    n, ed, disjoint_indices, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    for i in range(0, depth):

        eu = GatherNodesOutgoing(**gather_args)([n, disjoint_indices])
        if use_edge_features:
            eu = Concatenate(**concat_args)([eu, ed])

        eu = GraphMLP(**edge_mlp_args)([eu, batch_id_edge, count_edges])

        # Pool message
        if pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            nu = AggregateLocalEdgesLSTM(**pooling_args)([n, eu, disjoint_indices])
        else:
            nu = AggregateLocalEdges(**pooling_args)([n, eu, disjoint_indices])  # Summing for each node connection

        nu = Concatenate(**concat_args)([n, nu])  # Concatenate node features with new edge updates

        n = GraphMLP(**node_mlp_args)([nu, batch_id_node, count_nodes])

        n = GraphLayerNormalization()([n, batch_id_node, count_nodes])

    # Regression layer on output
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for `GraphSAGE`")
    return out