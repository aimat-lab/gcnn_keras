from keras.layers import Add, Concatenate, Multiply, Subtract
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import Embedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.geom import NodePosition, EuclideanNorm, EdgeDirectionNormalized, PositionEncodingBasisLayer
from kgcnn.layers.pooling import PoolingNodes


def model_disjoint(
        inputs,
        use_node_embedding,
        use_edge_embedding,
        input_node_embedding: dict = None,
        input_edge_embedding: dict = None,
        depth: int = None,
        euclidean_norm_kwargs: dict = None,
        node_mlp_initialize: dict = None,
        use_edge_attributes: bool = None,
        edge_mlp_kwargs: dict = None,
        edge_attention_kwargs: dict = None,
        use_normalized_difference: bool = None,
        expand_distance_kwargs: dict = None,
        coord_mlp_kwargs: dict = None,
        pooling_coord_kwargs: dict = None,
        pooling_edge_kwargs: dict = None,
        node_normalize_kwargs: dict = None,
        use_node_attributes: bool = None,
        node_mlp_kwargs: dict = None,
        use_skip: bool = None,
        node_decoder_kwargs: dict = None,
        node_pooling_kwargs: dict = None,
        output_embedding: str = None,
        output_mlp: dict = None
):
    h0, x, ed, edi, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs
    # Make input

    # Embedding, if no feature dimension
    if use_node_embedding:
        h0 = Embedding(**input_node_embedding)(h0)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    # Model
    h = GraphMLP(**node_mlp_initialize)([h0, batch_id_node, count_nodes]) if node_mlp_initialize else h0
    for i in range(0, depth):
        pos1, pos2 = NodePosition()([x, edi])
        diff_x = Subtract()([pos1, pos2])
        norm_x = EuclideanNorm(**euclidean_norm_kwargs)(diff_x)
        # Original code has a normalize option for coord-differences.
        if use_normalized_difference:
            diff_x = EdgeDirectionNormalized()([pos1, pos2])
        if expand_distance_kwargs:
            norm_x = PositionEncodingBasisLayer()(norm_x)

        # Edge model
        h_i, h_j = GatherNodes([0, 1], concat_axis=None)([h, edi])
        if use_edge_attributes:
            m_ij = Concatenate()([h_i, h_j, norm_x, ed])
        else:
            m_ij = Concatenate()([h_i, h_j, norm_x])
        if edge_mlp_kwargs:
            m_ij = GraphMLP(**edge_mlp_kwargs)([m_ij, batch_id_edge, count_edges])
        if edge_attention_kwargs:
            m_att = GraphMLP(**edge_attention_kwargs)([m_ij, batch_id_edge, count_edges])
            m_ij = Multiply()([m_att, m_ij])

        # Coord model
        if coord_mlp_kwargs:
            m_ij_weights = GraphMLP(**coord_mlp_kwargs)([m_ij, batch_id_edge, count_edges])
            x_trans = Multiply()([m_ij_weights, diff_x])
            agg = AggregateLocalEdges(**pooling_coord_kwargs)([h, x_trans, edi])
            x = Add()([x, agg])

        # Node model
        m_i = AggregateLocalEdges(**pooling_edge_kwargs)([h, m_ij, edi])
        if node_mlp_kwargs:
            m_i = Concatenate()([h, m_i])
            if use_node_attributes:
                m_i = Concatenate()([m_i, h0])
            m_i = GraphMLP(**node_mlp_kwargs)([m_i, batch_id_node, count_nodes])
        if node_normalize_kwargs:
            h = GraphLayerNormalization(**node_normalize_kwargs)([h, batch_id_node, count_nodes])
        if use_skip:
            h = Add()([h, m_i])
        else:
            h = m_i

    # Output embedding choice
    if node_decoder_kwargs:
        n = GraphMLP(**node_mlp_kwargs)([h, batch_id_node, count_nodes])
    else:
        n = h

    # Final step.
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_kwargs)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `SchNet`")

    return out
