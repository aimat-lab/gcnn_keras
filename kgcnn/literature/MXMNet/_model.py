from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, BesselBasisLayer, EdgeAngle, SphericalBasisLayer
from keras.layers import Concatenate, Subtract, Add
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.literature.DimeNetPP._layers import EmbeddingDimeBlock
from ._layers import MXMGlobalMP, MXMLocalMP


def model_disjoint(
    inputs,
    use_node_embedding,
    use_edge_embedding,
    input_node_embedding=None,
    input_edge_embedding=None,
    bessel_basis_local=None,
    spherical_basis_local=None,
    bessel_basis_global=None,
    use_edge_attributes=None,
    mlp_rbf_kwargs=None,
    mlp_sbf_kwargs=None,
    depth=None,
    global_mp_kwargs=None,
    local_mp_kwargs=None,
    node_pooling_args=None,
    output_embedding=None,
    use_output_mlp=None,
    output_mlp=None
):
    # Make input
    n, x, ed, ei_l, ri_g, ai_1, ai_2 = inputs[:7]
    batch_id_node, batch_id_edge, batch_id_ranges, batch_id_angles_1, batch_id_angles_2 = inputs[7:12]
    node_id, edge_id, range_id, angle_id1, angle_id2 = inputs[12:17]
    count_nodes, count_edges, count_ranges, count_angles1, count_angles2 = inputs[17:]

    # Rename to short names and make embedding, if no feature dimension.
    if use_node_embedding:
        n = EmbeddingDimeBlock(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = EmbeddingDimeBlock(**input_edge_embedding)(ed)

    # Calculate distances and spherical and bessel basis for local edges including angles.
    # For the first version, we restrict ourselves to 2-hop angles.
    pos1_l, pos2_l = NodePosition()([x, ei_l])
    d_l = NodeDistanceEuclidean()([pos1_l, pos2_l])
    rbf_l = BesselBasisLayer(**bessel_basis_local)(d_l)
    v12_l = Subtract()([pos1_l, pos2_l])
    a_l_1 = EdgeAngle()([v12_l, ai_1])
    a_l_2 = EdgeAngle(vector_scale=[1.0, -1.0])([v12_l, ai_2])
    sbf_l_1 = SphericalBasisLayer(**spherical_basis_local)([d_l, a_l_1, ai_1])
    sbf_l_2 = SphericalBasisLayer(**spherical_basis_local)([d_l, a_l_2, ai_2])

    # Calculate distance and bessel basis for global (range) edges.
    pos1_g, pos2_g = NodePosition()([x, ri_g])
    d_g = NodeDistanceEuclidean()([pos1_g, pos2_g])
    rbf_g = BesselBasisLayer(**bessel_basis_global)(d_g)

    if use_edge_attributes:
        rbf_l = Concatenate()([rbf_l, ed])

    rbf_l = GraphMLP(**mlp_rbf_kwargs)([rbf_l, batch_id_edge, count_edges])
    sbf_l_1 = GraphMLP(**mlp_sbf_kwargs)([sbf_l_1, batch_id_angles_1, count_angles1])
    sbf_l_2 = GraphMLP(**mlp_sbf_kwargs)([sbf_l_2, batch_id_angles_2, count_angles2])
    rbf_g = GraphMLP(**mlp_rbf_kwargs)([rbf_g, batch_id_ranges, count_ranges])

    # Model
    h = n
    nodes_list = []
    for i in range(0, depth):
        h = MXMGlobalMP(**global_mp_kwargs)([h, rbf_g, ri_g])
        h, t = MXMLocalMP(**local_mp_kwargs)([h, rbf_l, sbf_l_1, sbf_l_2, ei_l, ai_1, ai_2])
        nodes_list.append(t)

    # Output embedding choice
    out = Add()(nodes_list)
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)([count_nodes, out, batch_id_node])
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `MXMNet`")

    return out
