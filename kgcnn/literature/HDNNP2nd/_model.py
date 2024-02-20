from kgcnn.layers.mlp import MLP, GraphMLP, RelationalMLP
from keras.layers import Concatenate, Dense, Multiply, Add
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.norm import GraphBatchNormalization
from ._wacsf import wACSFRad, wACSFAng
from ._layers import CorrectPartialCharges
from ._acsf import ACSFG2, ACSFG4, ACSFConstNormalization


def model_disjoint_weighted(
        inputs,
        node_pooling_args: dict = None,
        w_acsf_ang_kwargs: dict = None,
        w_acsf_rad_kwargs: dict = None,
        normalize_kwargs: dict = None,
        const_normalize_kwargs: dict = None,
        mlp_kwargs: dict = None,
        output_embedding: str = None,
        use_output_mlp: bool = None,
        output_mlp: dict = None,
        predict_dipole: bool = None
):
    # Make input
    node_input, xyz_input, edge_index_input, angle_index_input, tot_charge, batch_id_node, count_nodes = inputs

    # ACSF representation.
    rep_rad = wACSFRad(**w_acsf_rad_kwargs)([node_input, xyz_input, edge_index_input])
    rep_ang = wACSFAng(**w_acsf_ang_kwargs)([node_input, xyz_input, angle_index_input])
    rep = Concatenate()([rep_rad, rep_ang])

    # Normalization
    if normalize_kwargs:
        rep = GraphBatchNormalization(**normalize_kwargs)([rep, batch_id_node, count_nodes])
    if const_normalize_kwargs:
        rep = ACSFConstNormalization(**const_normalize_kwargs)(rep)

    # learnable NN.
    n = RelationalMLP(**mlp_kwargs)([rep, node_input, batch_id_node, count_nodes])

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
        if predict_dipole:
            pc = Dense(units=1)(n)
            tc = PoolingNodes(pooling_method="sum")([count_nodes, pc, batch_id_node])
            if tot_charge is not None:
                pc_correct = CorrectPartialCharges()([tc, tot_charge, count_nodes, batch_id_node])
                pc = Add()([pc, pc_correct])
            p_dip = Multiply()([pc, xyz_input])
            dip = PoolingNodes(pooling_method="sum")([count_nodes, p_dip, batch_id_node])
            out = [out, dip]
            if tot_charge is None:
                out = out + [tc]
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `HDNNP2nd` .")

    return out


def model_disjoint_behler(
        inputs,
        node_pooling_args: dict = None,
        normalize_kwargs: dict = None,
        const_normalize_kwargs: dict = None,
        g2_kwargs: dict = None,
        g4_kwargs: dict = None,
        mlp_kwargs: dict = None,
        output_embedding: str = None,
        use_output_mlp: bool = None,
        output_mlp: dict = None,
        predict_dipole: bool = None
):
    # Make input
    node_input, xyz_input, edge_index_input, angle_index_input, tot_charge, batch_id_node, count_nodes = inputs

    # ACSF representation.
    rep_g2 = ACSFG2(**ACSFG2.make_param_table(**g2_kwargs))([node_input, xyz_input, edge_index_input])
    rep_g4 = ACSFG4(**ACSFG4.make_param_table(**g4_kwargs))([node_input, xyz_input, angle_index_input])
    rep = Concatenate()([rep_g2, rep_g4])

    # Normalization
    if normalize_kwargs:
        rep = GraphBatchNormalization(**normalize_kwargs)([rep, batch_id_node, count_nodes])
    if const_normalize_kwargs:
        rep = ACSFConstNormalization(**const_normalize_kwargs)(rep)

    # learnable NN.
    n = RelationalMLP(**mlp_kwargs)([rep, node_input, batch_id_node, count_nodes])

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
        if predict_dipole:
            pc = Dense(units=1)(n)
            tc = PoolingNodes(pooling_method="sum")([count_nodes, pc, batch_id_node])
            if tot_charge is not None:
                pc_correct = CorrectPartialCharges()([tc, tot_charge, count_nodes, batch_id_node])
                pc = Add()([pc, pc_correct])
            p_dip = Multiply()([pc, xyz_input])
            dip = PoolingNodes(pooling_method="sum")([count_nodes, p_dip, batch_id_node])
            out = [out, dip]
            if tot_charge is None:
                out = out + [tc]
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `HDNNP2nd`")

    return out


def model_disjoint_atom_wise(
        inputs,
        node_pooling_args: dict = None,
        mlp_kwargs: dict = None,
        output_embedding: str = None,
        use_output_mlp: bool = None,
        output_mlp: dict = None,
        predict_dipole: bool = None
):
    # Make input
    node_input, rep_input, tot_charge, batch_id_node, count_nodes = inputs

    # learnable NN.
    n = RelationalMLP(**mlp_kwargs)([rep_input, node_input, batch_id_node, count_nodes])

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
        if predict_dipole:
            pc = Dense(units=1)(n)
            tc = PoolingNodes(pooling_method="sum")([count_nodes, pc, batch_id_node])
            if tot_charge is not None:
                pc_correct = CorrectPartialCharges()([tc, tot_charge, count_nodes, batch_id_node])
                pc = Add()([pc, pc_correct])
            p_dip = Multiply()([pc, xyz_input])
            dip = PoolingNodes(pooling_method="sum")([count_nodes, p_dip, batch_id_node])
            out = [out, dip]
            if tot_charge is None:
                out = out + [tc]
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `HDNNP2nd`")

    return out
