from typing import Union
from kgcnn.layers.casting import (
    CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
    CastDisjointToBatchedGraphState, CastDisjointToBatchedAttributes,
    CastBatchedGraphStateToDisjoint, CastRaggedAttributesToDisjoint,
    CastRaggedIndicesToDisjoint, CastDisjointToRaggedAttributes
)


def template_cast_output(model_outputs,
                         output_embedding, output_tensor_type, input_tensor_type, cast_disjoint_kwargs):
    out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = model_outputs

    # Output embedding choice
    if output_embedding == 'graph':
        out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        if output_tensor_type in ["padded", "masked"]:
            if "static_batched_node_output_shape" in cast_disjoint_kwargs:
                out_node_shape = cast_disjoint_kwargs["static_batched_node_output_shape"]
            else:
                out_node_shape = None
            out = CastDisjointToBatchedAttributes(static_output_shape=out_node_shape, **cast_disjoint_kwargs)(
                [out, batch_id_node, node_id, count_nodes])
        if output_tensor_type in ["ragged", "jagged"]:
            out = CastDisjointToRaggedAttributes()([out, batch_id_node, node_id, count_nodes])
        else:
            out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'edge':
        if output_tensor_type in ["padded", "masked"]:
            if "static_batched_edge_output_shape" in cast_disjoint_kwargs:
                out_edge_shape = cast_disjoint_kwargs["static_batched_edge_output_shape"]
            else:
                out_edge_shape = None
            out = CastDisjointToBatchedAttributes(static_output_shape=out_edge_shape, **cast_disjoint_kwargs)(
                [out, batch_id_edge, edge_id, count_edges])
        if output_tensor_type in ["ragged", "jagged"]:
            out = CastDisjointToRaggedAttributes()([out, batch_id_edge, edge_id, count_edges])
        else:
            out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)
    else:
        raise NotImplementedError()

    return out


def template_cast_list_input(model_inputs,
                             input_tensor_type,
                             cast_disjoint_kwargs,
                             has_nodes: Union[int, bool] = True,
                             has_edges: Union[int, bool] = True,
                             has_angles: Union[int, bool] = False,
                             has_edge_indices: Union[int, bool] = True,
                             has_angle_indices: Union[int, bool] = False,
                             has_graph_state: Union[int, bool] = False,
                             has_crystal_input: Union[int, bool] = False,
                             return_sub_id: bool = True):

    standard_inputs = [x for x in model_inputs]

    batched_nodes = []
    batched_edges = []
    batched_angles = []
    batched_state = []
    batched_indices = []
    batched_angle_indices = []
    batched_crystal_info = []

    for i in range(int(has_nodes)):
        batched_nodes.append(standard_inputs.pop(0))
    for i in range(int(has_edges)):
        batched_edges.append(standard_inputs.pop(0))
    for i in range(int(has_angles)):
        batched_angles.append(standard_inputs.pop(0))
    for i in range(int(has_edge_indices)):
        batched_indices.append(standard_inputs.pop(0))
    for i in range(int(has_angle_indices)):
        batched_angle_indices.append(standard_inputs.pop(0))
    for i in range(int(has_graph_state)):
        batched_state.append(standard_inputs.pop(0))
    for i in range(int(has_crystal_input)):
        batched_crystal_info.append(standard_inputs.pop(0))

    batched_id = standard_inputs

    disjoint_nodes = []
    disjoint_edges = []
    disjoint_state = []
    disjoint_angles = []
    disjoint_indices = []
    disjoint_angle_indices = []
    disjoint_crystal_info = []
    disjoint_id = []

    if input_tensor_type in ["padded", "masked"]:

        if int(has_angle_indices) > 0:
            part_nodes, part_edges, part_angle = batched_id
        else:
            part_nodes, part_edges = batched_id
            part_angle = None

        for x in batched_indices:
            _, idx, batch_id_node, batch_id_edge, node_id, edge_id, len_nodes, len_edges = CastBatchedIndicesToDisjoint(
                **cast_disjoint_kwargs)([batched_nodes[0], x, part_nodes, part_edges])
            disjoint_indices.append(idx)

        for x in batched_angle_indices:
            _, idx, _, batch_id_ang, _, ang_id, _, len_ang = CastBatchedIndicesToDisjoint(
                **cast_disjoint_kwargs)([batched_indices[0], x, part_edges, part_angle])
            disjoint_angle_indices.append(idx)

        for x in batched_nodes:
            disjoint_nodes.append(
                CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([x, part_nodes])[0])

        for x in batched_edges:
            disjoint_edges.append(
                CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([x, part_edges])[0])

        for x in batched_angles:
            disjoint_angles.append(
                CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([x, part_angle])[0])

        for x in batched_state:
            disjoint_state.append(
                CastBatchedGraphStateToDisjoint(**cast_disjoint_kwargs)(x))

        if has_crystal_input > 0:
            disjoint_crystal_info.append(
                CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_crystal_info[0], part_edges])[0]
            )
            disjoint_crystal_info.append(
                CastBatchedGraphStateToDisjoint(**cast_disjoint_kwargs)(batched_crystal_info[1])
            )

    elif input_tensor_type in ["ragged", "jagged"]:

        for x in batched_indices:
            _, idx, batch_id_node, batch_id_edge, node_id, edge_id, len_nodes, len_edges = CastRaggedIndicesToDisjoint(
                **cast_disjoint_kwargs)([batched_nodes[0], x])
            disjoint_indices.append(idx)

        for x in batched_angle_indices:
            _, idx, _, batch_id_ang, _, ang_id, _, len_ang = CastRaggedIndicesToDisjoint(
                **cast_disjoint_kwargs)([batched_indices[0], x])
            disjoint_angle_indices.append(idx)

        for x in batched_nodes:
            disjoint_nodes.append(
                CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(x)[0])

        for x in batched_edges:
            disjoint_edges.append(
                CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(x)[0])

        for x in batched_angles:
            disjoint_angles.append(
                CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(x)[0])

        if has_crystal_input > 0:
            disjoint_crystal_info.append(
                CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(batched_crystal_info[0])[0]
            )
            disjoint_crystal_info.append(
                batched_crystal_info[1]
            )

        disjoint_state = batched_state

    else:
        disjoint_nodes = batched_nodes
        disjoint_edges = batched_edges
        disjoint_indices = batched_indices
        disjoint_state = batched_state
        disjoint_angle_indices = batched_angle_indices
        disjoint_angles = batched_angles
        disjoint_crystal_info = batched_crystal_info

    if input_tensor_type in ["ragged", "jagged", "padded", "masked"]:
        disjoint_id.append(batch_id_node)  # noqa
        disjoint_id.append(batch_id_edge)  # noqa
        if int(has_angle_indices) > 0:
            disjoint_id.append(batch_id_ang)  # noqa
        if return_sub_id:
            disjoint_id.append(node_id)  # noqa
            disjoint_id.append(edge_id)  # noqa
            if int(has_angle_indices) > 0:
                disjoint_id.append(ang_id)  # noqa
        disjoint_id.append(len_nodes)  # noqa
        disjoint_id.append(len_edges)  # noqa
        if int(has_angle_indices) > 0:
            disjoint_id.append(len_ang)  # noqa
    else:
        disjoint_id = batched_id

    disjoint_model_inputs = disjoint_nodes + disjoint_edges + disjoint_angles + disjoint_indices + disjoint_angle_indices + disjoint_state + disjoint_crystal_info  + disjoint_id

    return disjoint_model_inputs

