from typing import Union
from kgcnn.layers.casting import (
    CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
    CastDisjointToBatchedGraphState, CastDisjointToBatchedAttributes,
    CastBatchedGraphStateToDisjoint, CastRaggedAttributesToDisjoint,
    CastRaggedIndicesToDisjoint, CastDisjointToRaggedAttributes
)


def template_cast_output(model_outputs,
                         output_embedding, output_tensor_type, input_tensor_type, cast_disjoint_kwargs):
    """TODO"""

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
    r"""Template of listed graph input tensors, which should be compatible to previous kgcnn versions and
    defines the order as follows: :obj:`[nodes, edges, angles, edge_indices, angle_indices, graph_state, ...]` .
    Where '...' denotes further mask or ID tensors, which is required for certain input types (see below).
    Depending on the model, some inputs may not be used (see model description for information on supported inputs).
    For example if the model does not support angles and no graph attribute input, the input becomes:
    :obj:`[nodes, edges, edge_indices, ...]` .
    In case of crystal graphs lattice and translation information has to be added. This will give a possible input of
    :obj:`[nodes, edges, angles, edge_indices, angle_indices, graph_state, image_translation, lattice,...]` .
    Note that in place of nodes or edges also more than one tensor can be provided, depending on the model, for example
    :obj:`[nodes_1, nodes_2, edges_1, edges_2, edge_indices, ...]` .
    However, for future models we intend to used named inputs rather than a list that is sensible to ordering.

    Padded or Masked Inputs:
        list: :obj:`[nodes, edges, angles, edge_indices, angle_indices, graph_state, image_translation, lattice,
        node_mask/node_count, edge_mask/edge_count, angle_mask/angle_count]`

            - nodes (Tensor): Node attributes of shape `(batch, N, F)` or `(batch, N)`
              using an embedding layer.
            - edges (Tensor): Edge attributes of shape `(batch, M, F)` or `(batch, M)`
              using an embedding layer.
            - angles (Tensor): Angle attributes of shape `(batch, M, F)` or `(batch, K)`
              using an embedding layer.
            - edge_indices (Tensor): Index list for edges of shape `(batch, M, 2)` referring to nodes.
            - angle_indices (Tensor): Index list for angles of shape `(batch, K, 2)` referring to edges.
            - graph_state (Tensor): Graph attributes of shape `(batch, F)` .
            - image_translation (Tensor): Indices of the periodic image the sending node is located in.
            Shape is `(batch, M, 3)` .
            - lattice (Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)` .
            - node_mask (Tensor): Mask for padded nodes of shape `(batch, N)` .
            - edge_mask (Tensor): Mask for padded edges of shape `(batch, M)` .
            - angle_mask (Tensor): Mask for padded angles of shape `(batch, K)` .
            - node_count (Tensor): Total number of nodes if padding is used of shape `(batch, )` .
            - edge_count (Tensor): Total number of edges if padding is used of shape `(batch, )` .
            - angle_count (Tensor): Total number of angle if padding is used of shape `(batch, )` .

    Ragged or Jagged Inputs:
        list: :obj:`[nodes, edges, angles, edge_indices, angle_indices, graph_state, image_translation, lattice]`

            - nodes (RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edges (RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - angles (RaggedTensor): Angle attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (RaggedTensor): Index list for edges of shape `(batch, None, 2)` referring to nodes.
            - angle_indices (RaggedTensor): Index list for angles of shape `(batch, None, 2)` referring to edges.
            - graph_state (Tensor): Graph attributes of shape `(batch, F)` .
            - image_translation (RaggedTensor): Indices of the periodic image the sending node is located in.
            Shape is `(batch, None, 3)` .
            - lattice (Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)` .

    Disjoint Input:
        list: :obj:`[nodes, edges, angles, edge_indices, angle_indices, graph_state, image_translation, lattice,
        graph_id_node, graph_id_edge, graph_id_angle, nodes_id, edges_id, angle_id, nodes_count, edges_count,
        angles_count]`

            - nodes (Tensor): Node attributes of shape `([N], F)` or `([N], )` using an embedding layer.
            - edges (Tensor): Edge attributes of shape `([M], F)` or `([M], )` using an embedding layer.
            - angles (Tensor): Angle attributes of shape `([K], F)` or `([K], )` using an embedding layer.
            - edge_indices (Tensor): Index list for edges of shape `(2, [M])` referring to nodes.
            - angle_indices (Tensor): Index list for angles of shape `(2, [K])` referring to edges.
            - graph_state (Tensor): Graph attributes of shape `(batch, F)` .
            - image_translation (Tensor): Indices of the periodic image the sending node is located in.
            Shape is `([M], 3)` .
            - lattice (Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)` .
            - graph_id_node (Tensor): ID tensor of graph assignment in disjoint graph of shape `([N], )` .
            - graph_id_edge (Tensor): ID tensor of graph assignment in disjoint graph of shape `([M], )` .
            - graph_id_angle (Tensor): ID tensor of graph assignment in disjoint graph of shape `([K], )` .
            - nodes_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
            - edges_id (Tensor): The ID-tensor to assign each edge to its respective graph of shape `([M], )` .
            - angle_id (Tensor): The ID-tensor to assign each edge to its respective graph of shape `([K], )` .
            - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
            - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .
            - angles_count (Tensor): Tensor of number of angles for each graph of shape `(batch, )` .
    """
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

