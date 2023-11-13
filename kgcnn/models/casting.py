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


def template_cast_input(model_inputs,
                        input_tensor_type,
                        cast_disjoint_kwargs,
                        has_nodes: Union[int, bool] = True,
                        has_edges: Union[int, bool] = True,
                        has_graph_state: Union[int, bool] = False):

    standard_inputs = [x for x in model_inputs]
    batched_nodes = []
    batched_edges = []
    batched_state = []

    for i in range(int(has_nodes)):
        batched_nodes.append(standard_inputs.pop(0))
    for i in range(int(has_edges)):
        batched_edges.append(standard_inputs.pop(0))
    for i in range(int(has_graph_state)):
        batched_state.append(standard_inputs.pop(0))

    batched_indices = standard_inputs.pop(0)

    batched_id = standard_inputs

    disjoint_nodes = []
    disjoint_edges = []
    disjoint_state = []

    if input_tensor_type in ["padded", "masked"]:
        part_nodes, part_edges = batched_id

        n, idx, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
            **cast_disjoint_kwargs)([batched_nodes.pop(0), batched_indices, part_nodes, part_edges])
        disjoint_indices = [idx]
        disjoint_nodes.append(n)
        disjoint_id = [batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges]

        for x in batched_nodes:
            disjoint_nodes.append(
                CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([x, part_nodes])[0])
        for x in batched_edges:
            disjoint_edges.append(
                CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([x, part_edges])[0])
        for x in batched_state:
            disjoint_state.append(
                CastBatchedGraphStateToDisjoint(**cast_disjoint_kwargs)(x))

    elif input_tensor_type in ["ragged", "jagged"]:

        n, idx, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastRaggedIndicesToDisjoint(
            **cast_disjoint_kwargs)([batched_nodes.pop(0), batched_indices])
        disjoint_indices = [idx]
        disjoint_nodes.append(n)
        disjoint_id = [batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges]

        for x in batched_nodes:
            disjoint_nodes.append(
                CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(x)[0])
        for x in batched_edges:
            disjoint_edges.append(
                CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(x)[0])
        disjoint_state = batched_state

    else:
        disjoint_nodes = batched_nodes
        disjoint_edges = batched_edges
        disjoint_indices = [batched_indices]
        disjoint_state = batched_state
        disjoint_id = batched_id

    disjoint_model_inputs = disjoint_nodes + disjoint_edges + disjoint_state + disjoint_indices  + disjoint_id

    return disjoint_model_inputs
