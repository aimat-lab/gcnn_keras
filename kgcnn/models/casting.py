from kgcnn.layers.casting import (
    CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
    CastDisjointToBatchedGraphState, CastDisjointToBatchedAttributes,
    CastBatchedGraphStateToDisjoint, CastRaggedAttributesToDisjoint,
    CastRaggedIndicesToDisjoint, CastDisjointToRaggedAttributes
)


def template_cast_output(model_outputs,
                         output_embedding, output_tensor_type, input_tensor_type, cast_disjoint_kwargs,
                         batched_model_output = None):

    out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = model_outputs
    batched_nodes = batched_model_output[0] if batched_model_output is not None else None

    # Output embedding choice
    if output_embedding == 'graph':
        out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        if output_tensor_type in ["padded", "masked"]:
            if input_tensor_type in ["padded", "masked"]:
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

    return out


def template_cast_input(model_inputs,
                        input_tensor_type,
                        cast_disjoint_kwargs,
                        has_edges: bool = True):

    if input_tensor_type in ["padded", "masked"]:
        batched_nodes, batched_edges, batched_indices, total_nodes, total_edges = model_inputs
        n, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
            **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])
        ed, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_edges, total_edges])
    elif input_tensor_type in ["ragged", "jagged"]:
        batched_nodes, batched_edges, batched_indices = model_inputs
        n, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastRaggedIndicesToDisjoint(
            **cast_disjoint_kwargs)([batched_nodes, batched_indices])
        ed, _, _, _ = CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(batched_edges)
    else:
        n, ed, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = model_inputs

    disjoint_model_inputs = n, ed, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges

    return disjoint_model_inputs