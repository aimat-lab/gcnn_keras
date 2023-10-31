from kgcnn.layers.casting import (
    CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
    CastDisjointToBatchedGraphState, CastDisjointToBatchedAttributes,
    CastBatchedGraphStateToDisjoint, CastRaggedAttributesToDisjoint,
    CastRaggedIndicesToDisjoint, CastDisjointToRaggedAttributes
)


def template_cast_output(model_outputs, output_embedding, output_tensor_type, input_tensor_type, cast_disjoint_kwargs):

    if len(model_outputs) == 4:
        out, batch_id_node, node_id, count_nodes = model_outputs
    else:
        batched_nodes, out, batch_id_node, node_id, count_nodes = model_outputs

    # Output embedding choice
    if output_embedding == 'graph':
        out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        if output_tensor_type in ["padded", "masked"]:
            if input_tensor_type in ["padded", "masked"]:
                out = CastDisjointToBatchedAttributes(**cast_disjoint_kwargs)(
                    [batched_nodes, out, batch_id_node, node_id, count_nodes])  # noqa
            else:
                out = CastDisjointToBatchedAttributes(**cast_disjoint_kwargs)(
                    [out, batch_id_node, node_id, count_nodes])
        if output_tensor_type in ["ragged", "jagged"]:
            out = CastDisjointToRaggedAttributes()([out, batch_id_node, node_id, count_nodes])
        else:
            out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)

    return out
