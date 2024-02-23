from typing import Union
from kgcnn.layers.casting import (
    CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
    CastDisjointToBatchedGraphState, CastDisjointToBatchedAttributes,
    CastBatchedGraphStateToDisjoint, CastRaggedAttributesToDisjoint,
    CastRaggedIndicesToDisjoint, CastDisjointToRaggedAttributes
)


template_cast_output_docs = r"""
    The standard model output template returns a single tensor of either "graph", "node", or "edge"
    embeddings specified by :obj:`output_embedding` within the model.
    The return tensor type is determined by :obj:`output_tensor_type` . Options are:

    graph:
        Tensor: Graph labels of either type
        
            - padded (Tensor): Batched tensor of graphs of shape `(batch, F)` .
            - disjoint (Tensor): Tensor with potential disjoint padded graphs of shape `(batch, F)` .
            - ragged (Tensor): Tensor of shape `(batch, F)` . No ragged dimension is needed here.

    nodes:
        Tensor: Node labels for the graph of either type:

            - ragged (RaggedTensor): Single tensor of shape `(batch, None, F)` .
            - padded (Tensor): Padded tensor of shape `(batch, N, F)` .
            - disjoint (Tensor): Disjoint representation of shape `([N], F)` .

    edges:
        Tensor: Edge labels for the graph of either type:

            - ragged (RaggedTensor): Single tensor of shape `(batch, None, F)` .
            - padded (Tensor): Padded tensor of shape `(batch, M, F)`
            - disjoint (Tensor): Disjoint representation of shape `([M], F)` .
"""


def template_cast_output(model_outputs,
                         output_embedding,
                         output_tensor_type,
                         input_tensor_type,
                         cast_disjoint_kwargs):
    """Template to cast graph, node or edge output to the desired tensor representation.

    Args:
        model_outputs (list): List of output and additional ID tensors. The list must always be
            [model_output, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges]
            but can have None in place of ID tensors if not required.
        output_embedding (str): Embedding of the graph output, either "graph", "node" or "edge".
        output_tensor_type (str): The tensor representation of model output.
        input_tensor_type (str): The tensor representation of model input.
        cast_disjoint_kwargs (dict): Kwargs for casting layers.

    Returns:
        Tensor: Keras output tensor.
    """
    out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = model_outputs

    # Output embedding choice
    if output_embedding == 'graph':
        if output_tensor_type in ["padded", "masked"]:
            # Here we could also modify the behaviour for direct disjoint without removing the padding via
            # remove_padded_disjoint_from_batched_output
            out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)
        elif output_tensor_type in ["ragged", "jagged"]:
            pass
        else:
            pass
    elif output_embedding == 'node':
        if output_tensor_type in ["padded", "masked"]:
            if "static_batched_node_output_shape" in cast_disjoint_kwargs:
                out_node_shape = cast_disjoint_kwargs["static_batched_node_output_shape"]
            else:
                out_node_shape = None
            out = CastDisjointToBatchedAttributes(static_output_shape=out_node_shape, **cast_disjoint_kwargs)(
                [out, batch_id_node, node_id, count_nodes])
        elif output_tensor_type in ["ragged", "jagged"]:
            out = CastDisjointToRaggedAttributes()([out, batch_id_node, node_id, count_nodes])
        else:
            pass
    elif output_embedding == 'edge':
        if output_tensor_type in ["padded", "masked"]:
            if "static_batched_edge_output_shape" in cast_disjoint_kwargs:
                out_edge_shape = cast_disjoint_kwargs["static_batched_edge_output_shape"]
            else:
                out_edge_shape = None
            out = CastDisjointToBatchedAttributes(static_output_shape=out_edge_shape, **cast_disjoint_kwargs)(
                [out, batch_id_edge, edge_id, count_edges])
        elif output_tensor_type in ["ragged", "jagged"]:
            out = CastDisjointToRaggedAttributes()([out, batch_id_edge, edge_id, count_edges])
        else:
            pass
    else:
        raise NotImplementedError("Unknown output embedding choice.")

    return out


template_cast_list_input_docs = r"""
    Template of listed graph input tensors, which should be compatible to previous kgcnn versions and
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
    Whether to use mask or length tensor for padded as well as further parameter of casting has to be set with
    (dict) :obj:`cast_disjoint_kwargs` .

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
            - graph_id_node (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
            - graph_id_edge (Tensor): ID tensor of batch assignment in disjoint graph of shape `([M], )` .
            - graph_id_angle (Tensor): ID tensor of batch assignment in disjoint graph of shape `([K], )` .
            - nodes_id (Tensor): The ID-tensor to assign each node to its respective graph of shape `([N], )` .
            - edges_id (Tensor): The ID-tensor to assign each edge to its respective graph of shape `([M], )` .
            - angle_id (Tensor): The ID-tensor to assign each edge to its respective graph of shape `([K], )` .
            - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
            - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .
            - angles_count (Tensor): Tensor of number of angles for each graph of shape `(batch, )` .
"""


def template_cast_list_input(model_inputs: list,
                             input_tensor_type: str,
                             cast_disjoint_kwargs: dict,
                             mask_assignment: list = None,
                             index_assignment: list = None,
                             return_sub_id: bool = True):
    r"""Template to cast a list of model inputs to a list of disjoint tensors. The number of model inputs can be
    variable. The system is preset and explained by :obj:`template_cast_list_input_docs` .

    The ID information from the mask in tensor form is appended to the returned list.
    Ragged Tensor do not require a mask but use the mask information to generate joint ID tensors.

    Args:
        model_inputs (list): List of Keras inputs.
        input_tensor_type (str): Input tensor type. Either "padded", "ragged" or "disjoint".
        cast_disjoint_kwargs (dict): Kwargs for casting layers.
        mask_assignment (list): List that assigns Tensors to their mask.
            Inputs that do not require a mask must be marked by Nones. Different inputs can use the same mask.
        index_assignment (list): List that assigns index Tensors to their target to which the index refer to.
            Inputs that are not indices must be marked by Nones.
        return_sub_id (bool): Whether the returned list contains the sub-graph ID tensors.

    Returns:
        list: List of Keras Tensors for disjoint model.
    """
    out_tensor = []
    out_batch_id = []
    out_graph_id = []
    out_totals = []
    is_already_disjoint = False

    if input_tensor_type in ["padded", "masked"]:
        if mask_assignment is None or not isinstance(mask_assignment, (list, tuple)):
            raise ValueError("Mask assignment information is required or invalid.")

        reduced_mask = [x for x in mask_assignment if x is not None]
        if len(reduced_mask) == 0:
            num_mask = 0
        else:
            num_mask = max(reduced_mask) + 1
        if len(mask_assignment) + num_mask != len(model_inputs):
            raise ValueError()

        values_input = model_inputs[:-num_mask]
        mask_input = model_inputs[-num_mask:]

        if index_assignment is None:
            index_assignment = [None for _ in range(len(values_input))]
        if len(index_assignment) != len(mask_assignment):
            raise ValueError("Number of provided mask tensors does not match template specification.")

        out_tensor = [None for _ in range(len(values_input))]
        out_batch_id = [None for _ in range(num_mask)]
        out_graph_id = [None for _ in range(num_mask)]
        out_totals = [None for _ in range(num_mask)]

        for i, i_ref in enumerate(index_assignment):
            if i_ref is None:
                continue
            assert isinstance(i_ref, int), "Must provide positional index of for reference of indices."
            ref = values_input[i_ref]
            x = values_input[i]
            m, m_ref = mask_assignment[i], mask_assignment[i_ref]
            ref_mask = mask_input[m_ref]
            x_mask = mask_input[m]
            o_ref, o_x, b_r, b_x, g_r, g_x, t_r, t_x = CastBatchedIndicesToDisjoint(
                **cast_disjoint_kwargs)([ref, x, ref_mask, x_mask])
            out_tensor[i] = o_x
            # Important to not overwrite indices with simple values here!
            if out_tensor[i_ref] is None:
                out_tensor[i_ref] = o_ref
            out_batch_id[m] = b_x
            out_batch_id[m_ref] = b_r
            out_graph_id[m] = g_x
            out_graph_id[m_ref] = g_r
            out_totals[m] = t_x
            out_totals[m_ref] = t_r

        for i, x in enumerate(values_input):
            if out_tensor[i] is not None:
                continue
            m = mask_assignment[i]
            if m is None:
                out_tensor[i] = CastBatchedGraphStateToDisjoint(**cast_disjoint_kwargs)(x)
                continue

            x_mask = mask_input[m]
            o_x, bi, gi, tot = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([x, x_mask])
            out_tensor[i] = o_x
            out_batch_id[m] = bi
            out_graph_id[m] = gi
            out_totals[m] = tot

    elif input_tensor_type in ["ragged", "jagged"]:
        if index_assignment is None:
            index_assignment = [None for _ in range(len(model_inputs))]
        if len(index_assignment) != len(model_inputs):
            raise ValueError()

        reduced_mask = [x for x in mask_assignment if x is not None]
        if len(reduced_mask) == 0:
            num_mask = 0
        else:
            num_mask = max(reduced_mask) + 1

        out_tensor = [None for _ in range(len(model_inputs))]
        out_batch_id = [None for _ in range(num_mask)]
        out_graph_id = [None for _ in range(num_mask)]
        out_totals = [None for _ in range(num_mask)]

        for i, i_ref in enumerate(index_assignment):
            if i_ref is None:
                continue
            assert isinstance(i_ref, int), "Must provide positional index of for reference of indices."
            ref = model_inputs[i_ref]
            x = model_inputs[i]
            m, m_ref = mask_assignment[i], mask_assignment[i_ref]
            o_ref, o_x, b_r, b_x, g_r, g_x, t_r, t_x = CastRaggedIndicesToDisjoint(
                **cast_disjoint_kwargs)([ref, x])
            out_tensor[i] = o_x
            # Important to no overwrite indices with simple values here! This is the case if indices reference indices.
            if out_tensor[i_ref] is None:
                out_tensor[i_ref] = o_ref
            out_batch_id[m] = b_x
            out_batch_id[m_ref] = b_r
            out_graph_id[m] = g_x
            out_graph_id[m_ref] = g_r
            out_totals[m] = t_x
            out_totals[m_ref] = t_r

        for i, x in enumerate(model_inputs):
            if out_tensor[i] is not None:
                continue
            m = mask_assignment[i]
            if m is not None:
                o_x, bi, gi, tot = CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(x)
                out_tensor[i] = o_x
                out_batch_id[m] = bi
                out_graph_id[m] = gi
                out_totals[m] = tot
            else:
                out_tensor[i] = x

    else:
        is_already_disjoint = True

    if is_already_disjoint:
        out = model_inputs
    else:
        out = out_tensor + out_batch_id
        if return_sub_id:
            out = out + out_graph_id
        out = out + out_totals

    return out


def template_cast_dict_input(model_inputs: dict,
                             input_tensor_type: str,
                             cast_disjoint_kwargs: dict,
                             mask_assignment: dict = None,
                             index_assignment: dict = None,
                             return_sub_id: bool = True,
                             rename_mask_to_id: dict = None):
    """Template to cast a dictionary of model inputs to a dict of disjoint tensors. The number of model inputs can be
    variable. The system is rather flexible and explained by :obj:`template_cast_dict_input_docs` .

    The ID information from the mask in tensor form is appended to the returned dictionary.
    Ragged Tensor do not require a mask but use the mask information to generate joint ID tensors.

    Args:
        model_inputs (dict): Dictionary of Keras inputs.
        input_tensor_type (str): Input tensor type. Either "padded", "ragged" or "disjoint".
        cast_disjoint_kwargs (dict): Kwargs for casting layers.
        mask_assignment (dict): Dictionary of mask name for each input that requires a mask and is cast
            to a disjoint tensor representation.
        index_assignment (dict): Dictionary of assigning indices to the name of their target tensors to which
            the indices refer to.
        return_sub_id (bool): Whether the returned dict contains the sub-graph ID tensors.
        rename_mask_to_id (dict): Mapping of mask names to ID names.

    Returns:
        dict: Model input tensors in disjoint representation.
    """

    is_already_disjoint = False
    out_tensor = {}
    out_batch_id = {}
    out_graph_id = {}
    out_totals = {}

    if input_tensor_type in ["padded", "masked"]:
        if mask_assignment is None or not isinstance(mask_assignment, dict):
            raise ValueError("Mask assignment information is required or invalid.")

        reduced_mask = list(set(list(mask_assignment.values())))

        values_input = {key: value for key, value in model_inputs.items() if key not in reduced_mask}
        mask_input = {key: value for key, value in model_inputs.items() if key in reduced_mask}

        if index_assignment is None:
            index_assignment = {}

        for name_index, name_ref in index_assignment.items():
            ref = values_input[name_ref]
            x = values_input[name_index]
            m, m_ref = mask_assignment[name_index], mask_assignment[name_ref]
            ref_mask = mask_input[m_ref]
            x_mask = mask_input[m]
            o_ref, o_x, b_r, b_x, g_r, g_x, t_r, t_x = CastBatchedIndicesToDisjoint(
                **cast_disjoint_kwargs)([ref, x, ref_mask, x_mask])
            out_tensor[name_index] = o_x
            # Important to not overwrite indices with simple values here! This is the case if indices reference indices.
            if name_ref not in out_tensor.keys():
                out_tensor[name_ref] = o_ref
            out_batch_id[m] = b_x
            out_batch_id[m_ref] = b_r
            out_graph_id[m] = g_x
            out_graph_id[m_ref] = g_r
            out_totals[m] = t_x
            out_totals[m_ref] = t_r

        for name_i, x in values_input.items():

            if name_i in out_tensor.keys():
                continue

            if name_i not in mask_assignment.keys():
                out_tensor[name_i] = CastBatchedGraphStateToDisjoint(**cast_disjoint_kwargs)(x)
                continue

            m = mask_assignment[name_i]
            x_mask = mask_input[m]
            o_x, bi, gi, tot = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([x, x_mask])
            out_tensor[name_i] = o_x
            out_batch_id[m] = bi
            out_graph_id[m] = gi
            out_totals[m] = tot

    elif input_tensor_type in ["ragged", "jagged"]:
        if mask_assignment is None or not isinstance(mask_assignment, dict):
            raise ValueError("Mask assignment information is required or invalid.")

        if index_assignment is None:
            index_assignment = {}

        reduced_mask = list(set(list(mask_assignment.values())))

        values_input = {key: value for key, value in model_inputs.items() if key not in reduced_mask}

        for name_index, name_ref in index_assignment.items():
            ref = values_input[name_ref]
            x = values_input[name_index]
            m, m_ref = mask_assignment[name_index], mask_assignment[name_ref]
            o_ref, o_x, b_r, b_x, g_r, g_x, t_r, t_x = CastRaggedIndicesToDisjoint(
                **cast_disjoint_kwargs)([ref, x])
            out_tensor[name_index] = o_x
            # Important to no overwrite indices with simple values here! This is the case if indices reference indices.
            if name_ref not in out_tensor.keys():
                out_tensor[name_ref] = o_ref
            out_batch_id[m] = b_x
            out_batch_id[m_ref] = b_r
            out_graph_id[m] = g_x
            out_graph_id[m_ref] = g_r
            out_totals[m] = t_x
            out_totals[m_ref] = t_r

        for name_i, x in values_input.items():
            if name_i in out_tensor.keys():
                continue

            if name_i not in mask_assignment.keys():
                out_tensor[name_i] = x
                continue

            m = mask_assignment[name_i]
            o_x, bi, gi, tot = CastRaggedAttributesToDisjoint(**cast_disjoint_kwargs)(x)
            out_tensor[name_i] = o_x
            out_batch_id[m] = bi
            out_graph_id[m] = gi
            out_totals[m] = tot

    else:
        is_already_disjoint = True

    # Rename IDs.
    def map_mask_key(k):
        mapping = {} if rename_mask_to_id is None else rename_mask_to_id
        if k in mapping.keys():
            return mapping[k]
        return k
    out_batch_id = {"graph_id_%s" % map_mask_key(key): value for key, value in out_batch_id.items()}
    out_graph_id = {"%s_id" % map_mask_key(key): value for key, value in out_graph_id.items()}
    out_totals = {"%s_count" % map_mask_key(key): value for key, value in out_totals.items()}

    if is_already_disjoint:
        out = model_inputs
    else:
        out = {}
        out.update(out_tensor)
        out.update(out_batch_id)
        if return_sub_id:
            out.update(out_graph_id)
        out.update(out_totals)

    return out
