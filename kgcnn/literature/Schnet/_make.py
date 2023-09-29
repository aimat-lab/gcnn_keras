import keras_core as ks
from kgcnn.layers.casting import (CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
                                  CastDisjointToGraphState, CastDisjointToBatchedAttributes, CastGraphStateToDisjoint)
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint, model_disjoint_crystal
from kgcnn.models.utils import update_model_kwargs
from keras_core.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-09-07"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'Schnet' is not supported." % backend_to_use())

# Implementation of Schnet in `keras` from paper:
# by Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela,
# Alexandre Tkatchenko, Klaus-Robert Müller (2018)
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


model_default = {
    "name": "Schnet",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "make_distance": True,
    "expand_distance": True,
    "interaction_args": {"units": 128, "use_bias": True,
                         "activation": "kgcnn>shifted_softplus", "cfconv_pool": "scatter_sum"},
    "node_pooling_args": {"pooling_method": "sum"},
    "depth": 4,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "verbose": 10,
    "last_mlp": {"use_bias": [True, True], "units": [128, 64],
                 "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": True,
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["kgcnn>shifted_softplus", "linear"]}
}


@update_model_kwargs(model_default, update_recursive=0)
def make_model(inputs: list = None,
               cast_disjoint_kwargs: dict = None,
               input_node_embedding: dict = None,
               make_distance: bool = None,
               expand_distance: bool = None,
               gauss_args: dict = None,
               interaction_args: dict = None,
               node_pooling_args: dict = None,
               depth: int = None,
               name: str = None,
               verbose: int = None,
               last_mlp: dict = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    r"""Make `SchNet <https://arxiv.org/abs/1706.08566>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Schnet.model_default` .

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices, total_nodes, total_edges]`
        or `[node_attributes, node_coordinates, edge_indices, total_nodes, total_edges]`
        if :obj:`make_distance=True` and :obj:`expand_distance=True`
        to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, N, F)` or `(batch, N)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, M, D)` expanded
              in a basis of dimension `D` or `(batch, M, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, M, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - total_nodes(Tensor): Number of Nodes in graph if not same sized graphs of shape `(batch, )` .
            - total_edges(Tensor): Number of Edges in graph if not same sized graphs of shape `(batch, )` .

    Outputs:
        Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
        input_node_embedding (dict): Dictionary of embedding arguments for nodes etc.
            unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        interaction_args (dict): Dictionary of layer arguments unpacked in final :obj:`SchNetInteraction` layers.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.

    Returns:
        :obj:`ks.models.Model`
    """
    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    batched_nodes, batched_x, batched_indices, total_nodes, total_edges = model_inputs
    n, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])
    if make_distance:
        x, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_x, total_nodes])
    else:
        x, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_x, total_edges])

    out = model_disjoint(
        [n, x, disjoint_indices, batch_id_node, count_nodes],
        use_node_embedding=len(inputs[0]['shape']) < 2, input_node_embedding=input_node_embedding,
        make_distance=make_distance, expand_distance=expand_distance, gauss_args=gauss_args,
        interaction_args=interaction_args, node_pooling_args=node_pooling_args, depth=depth,
        last_mlp=last_mlp, output_embedding=output_embedding, use_output_mlp=use_output_mlp,
        output_mlp=output_mlp)

    # Output embedding choice
    if output_embedding == 'graph':
        out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        if output_to_tensor:
            out = CastDisjointToBatchedAttributes(**cast_disjoint_kwargs)([batched_nodes, out, batch_id_node, node_id])
        else:
            out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, batched_nodes])
        else:
            out = scaler(out)

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


model_crystal_default = {
    "name": "Schnet",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "float32"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (None, 3), "name": "edge_image", "dtype": "int64"},
        {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "make_distance": True, "expand_distance": True,
    "interaction_args": {"units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"},
    "node_pooling_args": {"pooling_method": "sum"},
    "depth": 4,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "verbose": 10,
    "last_mlp": {"use_bias": [True, True], "units": [128, 64],
                 "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["kgcnn>shifted_softplus", "linear"]}
}


@update_model_kwargs(model_default, update_recursive=0)
def make_crystal_model(inputs: list = None,
                       cast_disjoint_kwargs: dict = None,
                       input_node_embedding: dict = None,
                       make_distance: bool = None,
                       expand_distance: bool = None,
                       gauss_args: dict = None,
                       interaction_args: dict = None,
                       node_pooling_args: dict = None,
                       depth: int = None,
                       name: str = None,
                       verbose: int = None,
                       last_mlp: dict = None,
                       output_embedding: str = None,
                       use_output_mlp: bool = None,
                       output_to_tensor: bool = None,
                       output_mlp: dict = None,
                       output_scaling: dict = None
                       ):
    r"""Make `SchNet <https://arxiv.org/abs/1706.08566>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Schnet.model_crystal_default`.

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices, edge_image, lattice, total_nodes, total_edges]`
        or `[node_attributes, node_coordinates, edge_indices, edge_image, lattice, total_nodes, total_edges]`
        if :obj:`make_distance=True` and :obj:`expand_distance=True` to compute edge distances from node coordinates
        within the model.

            - node_attributes (Tensor): Node attributes of shape `(batch, N, F)` or `(batch, N)`
              using an embedding layer.
            - edge_distance (Tensor): Edge distance of shape `(batch, M, D)` expanded
              in a basis of dimension `D` or `(batch, M, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (Tensor): Index list for edges of shape `(batch, M, 2)`.
            - node_coordinates (Tensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - lattice (Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - edge_image (Tensor): Indices of the periodic image the sending node is located.
            - total_nodes(Tensor): Number of Nodes in graph if not same sized graphs of shape `(batch, )` .
            - total_edges(Tensor): Number of Edges in graph if not same sized graphs of shape `(batch, )` .

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
        input_node_embedding (dict): Dictionary of embedding arguments for nodes etc.
            unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        interaction_args (dict): Dictionary of layer arguments unpacked in final :obj:`SchNetInteraction` layers.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    batched_nodes, batched_x, batched_indices, edge_image, lattice, total_nodes, total_edges = model_inputs
    n, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])
    lattice = CastGraphStateToDisjoint(**cast_disjoint_kwargs)(lattice)
    if make_distance:
        x, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_x, total_nodes])
        edge_image, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([edge_image, total_edges])
    else:
        x, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_x, total_edges])

    # Wrapp disjoint model
    out = model_disjoint_crystal(
        [n, x, disjoint_indices, edge_image, lattice, batch_id_node, batch_id_edge, count_nodes],
        use_node_embedding=len(inputs[0]['shape']) < 2, input_node_embedding=input_node_embedding,
        make_distance=make_distance, expand_distance=expand_distance, gauss_args=gauss_args,
        interaction_args=interaction_args, node_pooling_args=node_pooling_args, depth=depth, last_mlp=last_mlp,
        output_embedding=output_embedding, use_output_mlp=use_output_mlp, output_mlp=output_mlp
    )

    # Output embedding choice
    if output_embedding == 'graph':
        out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        if output_to_tensor:
            out = CastDisjointToBatchedAttributes(**cast_disjoint_kwargs)([batched_nodes, out, batch_id_node, node_id])
        else:
            out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, batched_nodes])
        else:
            out = scaler(out)

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model
