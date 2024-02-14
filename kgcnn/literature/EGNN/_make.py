import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint
from kgcnn.layers.modules import Input
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-04"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'EGNN' is not supported." % backend_to_use())

# Implementation of EGNN in `keras` from paper:
# E(n) Equivariant Graph Neural Networks
# by Victor Garcia Satorras, Emiel Hoogeboom, Max Welling (2021)
# https://arxiv.org/abs/2102.09844


model_default = {
    "name": "EGNN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 10), "name": "edge_attributes", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
    ],
    "input_tensor_type": "padded",
    "cast_disjoint_kwargs": {},
    "input_embedding": None,
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 95, "output_dim": 64},
    "depth": 4,
    "node_mlp_initialize": None,
    "euclidean_norm_kwargs": {"keepdims": True, "axis": -1},
    "use_edge_attributes": True,
    "edge_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
    "edge_attention_kwargs": None,  # {"units: 1", "activation": "sigmoid"}
    "use_normalized_difference": False,
    "expand_distance_kwargs": None,
    "coord_mlp_kwargs": {"units": [64, 1], "activation": ["swish", "linear"]},  # option: "tanh" at the end.
    "pooling_coord_kwargs": {"pooling_method": "mean"},
    "pooling_edge_kwargs": {"pooling_method": "sum"},
    "node_normalize_kwargs": None,
    "use_node_attributes": False,
    "node_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
    "use_skip": True,
    "verbose": 10,
    "node_decoder_kwargs": None,
    "node_pooling_kwargs": {"pooling_method": "sum"},
    "output_embedding": "graph",
    "output_to_tensor": None,  # deprecated
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]},
    "output_scaling": None,
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(name: str = None,
               inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
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
               verbose: int = None,
               node_decoder_kwargs: dict = None,
               node_pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               output_tensor_type: str = None,
               output_scaling: dict = None
               ):
    r"""Make `EGNN <https://arxiv.org/abs/2102.09844>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.EGNN.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, node_coordinates, edge_attributes, edge_indices, ...]`
    with '...' indicating mask or ID tensors following the template below.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        name (str): Name of the model. Default is "EGNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        euclidean_norm_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`EuclideanNorm`.
        node_mlp_initialize (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer for start embedding.
        use_edge_attributes (bool): Whether to use edge attributes including for example further edge information.
        edge_mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        edge_attention_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        use_normalized_difference (bool): Whether to use a normalized difference vector for nodes.
        expand_distance_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PositionEncodingBasisLayer`.
        coord_mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        pooling_coord_kwargs (dict):
        pooling_edge_kwargs (dict):
        node_normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphLayerNormalization` layer.
        use_node_attributes (bool): Whether to add node attributes before node MLP.
        node_mlp_kwargs (dict):
        use_skip (bool):
        verbose (int): Level of verbosity.
        node_decoder_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer after graph network.
        node_pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        index_assignment=[None, None, None, 0],
        mask_assignment=[0, 0, 1, 1]
    )

    n, x, ed, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj

    out = model_disjoint(
        [n, x, ed, disjoint_indices, batch_id_node, batch_id_edge, count_nodes, count_edges],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[2]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        depth=depth,
        euclidean_norm_kwargs=euclidean_norm_kwargs,
        node_mlp_initialize=node_mlp_initialize,
        use_edge_attributes=use_edge_attributes,
        edge_mlp_kwargs=edge_mlp_kwargs,
        edge_attention_kwargs=edge_attention_kwargs,
        use_normalized_difference=use_normalized_difference,
        expand_distance_kwargs=expand_distance_kwargs,
        coord_mlp_kwargs=coord_mlp_kwargs,
        pooling_coord_kwargs=pooling_coord_kwargs,
        pooling_edge_kwargs=pooling_edge_kwargs,
        node_normalize_kwargs=node_normalize_kwargs,
        use_node_attributes=use_node_attributes,
        node_mlp_kwargs=node_mlp_kwargs,
        use_skip=use_skip,
        node_decoder_kwargs=node_decoder_kwargs,
        node_pooling_kwargs=node_pooling_kwargs,
        output_embedding=output_embedding,
        output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, n, batch_id_node])
        else:
            out = scaler(out)

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
