import keras as ks
from kgcnn.layers.modules import Input
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use
from kgcnn.layers.scale import get as get_scaler
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from ._model import model_disjoint_crystal

# To be updated if model is changed in a significant way.
__model_version__ = "2023-11-28"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]

if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'PAiNN' is not supported." % backend_to_use())

# Implementation of CGCNN in `keras` from paper:
# Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties
# Tian Xie and Jeffrey C. Grossman
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301


model_crystal_default = {
    'name': 'CGCNN',
    'inputs': [
        {'shape': (None,), 'name': 'node_number', 'dtype': 'int64'},
        {'shape': (None, 3), 'name': 'node_frac_coordinates', 'dtype': 'float64'},
        {'shape': (None, 2), 'name': 'edge_indices', 'dtype': 'int64'},
        {'shape': (None, 3), 'name': 'cell_translations', 'dtype': 'float32'},
        {'shape': (3, 3), 'name': 'lattice_matrix', 'dtype': 'float64'},
        # {'shape': (None, 1), 'name': 'multiplicities', 'dtype': 'float32'},  # For asu"
        # {'shape': (None, 4, 4), 'name': 'symmops', 'dtype': 'float64'},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    'representation': 'unit',  # None, 'asu' or 'unit'
    'expand_distance': True,
    'make_distances': True,
    'gauss_args': {'bins': 40, 'distance': 8, 'offset': 0.0, 'sigma': 0.4},
    'depth': 3,
    "verbose": 10,
    'conv_layer_args': {
        'units': 64,
        'activation_s': 'softplus',
        'activation_out': 'softplus',
        'batch_normalization': True,
    },
    'node_pooling_args': {'pooling_method': 'scatter_mean'},
    "output_embedding": "graph",
    "output_to_tensor": None,  # deprecated
    "output_scaling": None,
    "output_tensor_type": "padded",
    'output_mlp': {'use_bias': [True, False], 'units': [64, 1],
                   'activation': ['softplus', 'linear']},
}


@update_model_kwargs(model_crystal_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_crystal_model(inputs: list = None,
                       input_tensor_type: str = None,
                       input_embedding: dict = None,
                       input_node_embedding: dict = None,
                       cast_disjoint_kwargs: dict = None,
                       representation: str = None,
                       make_distances: bool = None,
                       conv_layer_args: dict = None,
                       expand_distance: bool = None,
                       depth: int = None,
                       name: str = None,
                       verbose: int = None,
                       gauss_args: dict = None,
                       node_pooling_args: dict = None,
                       output_to_tensor: dict = None,
                       output_mlp: dict = None,
                       output_embedding: str = None,
                       output_scaling: dict = None,
                       output_tensor_type: str = None
                       ):
    r"""Make `CGCNN <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`__ graph network
    via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.CGCNN.model_crystal_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    Model supports :obj:`[node_attributes, node_frac_coordinates, bond_indices, lattice, cell_translations, ...]`
    if representation='unit'` and `make_distances=True` or
    :obj:`[node_attributes, node_frac_coords, bond_indices, lattice, cell_translations, multiplicities, symmops, ...]`
    if `representation='asu'` and `make_distances=True`
    or :obj:`[node_attributes, edge_distance, bond_indices, ...]`
    if `make_distances=False` .
    The optional tensor :obj:`multiplicities` is a node-like feature tensor with a single value that gives
    the multiplicity for each node.
    The optional tensor :obj:`symmops` is an edge-like feature tensor with a matrix of shape `(4, 4)` for each edge
    that defines the symmetry operation.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        make_distances (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        representation (str): The representation of unit cell. Can be either `None`, 'asu' or 'unit'. Default is 'unit'.
        conv_layer_args (dict):
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    d_in = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0 if make_distances else 1, 1, 1, None] + ([0, 1] if representation == "asu" else []),
        index_assignment=[None, None, 0, None, None] + ([None, None] if representation == "asu" else [])
    )

    if representation == "asu":
        n, x, djx, img, lattice, m, sym, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = d_in
    else:
        n, x, djx, img, lattice, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = d_in
        m, sym = None, None

    # Wrapp disjoint model
    out = model_disjoint_crystal(
        [n, m, x, sym, djx, img, lattice, batch_id_node, batch_id_edge, count_nodes, count_edges],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        representation=representation,
        output_embedding=output_embedding,
        input_node_embedding=input_node_embedding,
        expand_distance=expand_distance,
        conv_layer_args=conv_layer_args,
        make_distances=make_distances,
        depth=depth,
        gauss_args=gauss_args,
        node_pooling_args=node_pooling_args,
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
        output_embedding=output_embedding,
        output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_crystal_model.__doc__ = make_crystal_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
