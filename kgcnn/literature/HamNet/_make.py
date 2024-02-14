import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint
from kgcnn.layers.modules import Input
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-09-07"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'HamNet' is not supported." % backend_to_use())

# Implementation of HamNet in `keras` from paper:
# HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks
# by Ziyao Li, Shuwen Yang, Guojie Song, Lingsheng Cai
# Link to paper: https://arxiv.org/abs/2105.03688
# Original implementation: https://github.com/PKUterran/HamNet
# Later implementation: https://github.com/PKUterran/MoleculeClub
# Note: the 2. implementation is cleaner than the original code and has been used as template.


model_default = {
    "name": "HamNet",
    "inputs": [
        {'shape': (None,), 'name': "node_number", 'dtype': 'int64'},
        {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32'},
        {'shape': (None, 64), 'name': "edge_attributes", 'dtype': 'float32'},
        {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64'},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
    "message_kwargs": {"units": 128, "units_edge": 128},
    "fingerprint_kwargs": {"units": 128, "units_attend": 128, "depth": 2},
    "gru_kwargs": {"units": 128},
    "verbose": 10,
    "depth": 1,
    "union_type_node": "gru",
    "union_type_edge": "None",
    "given_coordinates": True,
    "output_embedding": "graph",
    "output_tensor_type": "padded",
    "output_to_tensor": None,  # deprecated
    'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ['relu', 'relu', 'linear']},
    "output_scaling": None
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(name: str = None,
               inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               verbose: int = None,
               message_kwargs: dict = None,
               gru_kwargs: dict = None,
               fingerprint_kwargs: dict = None,
               union_type_node: str = None,
               union_type_edge: str = None,
               given_coordinates: bool = None,
               depth: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               output_tensor_type: str = None,
               output_scaling: dict = None
               ):
    r"""Make `HamNet <https://arxiv.org/abs/2105.03688>`__ graph model via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.HamNet.model_default` .

    .. note::

        At the moment only the Fingerprint Generator for graph embeddings is implemented and coordinates must
        be provided as model input.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edges, edge_indices, ...]` with `given_coordinates` and
    with '...' indicating mask or ID tensors following the template below.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        name (str): Name of the model.
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of embedding arguments for edges unpacked in :obj:`Embedding` layers.
        verbose (int): Level of verbosity. For logging and printing.
        message_kwargs (dict): Dictionary of layer arguments unpacked in message passing layer for node updates.
        gru_kwargs (dict): Dictionary of layer arguments unpacked in gated recurrent unit update layer.
        fingerprint_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`HamNetFingerprintGenerator` layer.
        given_coordinates (bool): Whether coordinates are provided as model input, or are computed by the Model.
        union_type_edge (str): Union type of edge updates. Choose "gru", "naive" or "None".
        union_type_node (str): Union type of node updates. Choose "gru", "naive" or "None".
        depth (int): Depth or number of (message passing) layers of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    di_inputs = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0, 1, 1],
        index_assignment=[None, None, None, 0]
    )

    n, x, ed, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = di_inputs

    # Wrapping disjoint model.
    out = model_disjoint(
        [n, x, ed, disjoint_indices, batch_id_node, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[2]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        given_coordinates=given_coordinates,
        gru_kwargs=gru_kwargs,
        message_kwargs=message_kwargs,
        fingerprint_kwargs=fingerprint_kwargs,
        output_embedding=output_embedding,
        output_mlp=output_mlp,
        union_type_edge=union_type_edge,
        union_type_node=union_type_node,
        depth=depth
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
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


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
