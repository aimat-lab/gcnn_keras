import keras as ks
from kgcnn.layers.modules import Input
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use
from kgcnn.layers.scale import get as get_scaler
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from ._model import model_disjoint, model_disjoint_crystal

# To be updated if model is changed in a significant way.
__model_version__ = "2023-10-04"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]

if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'PAiNN' is not supported." % backend_to_use())

# Implementation of PAiNN in `keras` from paper:
# Equivariant message passing for the prediction of tensorial properties and molecular spectra
# Kristof T. Schuett, Oliver T. Unke and Michael Gastegger
# https://arxiv.org/pdf/2102.03150.pdf

model_default = {
    "name": "PAiNN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 128},
    "has_equivariant_input": False,
    "equiv_initialize_kwargs": {"dim": 3, "method": "zeros", "units": 128},
    "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    "pooling_args": {"pooling_method": "scatter_sum"},
    "conv_args": {"units": 128, "cutoff": None, "conv_pool": "scatter_sum"},
    "update_args": {"units": 128, "add_eps": False},
    "equiv_normalization": False, "node_normalization": False,
    "depth": 3,
    "verbose": 10,
    "output_embedding": "graph",
    "output_to_tensor": None,  # deprecated
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               has_equivariant_input: bool = None,
               equiv_initialize_kwargs: dict = None,
               bessel_basis: dict = None,
               depth: int = None,
               pooling_args: dict = None,
               conv_args: dict = None,
               update_args: dict = None,
               equiv_normalization: bool = None,
               node_normalization: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               output_scaling: dict = None,
               output_tensor_type: str = None
               ):
    r"""Make `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.PAiNN.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, ...]`
    with '...' indicating mask or ID tensors following the template below.
    If equivariant input is used via `has_equivariant_input` then input is extended to
    :obj:`[equiv, nodes, coordinates, edge_indices, ...]`

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        equiv_initialize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`EquivariantInitialize` layer.
        bessel_basis (dict): Dictionary of layer arguments unpacked in final :obj:`BesselBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        has_equivariant_input (bool): Whether the first equivariant node embedding is passed to the model.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        conv_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNconv` layer.
        update_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNUpdate` layer.
        equiv_normalization (bool): Whether to apply :obj:`GraphLayerNormalization` to equivariant tensor update.
        node_normalization (bool): Whether to apply :obj:`GraphBatchNormalization` to node tensor update.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    disjoint_inputs = template_cast_list_input(
        model_inputs, input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=([0] if has_equivariant_input else []) + [0, 0, 1],
        index_assignment=([None] if has_equivariant_input else []) + [None, None, 0 + int(has_equivariant_input)]
    )

    if not has_equivariant_input:
        z, x, edi, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = disjoint_inputs
        v = None
    else:
        v, z, x, edi, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = disjoint_inputs

    # Wrapping disjoint model.
    out = model_disjoint(
        [z, x, edi, batch_id_node, batch_id_edge, count_nodes, count_edges, v],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        equiv_initialize_kwargs=equiv_initialize_kwargs,
        bessel_basis=bessel_basis, depth=depth, pooling_args=pooling_args, conv_args=conv_args,
        update_args=update_args, equiv_normalization=equiv_normalization, node_normalization=node_normalization,
        output_embedding=output_embedding, output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, z, batch_id_node])
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


model_crystal_default = {
    "name": "PAiNN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64'},
        {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32'},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 128},
    "has_equivariant_input": False,
    "equiv_initialize_kwargs": {"dim": 3, "method": "zeros"},
    "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    "pooling_args": {"pooling_method": "scatter_sum"},
    "conv_args": {"units": 128, "cutoff": None, "conv_pool": "scatter_sum"},
    "update_args": {"units": 128},
    "equiv_normalization": False,
    "node_normalization": False,
    "depth": 3,
    "verbose": 10,
    "output_embedding": "graph",
    "output_to_tensor": None,  # deprecated
    "output_scaling": None,
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_crystal_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_crystal_model(inputs: list = None,
                       input_tensor_type: str = None,
                       input_embedding: dict = None,  # noqa
                       cast_disjoint_kwargs: dict = None,
                       has_equivariant_input: bool = None,
                       input_node_embedding: dict = None,
                       equiv_initialize_kwargs: dict = None,
                       bessel_basis: dict = None,
                       depth: int = None,
                       pooling_args: dict = None,
                       conv_args: dict = None,
                       update_args: dict = None,
                       equiv_normalization: bool = None,
                       node_normalization: bool = None,
                       name: str = None,
                       verbose: int = None,  # noqa
                       output_embedding: str = None,
                       output_to_tensor: bool = None,  # noqa
                       output_mlp: dict = None,
                       output_scaling: dict = None,
                       output_tensor_type: str = None
                       ):
    r"""Make `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.PAiNN.model_crystal_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, image_translation, lattice, ...]`
    with '...' indicating mask or ID tensors following the template below.
    If equivariant input is used via `has_equivariant_input` then input is extended to
    :obj:`[equiv, nodes, coordinates, edge_indices, image_translation, lattice, ...]`

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        bessel_basis (dict): Dictionary of layer arguments unpacked in final :obj:`BesselBasisLayer` layer.
        equiv_initialize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`EquivariantInitialize` layer.
        depth (int): Number of graph embedding units or depth of the network.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        has_equivariant_input (bool): Whether the first equivariant node embedding is passed to the model.
        conv_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNconv` layer.
        update_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNUpdate` layer.
        equiv_normalization (bool): Whether to apply :obj:`GraphLayerNormalization` to equivariant tensor update.
        node_normalization (bool): Whether to apply :obj:`GraphBatchNormalization` to node tensor update.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj_inputs = template_cast_list_input(
        model_inputs, input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=([0] if has_equivariant_input else []) + [
            0, 0, 1, 1, None],
        index_assignment=([0] if has_equivariant_input else []) + [
            None, None, 0 + int(has_equivariant_input), None, None]
    )

    if not has_equivariant_input:
        z, x, edi, img, lattice, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj_inputs
        v = None
    else:
        v, z, x, edi, img, lattice, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj_inputs

    # Wrapping disjoint model.
    out = model_disjoint_crystal(
        [z, x, edi, img, lattice, batch_id_node, batch_id_edge, count_nodes, count_edges, v],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        input_node_embedding=input_node_embedding, equiv_initialize_kwargs=equiv_initialize_kwargs,
        bessel_basis=bessel_basis, depth=depth, pooling_args=pooling_args, conv_args=conv_args,
        update_args=update_args, equiv_normalization=equiv_normalization, node_normalization=node_normalization,
        output_embedding=output_embedding, output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, z, batch_id_node])
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

    model.__kgcnn_model_version__ = __model_version__
    return model


make_crystal_model.__doc__ = make_crystal_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
