import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint, model_disjoint_crystal
from kgcnn.layers.modules import Input
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-11-22"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'NMPN' is not supported." % backend_to_use())

# Implementation of NMPN in `keras` from paper:
# Neural Message Passing for Quantum Chemistry
# by Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
# http://arxiv.org/abs/1704.01212


model_default = {
    "name": "NMPN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None,), "name": "edge_number", "dtype": "int64"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "cast_disjoint_kwargs": {},
    "input_embedding": None,  # deprecated
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
    "geometric_edge": False,
    "make_distance": False,
    "expand_distance": False,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "set2set_args": {"channels": 32, "T": 3, "pooling_method": "scatter_sum",
                     "init_qstar": "0"},
    "pooling_args": {"pooling_method": "scatter_sum"},
    "edge_mlp": {"use_bias": True, "activation": "swish", "units": [64, 64, 64]},
    "use_set2set": True,
    "depth": 3,
    "node_dim": 64,
    "verbose": 10,
    "output_embedding": 'graph',
    "output_to_tensor": None,  # deprecated
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["selu", "selu", "sigmoid"]},
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               geometric_edge: bool = None,
               make_distance: bool = None,
               expand_distance: bool = None,
               gauss_args: dict = None,
               set2set_args: dict = None,
               pooling_args: dict = None,
               edge_mlp: dict = None,
               use_set2set: bool = None,
               node_dim: int = None,
               depth: int = None,
               verbose: int = None,
               name: str = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               output_tensor_type: str = None,
               output_scaling: dict = None
               ):
    r"""Make `NMPN <http://arxiv.org/abs/1704.01212>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.NMPN.model_default`.

    Model inputs:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, ...]` with `make_distance` and `geometric_edge`
    with '...' indicating mask or ID tensors following the template below.
    Note that you could also supply edge features with `make_distance` to False, which would make the input
    :obj:`[nodes, edges, edge_indices, ...]` .

    %s

    Model outputs:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layer.
        input_edge_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layer.
        geometric_edge (bool): Whether the edges are geometric, like distance or coordinates.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2SetEncoder` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            `AggregateLocalEdges` layers.
        edge_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge matrix.
        use_set2set (bool): Whether to use :obj:`PoolingSet2SetEncoder` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.
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
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0 if make_distance else 1, 1],
        index_assignment=[None, None, 0]
    )

    n, x, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = disjoint_inputs

    out = model_disjoint(
        [n, x, disjoint_indices, batch_id_node, batch_id_edge, count_nodes, count_edges],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        geometric_edge=geometric_edge,
        make_distance=make_distance,
        expand_distance=expand_distance,
        gauss_args=gauss_args,
        set2set_args=set2set_args,
        pooling_args=pooling_args,
        edge_mlp=edge_mlp,
        use_set2set=use_set2set,
        node_dim=node_dim,
        depth=depth,
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


model_crystal_default = {
    "name": "NMPN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64', 'ragged': True},
        {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
    ],
    "input_tensor_type": "ragged",
    "cast_disjoint_kwargs": {},
    "input_embedding": None,  # deprecated
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
    "geometric_edge": True,
    "make_distance": True,
    "expand_distance": True,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum",
                     "init_qstar": "0"},
    "pooling_args": {"pooling_method": "sum"},
    "edge_mlp": {"use_bias": True, "activation": "swish", "units": [64, 64, 64]},
    "use_set2set": True, "depth": 3, "node_dim": 64,
    "verbose": 10,
    "output_embedding": 'graph',
    "output_to_tensor": None,
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["selu", "selu", "sigmoid"]},
}


@update_model_kwargs(model_crystal_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_crystal_model(inputs: list = None,
                       input_tensor_type: str = None,
                       cast_disjoint_kwargs: dict = None,
                       input_embedding: dict = None,  # noqa
                       input_node_embedding: dict = None,
                       input_edge_embedding: dict = None,
                       geometric_edge: bool = None,
                       make_distance: bool = None,
                       expand_distance: bool = None,
                       gauss_args: dict = None,
                       set2set_args: dict = None,
                       pooling_args: dict = None,
                       edge_mlp: dict = None,
                       use_set2set: bool = None,
                       node_dim: int = None,
                       depth: int = None,
                       verbose: int = None,  # noqa
                       name: str = None,
                       output_embedding: str = None,
                       output_to_tensor: bool = None,  # noqa
                       output_mlp: dict = None,
                       output_tensor_type: str = None,
                       output_scaling: dict = None
                       ):
    r"""Make `NMPN <http://arxiv.org/abs/1704.01212>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.NMPN.model_crystal_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, image_translation, lattice, ...]`
    with '...' indicating mask or ID tensors following the template below.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layer.
        input_edge_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layer.
        geometric_edge (bool): Whether the edges are geometric, like distance or coordinates.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2SetEncoder` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            `AggregateLocalEdges` layers.
        edge_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge matrix.
        use_set2set (bool): Whether to use :obj:`PoolingSet2SetEncoder` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.
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

    dj = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0 if make_distance else 1, 1, 1, None],
        index_assignment=[None, None, 0, None, None]
    )

    n, x, d_indices, img, lattice, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj

    out = model_disjoint_crystal(
        [n, x, d_indices, img, lattice, batch_id_node, batch_id_edge, count_nodes, count_edges],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        geometric_edge=geometric_edge,
        make_distance=make_distance,
        expand_distance=expand_distance,
        gauss_args=gauss_args,
        set2set_args=set2set_args,
        pooling_args=pooling_args,
        edge_mlp=edge_mlp,
        use_set2set=use_set2set,
        node_dim=node_dim,
        depth=depth,
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


make_crystal_model.__doc__ = make_crystal_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
