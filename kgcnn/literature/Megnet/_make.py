import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint, model_disjoint_crystal
from kgcnn.layers.modules import Input
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-05"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'Megnet' is not supported." % backend_to_use())

# Implementation of Megnet in `tf.keras` from paper:
# Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals
# by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong*
# https://github.com/materialsvirtuallab/megnet
# https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294


model_default = {
    "name": "Megnet",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {'shape': [1], 'name': "charge", 'dtype': 'float32'}, # graph state
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_graph_embedding": {"input_dim": 100, "output_dim": 64},
    "make_distance": True, "expand_distance": True,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "meg_block_args": {
        "node_embed": [64, 32, 32],
        "edge_embed": [64, 32, 32],
        "env_embed": [64, 32, 32],
        "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    "set2set_args": {"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
    "node_ff_args": {
        "units": [64, 32],
        "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    "edge_ff_args": {
        "units": [64, 32],
        "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    "state_ff_args": {
        "units": [64, 32],
        "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    "nblocks": 3, "has_ff": True, "dropout": None, "use_set2set": True,
    "verbose": 10,
    "output_embedding": "graph",
    "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                   "activation": [
                       {"class_name": "function", "config": "kgcnn>softplus2"},
                       {"class_name": "function", "config": "kgcnn>softplus2"},
                       "linear"
                   ]},
    "output_scaling": None
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               input_graph_embedding: dict = None,
               expand_distance: bool = None,
               make_distance: bool = None,
               gauss_args: dict = None,
               meg_block_args: dict = None,
               set2set_args: dict = None,
               node_ff_args: dict = None,
               edge_ff_args: dict = None,
               state_ff_args: dict = None,
               use_set2set: bool = None,
               nblocks: int = None,
               has_ff: bool = None,
               dropout: float = None,
               name: str = None,
               verbose: int = None,  # noqa
               output_embedding: str = None,
               output_mlp: dict = None,
               output_tensor_type: str = None,
               output_scaling: dict = None
               ):
    r"""Make `MegNet <https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Megnet.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, graph_state, ...]` with `make_distance` and
    with '...' indicating mask or ID tensors following the template below.
    Note that you could also supply edge features with `make_distance` to False, which would make the input
    :obj:`[nodes, edges, edge_indices, graph_state...]` .

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        input_graph_embedding (dict): Dictionary of embedding arguments for graph unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        meg_block_args (dict): Dictionary of layer arguments unpacked in :obj:`MEGnetBlock` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in `:obj:PoolingSet2SetEncoder` layer.
        node_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        edge_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        state_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2SetEncoder` layer.
        nblocks (int): Number of graph embedding blocks or depth of the network.
        has_ff (bool): Use feed-forward MLP in each block.
        dropout (int): Dropout to use. Default is None.
        name (str): Name of the model.
        verbose (int): Verbosity level of print.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
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
        mask_assignment=[0, 0 if make_distance else 1, 1, None],
        index_assignment=[None, None, 0, None]
    )

    n, x, disjoint_indices, gs, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj

    out = model_disjoint(
        [n, x, disjoint_indices, gs, batch_id_node, batch_id_edge, count_nodes, count_edges],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_graph_embedding=("int" in inputs[3]['dtype']) if input_graph_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_graph_embedding=input_graph_embedding,
        expand_distance=expand_distance,
        make_distance=make_distance,
        gauss_args=gauss_args,
        meg_block_args=meg_block_args,
        set2set_args=set2set_args,
        node_ff_args=node_ff_args,
        edge_ff_args=edge_ff_args,
        state_ff_args=state_ff_args,
        use_set2set=use_set2set,
        nblocks=nblocks,
        has_ff=has_ff,
        dropout=dropout,
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
    'name': "Megnet",
    'inputs': [
        {'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
        {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
        {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
        {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
        {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64', 'ragged': True},
        {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
    ],
    "input_tensor_type": "ragged",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_graph_embedding": {"input_dim": 100, "output_dim": 64},
    "make_distance": True, "expand_distance": True,
    'gauss_args': {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    'meg_block_args': {
        'node_embed': [64, 32, 32],
        'edge_embed': [64, 32, 32],
        'env_embed': [64, 32, 32],
        'activation': {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    'set2set_args': {'channels': 16, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},
    'node_ff_args': {
        "units": [64, 32],
        "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    'edge_ff_args': {
        "units": [64, 32],
        "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    'state_ff_args': {
        "units": [64, 32],
        "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
    },
    'nblocks': 3, 'has_ff': True, 'dropout': None, 'use_set2set': True,
    'verbose': 10,
    'output_embedding': 'graph',
    'output_mlp': {"use_bias": [True, True, True], "units": [32, 16, 1],
                   "activation": [
                       {"class_name": "function", "config": "kgcnn>softplus2"},
                       {"class_name": "function", "config": "kgcnn>softplus2"},
                       'linear'
                   ]},
    "output_scaling": None
}


@update_model_kwargs(model_crystal_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_crystal_model(inputs: list = None,
                       input_tensor_type: str = None,
                       cast_disjoint_kwargs: dict = None,
                       input_embedding: dict = None,
                       input_node_embedding: dict = None,
                       input_graph_embedding: dict = None,
                       expand_distance: bool = None,
                       make_distance: bool = None,
                       gauss_args: dict = None,
                       meg_block_args: dict = None,
                       set2set_args: dict = None,
                       node_ff_args: dict = None,
                       edge_ff_args: dict = None,
                       state_ff_args: dict = None,
                       use_set2set: bool = None,
                       nblocks: int = None,
                       has_ff: bool = None,
                       dropout: float = None,
                       name: str = None,
                       verbose: int = None,
                       output_embedding: str = None,
                       output_mlp: dict = None,
                       output_tensor_type: str = None,
                       output_scaling: dict = None
                       ):
    r"""Make `MegNet <https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Megnet.model_crystal_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, graph_state, image_translation, lattice, ...]`
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
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        input_graph_embedding (dict): Dictionary of embedding arguments for graph unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        meg_block_args (dict): Dictionary of layer arguments unpacked in :obj:`MEGnetBlock` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in `:obj:PoolingSet2SetEncoder` layer.
        node_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        edge_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        state_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2SetEncoder` layer.
        nblocks (int): Number of graph embedding blocks or depth of the network.
        has_ff (bool): Use feed-forward MLP in each block.
        dropout (int): Dropout to use. Default is None.
        name (str): Name of the model.
        verbose (int): Verbosity level of print.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
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
        model_inputs, input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0 if make_distance else 1, 1, None, 1, None],
        index_assignment=[None, None, 0, None, None, None]
    )

    n, x, djx, gs, img, lattice, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj

    # Wrapp disjoint model
    out = model_disjoint_crystal(
        [n, x, djx, gs, img, lattice, batch_id_node, batch_id_edge, count_nodes, count_edges],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_graph_embedding=("int" in inputs[3]['dtype']) if input_graph_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_graph_embedding=input_graph_embedding,
        expand_distance=expand_distance,
        make_distance=make_distance,
        gauss_args=gauss_args,
        meg_block_args=meg_block_args,
        set2set_args=set2set_args,
        node_ff_args=node_ff_args,
        edge_ff_args=edge_ff_args,
        state_ff_args=state_ff_args,
        use_set2set=use_set2set,
        nblocks=nblocks,
        has_ff=has_ff,
        dropout=dropout,
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
