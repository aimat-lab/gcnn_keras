import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint, model_disjoint_weighted
from kgcnn.layers.modules import Input
from kgcnn.models.utils import update_model_kwargs
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from keras.backend import backend as backend_to_use


# Keep track of model version from commit date in literature.
__kgcnn_model_version__ = "2023-09-30"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'GCN' is not supported." % backend_to_use())

# Implementation of GCN in `keras` from paper:
# Semi-Supervised Classification with Graph Convolutional Networks
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn

model_default = {
    "name": "GCN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
    "gcn_args": {"units": 100, "use_bias": True, "activation": "relu", "pooling_method": "sum"},
    "depth": 3,
    "verbose": 10,
    "node_pooling_args": {"pooling_method": "scatter_sum"},
    "output_embedding": "graph",
    "output_to_tensor": None,  # deprecated
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]},
    "output_scaling": None,
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               depth: int = None,
               gcn_args: dict = None,
               name: str = None,
               verbose: int = None,
               node_pooling_args: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_tensor_type: str = None,
               output_mlp: dict = None,
               output_scaling: dict = None):
    r"""Make `GCN <https://arxiv.org/abs/1609.02907>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GCN.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, edges, edge_indices, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Edges are actually edge single weight values which are entries of the pre-scaled adjacency matrix.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gcn_args (dict): Dictionary of layer arguments unpacked in :obj:`GCN` convolutional layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    if inputs[1]['shape'][-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): %s." % inputs[1]['shape'])

    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj_inputs = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 1, 1],
        index_assignment=[None, None, 0]
    )

    n, ed, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj_inputs

    out = model_disjoint(
        [n, ed, disjoint_indices, batch_id_node, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding, input_edge_embedding=input_edge_embedding,
        depth=depth, gcn_args=gcn_args, node_pooling_args=node_pooling_args, output_embedding=output_embedding,
        output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        out = scaler(out)

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __kgcnn_model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)
    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)


model_default_weighted = {
    "name": "GCN_weighted",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 1), "name": "node_weights", "dtype": "float32"},
        {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
    "gcn_args": {"units": 100, "use_bias": True, "activation": "relu", "pooling_method": "sum"},
    "depth": 3, "verbose": 1,
    "output_embedding": "graph",
    "output_to_tensor": None,  # deprecated
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]},
    "output_scaling": None
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model_weighted(inputs: list = None,
                        input_tensor_type: str = None,
                        cast_disjoint_kwargs: dict = None,
                        input_embedding: dict = None,  # noqa
                        input_node_embedding: dict = None,
                        input_edge_embedding: dict = None,
                        depth: int = None,
                        gcn_args: dict = None,
                        name: str = None,
                        verbose: int = None,  # noqa
                        node_pooling_args: dict = None,
                        output_embedding: str = None,
                        output_to_tensor: bool = None,  # noqa
                        output_tensor_type: str = None,
                        output_mlp: dict = None,
                        output_scaling: dict = None):
    r"""Make weighted `GCN <https://arxiv.org/abs/1609.02907>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GCN.model_default_weighted`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, node_weights, edges, edge_indices, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Edges are actually edge single weight values which are entries of the pre-scaled adjacency matrix.
    The node weights are used in the global pooling step.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gcn_args (dict): Dictionary of layer arguments unpacked in :obj:`GCN` convolutional layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    if inputs[2]['shape'][-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): %s." % inputs[1]['shape'])

    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj_inputs = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0, 1, 1],
        index_assignment=[None, None, None, 0]
    )

    n, nw, ed, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj_inputs

    out = model_disjoint_weighted(
        [n, nw, ed, disjoint_indices, batch_id_node, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding, input_edge_embedding=input_edge_embedding,
        depth=depth, gcn_args=gcn_args, node_pooling_args=node_pooling_args, output_embedding=output_embedding,
        output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        out = scaler(out)

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __kgcnn_model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model_weighted.__doc__ = make_model_weighted.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
