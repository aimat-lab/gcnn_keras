import keras as ks
from kgcnn.layers.scale import get as get_scaler
from kgcnn.models.utils import update_model_kwargs
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from keras.backend import backend as backend_to_use
from kgcnn.layers.modules import Input
from ._model import model_disjoint

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023-10-23"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]

if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'DGIN' is not supported." % backend_to_use())

# Implementation of DGIN in `keras` from paper:
# Analyzing Learned Molecular Representations for Property Prediction
# by Oliver Wieder, Mélaine Kuenemann, Marcus Wieder, Thomas Seidel,
# Christophe Meyer, Sharon D Bryant and Thierry Langer
# https://pubmed.ncbi.nlm.nih.gov/34684766/

model_default = {
    "name": "DGIN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None,), "name": "edge_number", "dtype": "int64"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
        {"shape": (), "name": "total_reverse", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "cast_disjoint_kwargs": {},
    "input_embedding": None,  # deprecated
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
    "input_graph_embedding": {"input_dim": 100, "output_dim": 64},
    "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                "use_normalization": True, "normalization_technique": "graph_batch"},
    "gin_args": {},
    "last_mlp": {"use_bias": [True, True], "units": [64, 64],
                 "activation": ["relu", "relu"]},
    "pooling_args": {"pooling_method": "sum"},
    "use_graph_state": False,
    "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
    "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
    "node_dense": {"units": 128, "use_bias": True, "activation": "linear"},
    "edge_activation": {"activation": "relu"},
    "verbose": 10,
    "depthDMPNN": 4,
    "depthGIN": 4,
    "dropoutDMPNN": {"rate": 0.15},
    "dropoutGIN": {"rate": 0.15},
    "output_embedding": "graph",
    "node_pooling_kwargs": {"pooling_method": "mean"},
    "output_to_tensor": None,  # deprecated
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "linear"},
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
               input_graph_embedding: dict = None,
               pooling_args: dict = None,
               edge_initialize: dict = None,
               edge_dense: dict = None,
               node_dense: dict = None,
               edge_activation: dict = None,
               dropoutDMPNN: dict = None,
               dropoutGIN: dict = None,
               depthDMPNN: int = None,
               depthGIN: int = None,
               gin_args: dict = None,
               gin_mlp: dict = None,
               last_mlp: dict = None,
               verbose: int = None,
               node_pooling_kwargs: dict = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_tensor_type: str = None,
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    r"""Make `DGIN <https://pubmed.ncbi.nlm.nih.gov/34684766/>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DGIN.model_default` .

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, edges, edge_indices, reverse_indices, (graph_state), ...]`
    with '...' indicating mask or id tensors following the template below.
    Here, reverse indices are in place of angle indices and refer to edges. The graph state is optional and controlled
    by `use_graph_state` parameter.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        name (str): Name of the model. Should be "DGIN".
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        input_graph_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`AggregateLocalEdges` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge embedding.
        node_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for node embedding.
        edge_activation (dict): Edge Activation after skip connection.
        depthDMPNN (int): Number of graph embedding units or depth of the DMPNN subnetwork.
        depthGIN (int): Number of graph embedding units or depth of the GIN subnetwork.
        dropoutDMPNN (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        dropoutGIN (float): dropout rate.
        gin_args (dict): Kwargs unpacked in :obj:`GIN_D` convolutional unit.
        gin_mlp (dict): Kwargs unpacked in :obj:`MLP` for GIN layer.
        last_mlp (dict): Kwargs unpacked in last :obj:`MLP` .
        verbose (int): Level for print information.
        use_graph_state (bool): Whether to use graph state information. Default is False.
        node_pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): WDeprecated in favour of `output_tensor_type` .
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Kwargs for scaling layer, if scaling layer is to be used.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    di = template_cast_list_input(
        model_inputs, input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        index_assignment=[None, None, 0, 2] + ([None] if use_graph_state else []),
        mask_assignment=[0, 1, 1, 2] + ([None] if use_graph_state else [])
    )

    if use_graph_state:
        n, ed, edi, e_pairs, gs, batch_id_node, batch_id_edge, _, node_id, edge_id, _, count_nodes, count_edges, _ = di
    else:
        n, ed, edi, e_pairs, batch_id_node, batch_id_edge, _, node_id, edge_id, _, count_nodes, count_edges, _ = di
        gs = None

    # Wrapping disjoint model.
    out = model_disjoint(
        [n, ed, edi, batch_id_node, e_pairs, count_nodes, gs],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        use_graph_embedding=False if not use_graph_state else (
                "int" in inputs[4]['dtype']) if input_graph_embedding is not None else False,
        use_graph_state=use_graph_state,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        input_graph_embedding=input_graph_embedding,
        edge_initialize=edge_initialize,
        edge_activation=edge_activation,
        edge_dense=edge_dense,
        depthDMPNN=depthDMPNN,
        dropoutDMPNN=dropoutDMPNN,
        pooling_args=pooling_args,
        gin_mlp=gin_mlp,
        depthGIN=depthGIN,
        gin_args=gin_args,
        output_embedding=output_embedding,
        node_pooling_kwargs=node_pooling_kwargs,
        last_mlp=last_mlp,
        dropoutGIN=dropoutGIN,
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
    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
