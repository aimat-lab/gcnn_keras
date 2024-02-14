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
    raise NotImplementedError("Backend '%s' for model 'DMPNN' is not supported." % backend_to_use())

# Implementation of DMPNN in `keras` from paper:
# Analyzing Learned Molecular Representations for Property Prediction
# by Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao,
# Angel Guzman-Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, Andrew Palmer,
# Volker Settels, Tommi Jaakkola, Klavs Jensen, and Regina Barzilay
# https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237


model_default = {
    "name": "DMPNN",
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
    "pooling_args": {"pooling_method": "scatter_sum"},
    "use_graph_state": False,
    "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
    "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
    "edge_activation": {"activation": "relu"},
    "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
    "verbose": 10, "depth": 5, "dropout": {"rate": 0.1},
    "output_embedding": "graph",
    "output_to_tensor": None,  # deprecated
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                   "activation": ["relu", "relu", "linear"]},
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
               edge_activation: dict = None,
               node_dense: dict = None,
               dropout: dict = None,
               depth: int = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_tensor_type: str = None,
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    r"""Make `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DMPNN.model_default`.

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
        name (str): Name of the model. Should be "DMPNN".
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        input_graph_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            :obj:`AggregateLocalEdges` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge embedding.
        edge_activation (dict): Edge Activation after skip connection.
        node_dense (dict): Dense kwargs for node embedding layer.
        depth (int): Number of graph embedding units or depth of the network.
        dropout (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        verbose (int): Level for print information.
        use_graph_state (bool): Whether to use graph state information. Default is False.
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
        model_inputs, input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0,1,1,2] + ([None] if use_graph_state else []),
        index_assignment=[None, None, 0, 2] + ([None] if use_graph_state else [])
    )

    if use_graph_state:
        n, ed, edi, e_pairs, gs, batch_id_node, batch_id_edge, _, node_id, edge_id, _, count_nodes, count_edges, _ = di
    else:
        n, ed, edi, e_pairs, batch_id_node, batch_id_edge, _,  node_id, edge_id, _, count_nodes, count_edges, _ = di
        gs = None

    # Wrapping disjoint model.
    out = model_disjoint(
        [n, ed, edi, batch_id_node, e_pairs, count_nodes, gs],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        use_graph_embedding=False if not use_graph_state else (
                "int" in inputs[4]['dtype']) if input_graph_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        input_graph_embedding=input_graph_embedding,
        pooling_args=pooling_args, edge_initialize=edge_initialize, edge_activation=edge_activation,
        node_dense=node_dense, dropout=dropout, depth=depth, use_graph_state=use_graph_state,
        output_embedding=output_embedding, output_mlp=output_mlp, edge_dense=edge_dense
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
