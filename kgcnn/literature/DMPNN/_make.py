import keras_core as ks
from kgcnn.layers.casting import (CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
                                  CastDisjointToBatchedGraphState, CastDisjointToBatchedAttributes, CastBatchedGraphStateToDisjoint)
from kgcnn.layers.scale import get as get_scaler
from kgcnn.models.utils import update_model_kwargs
from keras_core.backend import backend as backend_to_use
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
        {"shape": (None,), "name": "node_attributes", "dtype": "float32"},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "cast_disjoint_kwargs": {},
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
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                   "activation": ["relu", "relu", "linear"]},
    "output_scaling": None
}


@update_model_kwargs(model_default, update_recursive=0)
def make_model(name: str = None,
               inputs: list = None,
               cast_disjoint_kwargs: dict = None,
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
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    r"""Make `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DMPNN.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_pairs, total_nodes, total_edges]` or
        `[node_attributes, edge_attributes, edge_indices, edge_pairs, total_nodes, total_edges, state_attributes]`
        if `use_graph_state=True` .

            - node_attributes (Tensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (Tensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (Tensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_pairs (Tensor): Pair mappings for reverse edge for each edge `(batch, None, 1)`.
            - state_attributes (Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.
            - total_nodes(Tensor): Number of Nodes in graph of shape `(batch, )` .
            - total_edges(Tensor): Number of Edges in graph of shape `(batch, )` .

    Outputs:
        Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "DMPNN".
        inputs (list): List of dictionaries unpacked in :obj:`keras.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
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
        output_to_tensor (bool): Whether to cast model output to :obj:`Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Kwargs for scaling layer, if scaling layer is to be used.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    batched_nodes, batched_edges, batched_indices, batched_reverse, total_nodes, total_edges = model_inputs[:6]

    # Casting
    graph_state = CastBatchedGraphStateToDisjoint(**cast_disjoint_kwargs)(model_inputs[7]) if use_graph_state else None
    n, edi, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])
    ed, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_edges, total_edges])
    _, ed_pairs, _, _, _, _, _, _ = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_indices, batched_reverse, count_edges, count_edges])

    # Wrapping disjoint model.
    out = model_disjoint(
        [n, ed, edi, batch_id_node, ed_pairs, count_nodes, graph_state],
        use_node_embedding=len(inputs[0]['shape']) < 2, use_edge_embedding=len(inputs[1]['shape']) < 2,
        use_graph_embedding=len(inputs[7]["shape"]) < 1 if use_graph_state else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding, input_graph_embedding=input_graph_embedding,
        pooling_args=pooling_args, edge_initialize=edge_initialize, edge_activation=edge_activation,
        node_dense=node_dense, dropout=dropout, depth=depth, use_graph_state=use_graph_state,
        output_embedding=output_embedding, output_mlp=output_mlp, edge_dense=edge_dense
    )

    if output_embedding == 'graph':
        out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        if output_to_tensor:
            out = CastDisjointToBatchedAttributes(**cast_disjoint_kwargs)([batched_nodes, out, batch_id_node, node_id])
        else:
            out = CastDisjointToBatchedGraphState(**cast_disjoint_kwargs)(out)

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        out = scaler(out)

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)
    return model
