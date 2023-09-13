import keras_core as ks
from keras_core.layers import Dense
from kgcnn.layers_core.modules import Embedding
from kgcnn.layers_core.casting import CastBatchedIndicesToDisjoint, CastDisjointToGraph, CastBatchedAttributesToDisjoint
from kgcnn.layers_core.conv import GCN
from kgcnn.layers_core.mlp import MLP
from kgcnn.layers_core.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from keras_core.backend import backend as backend_to_use

# from keras_core.layers import Activation
# from kgcnn.layers_core.aggr import AggregateWeightedLocalEdges
# from kgcnn.layers_core.gather import GatherNodesOutgoing

# Keep track of model version from commit date in literature.
__kgcnn_model_version__ = "2023.09.30"

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
        {"shape": (None,), "name": "node_attributes", "dtype": "float32"},
        {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
    "gcn_args": {"units": 100, "use_bias": True, "activation": "relu", "pooling_method": "sum"},
    "depth": 3,
    "verbose": 10,
    "node_pooling_args": {"pooling_method": "scatter_sum"},
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]}
}


@update_model_kwargs(model_default, update_recursive=0)
def make_model(inputs: list = None,
               cast_disjoint_kwargs: dict = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               depth: int = None,
               gcn_args: dict = None,
               name: str = None,
               verbose: int = None,
               node_pooling_args: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None):
    r"""Make `GCN <https://arxiv.org/abs/1609.02907>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GCN.model_default`.

    Inputs:
        list: `[node_attributes, edge_weights, edge_indices, total_nodes, total_edges]`

            - node_attributes (Tensor): Node attributes of shape `(batch, N, F)` or `(batch, N)`
              using an embedding layer.
            - edge_weights (Tensor): Edge weights of shape `(batch, M, 1)` , that are entries of a scaled
              adjacency matrix.
            - edge_indices (Tensor): Index list for edges of shape `(batch, M, 2)` .
            - total_nodes(Tensor, optional): Number of Nodes in graph if not same sized graphs of shape `(batch, )` .
            - total_edges(Tensor, optional): Number of Edges in graph if not same sized graphs of shape `(batch, )` .

    Outputs:
        Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`ks.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gcn_args (dict): Dictionary of layer arguments unpacked in :obj:`GCN` convolutional layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`ks.models.Model`
    """
    if inputs[1]['shape'][-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): %s." % inputs[1]['shape'])

    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    batched_nodes, batched_edges, batched_indices, total_nodes, total_edges = model_inputs
    n, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])
    e, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_edges, total_edges])

    # Embedding, if no feature dimension
    if len(inputs[0]['shape']) < 2:
        n = Embedding(**input_node_embedding)(n)
    if len(inputs[1]['shape']) < 2:
        e = Embedding(**input_edge_embedding)(e)

    # Model
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # Map to units
    for i in range(0, depth):
        n = GCN(**gcn_args)([n, e, disjoint_indices])

        # # Equivalent as:
        # no = Dense(gcn_args["units"], activation="linear")(n)
        # no = GatherNodesOutgoing()([no, disjoint_indices])
        # nu = AggregateWeightedLocalEdges()([n, no, disjoint_indices, e])
        # n = Activation(gcn_args["activation"])(nu)

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])  # will return tensor
        out = MLP(**output_mlp)(out)
        out = CastDisjointToGraph(**cast_disjoint_kwargs)(out)
    elif output_embedding == "node":
        out = n
        out = MLP(**output_mlp)(out)
    else:
        raise ValueError("Unsupported output embedding for `GCN` .")

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __kgcnn_model_version__
    return model
