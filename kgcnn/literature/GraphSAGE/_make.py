import keras_core as ks
from kgcnn.layers.gather import GatherNodesOutgoing
from keras_core.layers import Concatenate
from kgcnn.layers.modules import Embedding
from kgcnn.layers.casting import (CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
                                  CastDisjointToGraphState, CastDisjointToBatchedAttributes)
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.aggr import AggregateLocalEdgesLSTM, AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.models.utils import update_model_kwargs
from kgcnn.layers.scale import get as get_scaler
from keras_core.backend import backend as backend_to_use

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023-09-18"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'GraphSAGE' is not supported." % backend_to_use())

# Implementation of GraphSAGE in `keras` from paper:
# Inductive Representation Learning on Large Graphs
# by William L. Hamilton and Rex Ying and Jure Leskovec
# http://arxiv.org/abs/1706.02216


model_default = {
    'name': "GraphSAGE",
    'inputs': [
        {"shape": (None, ), "name": "node_attributes", "dtype": "float32"},
        {"shape": (None, ), "name": "edge_attributes", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "cast_disjoint_kwargs": {},
    "input_node_embedding":  {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
    'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
    'edge_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
    'pooling_args': {'pooling_method': "scatter_mean"}, 'gather_args': {},
    'concat_args': {"axis": -1},
    'use_edge_features': True, 'pooling_nodes_args': {'pooling_method': "scatter_mean"},
    'depth': 3, 'verbose': 10,
    'output_embedding': 'graph', "output_to_tensor": True,
    'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ['relu', 'relu', 'sigmoid']},
    "output_scaling": None,
}


@update_model_kwargs(model_default, update_recursive=0)
def make_model(inputs: list = None,
               cast_disjoint_kwargs: dict = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               node_mlp_args: dict = None,
               edge_mlp_args: dict = None,
               pooling_args: dict = None,
               pooling_nodes_args: dict = None,
               gather_args: dict = None,
               concat_args: dict = None,
               use_edge_features: bool = None,
               depth: int = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    r"""Make `GraphSAGE <http://arxiv.org/abs/1706.02216>`__ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.GraphSAGE.model_default` .

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, total_nodes, total_edges]`

            - node_attributes (Tensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (Tensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (Tensor): Index list for edges of shape `(batch, None, 2)`.
            - total_nodes(Tensor, optional): Number of Nodes in graph if not same sized graphs of shape `(batch, )` .
            - total_edges(Tensor, optional): Number of Edges in graph if not same sized graphs of shape `(batch, )` .

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
        input_node_embedding (dict): Dictionary of arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        node_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for node updates.
        edge_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge updates.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingLocalMessages` layer.
        pooling_nodes_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        gather_args (dict): Dictionary of layer arguments unpacked in :obj:`GatherNodes` layer.
        concat_args (dict): Dictionary of layer arguments unpacked in :obj:`Concatenate` layer.
        use_edge_features (bool): Whether to add edge features in message step.
        depth (int): Number of graph embedding units or depth of the network.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.

    Returns:
        :obj:`ks.models.Model`
    """
    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    batched_nodes, batched_edges, batched_indices, total_nodes, total_edges = model_inputs
    n, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])
    ed, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_edges, total_edges])

    # Embedding, if no feature dimension
    if len(inputs[0]['shape']) < 2:
        n = Embedding(**input_node_embedding)(n)
    if len(inputs[1]['shape']) < 2:
        ed = Embedding(**input_edge_embedding)(ed)

    for i in range(0, depth):

        eu = GatherNodesOutgoing(**gather_args)([n, disjoint_indices])
        if use_edge_features:
            eu = Concatenate(**concat_args)([eu, ed])

        eu = GraphMLP(**edge_mlp_args)([eu, batch_id_edge, count_edges])

        # Pool message
        if pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            nu = AggregateLocalEdgesLSTM(**pooling_args)([n, eu, disjoint_indices])
        else:
            nu = AggregateLocalEdges(**pooling_args)([n, eu, disjoint_indices])  # Summing for each node connection

        nu = Concatenate(**concat_args)([n, nu])  # Concatenate node features with new edge updates

        n = GraphMLP(**node_mlp_args)([nu, batch_id_node, count_nodes])

        n = GraphLayerNormalization()([n, batch_id_node, count_nodes])

    # Regression layer on output
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
        out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)

    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
        if output_to_tensor:
            out = CastDisjointToBatchedAttributes(**cast_disjoint_kwargs)([batched_nodes, out, batch_id_node, node_id])
        else:
            out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)

    else:
        raise ValueError("Unsupported output embedding for `GraphSAGE`")

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


def model_disjoint(
        inputs,
        use_node_embedding: bool = None,
        use_edge_embedding: bool = None,
        input_node_embedding: dict = None,
        input_edge_embedding: dict = None,
        node_mlp_args: dict = None,
        edge_mlp_args: dict = None,
        pooling_args: dict = None,
        pooling_nodes_args: dict = None,
        gather_args: dict = None,
        concat_args: dict = None,
        use_edge_features: bool = None,
        depth: int = None,
        output_embedding: str = None,
        output_mlp: dict = None,
    ):
    n, ed, disjoint_indices, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    for i in range(0, depth):

        eu = GatherNodesOutgoing(**gather_args)([n, disjoint_indices])
        if use_edge_features:
            eu = Concatenate(**concat_args)([eu, ed])

        eu = GraphMLP(**edge_mlp_args)([eu, batch_id_edge, count_edges])

        # Pool message
        if pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            nu = AggregateLocalEdgesLSTM(**pooling_args)([n, eu, disjoint_indices])
        else:
            nu = AggregateLocalEdges(**pooling_args)([n, eu, disjoint_indices])  # Summing for each node connection

        nu = Concatenate(**concat_args)([n, nu])  # Concatenate node features with new edge updates

        n = GraphMLP(**node_mlp_args)([nu, batch_id_node, count_nodes])

        n = GraphLayerNormalization()([n, batch_id_node, count_nodes])

    # Regression layer on output
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for `GraphSAGE`")
    return out