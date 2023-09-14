import keras_core as ks
from kgcnn.layers_core.attention import AttentionHeadGAT
from kgcnn.layers_core.modules import Embedding
from kgcnn.layers_core.casting import CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint, CastDisjointToGraph
from keras_core.layers import Concatenate, Dense, Average, Activation
from kgcnn.layers_core.mlp import MLP
from kgcnn.layers_core.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from keras_core.backend import backend as backend_to_use
from kgcnn.ops_core.activ import *

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023.09.08"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'GCN' is not supported." % backend_to_use())

# Implementation of GAT in `keras` from paper:
# Graph Attention Networks
# by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (2018)
# https://arxiv.org/abs/1710.10903


model_default = {
    "name": "GAT",
    "inputs": [
        {"shape": (None, ), "name": "node_attributes", "dtype": "float32"},
        {"shape": (None, ), "name": "edge_attributes", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "cast_disjoint_kwargs": {},
    "input_node_embedding":  {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
    "attention_args": {"units": 32, "use_final_activation": False, "use_edge_features": True,
                       "has_self_loops": True, "activation": "kgcnn>leaky_relu", "use_bias": True},
    "pooling_nodes_args": {"pooling_method": "scatter_mean"},
    "depth": 3, "attention_heads_num": 5,
    "attention_heads_concat": False,
    "verbose": 10,
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
               attention_args: dict = None,
               pooling_nodes_args: dict = None,
               depth: int = None,
               attention_heads_num: int = None,
               attention_heads_concat: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `GAT <https://arxiv.org/abs/1710.10903>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GAT.model_default`.

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
        Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
        input_node_embedding (dict): Dictionary of arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`AttentionHeadGAT` layer.
        pooling_nodes_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        depth (int): Number of graph embedding units or depth of the network.
        attention_heads_num (int): Number of attention heads to use.
        attention_heads_concat (bool): Whether to concat attention heads, or simply average heads.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
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

    # Model
    nk = Dense(units=attention_args["units"], activation="linear")(n)
    for i in range(0, depth):
        heads = [AttentionHeadGAT(**attention_args)([nk, ed, disjoint_indices]) for _ in range(attention_heads_num)]
        if attention_heads_concat:
            nk = Concatenate(axis=-1)(heads)
        else:
            nk = Average()(heads)
            nk = Activation(activation=attention_args["activation"])(nk)
    n = nk

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
        out = CastDisjointToGraph(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        out = MLP(**output_mlp)(n)
    else:
        raise ValueError("Unsupported output embedding for `GAT` .")

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    return model
