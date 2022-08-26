import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.modules import LazyConcatenate, OptionalInputEmbedding
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes, PoolingLocalMessages, PoolingLocalEdgesLSTM
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of GraphSAGE in `tf.keras` from paper:
# Inductive Representation Learning on Large Graphs
# by William L. Hamilton and Rex Ying and Jure Leskovec
# http://arxiv.org/abs/1706.02216


hyper_model_default = {
    'name': "GraphSAGE",
    'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
    'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
    'edge_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
    'pooling_args': {'pooling_method': "segment_mean"}, 'gather_args': {},
    'concat_args': {"axis": -1},
    'use_edge_features': True, 'pooling_nodes_args': {'pooling_method': "mean"},
    'depth': 3, 'verbose': 10,
    'output_embedding': 'graph', "output_to_tensor": True,
    'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ['relu', 'relu', 'sigmoid']}
}


@update_model_kwargs(hyper_model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
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
               output_mlp: dict = None
               ):
    r"""Make `GraphSAGE <http://arxiv.org/abs/1706.02216>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GraphSAGE.hyper_model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        node_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for node updates.
        edge_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge updates.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingLocalMessages` layer.
        pooling_nodes_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        gather_args (dict): Dictionary of layer arguments unpacked in :obj:`GatherNodes` layer.
        concat_args (dict): Dictionary of layer arguments unpacked in :obj:`LazyConcatenate` layer.
        use_edge_features (bool): Whether to add edge features in message step.
        depth (int): Number of graph embedding units or depth of the network.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # Make input embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu = GatherNodesOutgoing(**gather_args)([n, edi])
        if use_edge_features:
            eu = LazyConcatenate(**concat_args)([eu, ed])

        eu = GraphMLP(**edge_mlp_args)(eu)
        # Pool message
        if pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            nu = PoolingLocalEdgesLSTM(**pooling_args)([n, eu, edi])
        else:
            nu = PoolingLocalMessages(**pooling_args)([n, eu, edi])  # Summing for each node connection

        nu = LazyConcatenate(**concat_args)([n, nu])  # LazyConcatenate node features with new edge updates

        n = GraphMLP(**node_mlp_args)(nu)
        n = GraphLayerNormalization()(n)  # Normalize

    # Regression layer on output
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)(n)
        out = ks.layers.Flatten()(out)  # will be tensor
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for `GraphSAGE`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)
    return model
