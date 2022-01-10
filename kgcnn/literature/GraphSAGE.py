import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.modules import LazyConcatenate, OptionalInputEmbedding
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes, PoolingLocalMessages, PoolingLocalEdgesLSTM
from kgcnn.utils.models import update_model_kwargs

# 'Inductive Representation Learning on Large Graphs'
# William L. Hamilton and Rex Ying and Jure Leskovec
# http://arxiv.org/abs/1706.02216


hyper_model_default = {'name': "GraphSAGE",
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
                       'output_embedding': 'graph',
                       'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                      "activation": ['relu', 'relu', 'sigmoid']}
                       }


@update_model_kwargs(hyper_model_default)
def make_model(inputs=None,
               input_embedding=None,
               node_mlp_args=None,
               edge_mlp_args=None,
               pooling_args=None,
               pooling_nodes_args=None,
               gather_args=None,
               concat_args=None,
               use_edge_features=None,
               depth=None,
               name=None,
               verbose=None,
               output_embedding=None,
               output_mlp=None
               ):
    """Make GraphSAGE graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        node_mlp_args (dict): Dictionary of layer arguments unpacked in `MLP` layer for node updates.
        edge_mlp_args (dict): Dictionary of layer arguments unpacked in `MLP` layer for edge updates.
        pooling_args (dict): Dictionary of layer arguments unpacked in `PoolingLocalMessages` layer.
        pooling_nodes_args (dict): Dictionary of layer arguments unpacked in `PoolingNodes` layer.
        gather_args (dict): Dictionary of layer arguments unpacked in `GatherNodes` layer.
        concat_args (dict): Dictionary of layer arguments unpacked in `LazyConcatenate` layer.
        use_edge_features (bool): Whether to add edge features in message step.
        depth (int): Number of graph embedding units or depth of the network.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        tf.keras.models.Model
    """

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
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
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `GraphSAGE`")

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model
