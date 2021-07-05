import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Concatenate, LayerNormalization
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingNodes, PoolingLocalMessages, PoolingLocalEdgesLSTM
from kgcnn.utils.models import generate_node_embedding, update_model_args, generate_edge_embedding


# 'Inductive Representation Learning on Large Graphs'
# William L. Hamilton and Rex Ying and Jure Leskovec
# http://arxiv.org/abs/1706.02216

def make_graph_sage(  # Input
        input_node_shape,
        input_edge_shape,
        input_embedding: dict = None,
        # Output
        output_embedding: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        depth=3,
        use_edge_features=False,
        node_mlp_args: dict = None,
        edge_mlp_args: dict = None,
        pooling_args: dict = None
):
    """Generate GraphSAGE network.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedding (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"nodes": {"input_dim": 95, "output_dim": 64},
            "edges": {"input_dim": 5, "output_dim": 64},
            "state": {"input_dim": 100, "output_dim": 64}}.
        output_embedding (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_tensor_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classification layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int): Number of convolution layers. Default is 3.
        use_edge_features (bool): Whether to concatenate edges with nodes in aggregate. Default is False.
        node_mlp_args (dict): Dictionary of arguments for MLP for node update. Default is
            {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]}
        edge_mlp_args (dict): Dictionary of arguments for MLP for interaction update. Default is
            {"units": [100, 100, 100, 100, 50],
            "activation": ['relu', 'relu', 'relu', 'relu', "linear"]}
        pooling_args (dict): Dictionary for message pooling arguments. Default is
            {'pooling_method': "segment_mean"}

    Returns:
        tf.keras.models.Model: GraphSAGE model.
    """
    # default values
    model_default = {'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'edge_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'pooling_args': {'pooling_method': "segment_mean"}
                     }

    # Update default values
    input_embedding = update_model_args(model_default['input_embedding'], input_embedding)
    output_embedding = update_model_args(model_default['output_embedding'], output_embedding)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    node_mlp_args = update_model_args(model_default['node_mlp_args'], node_mlp_args)
    edge_mlp_args = update_model_args(model_default['edge_mlp_args'], edge_mlp_args)
    pooling_args = update_model_args(model_default['pooling_args'], pooling_args)
    pooling_nodes_args = {'pooling_method': "mean"}
    gather_args = {}
    concat_args = {"axis": -1}

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    edi = edge_index_input

    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu = GatherNodesOutgoing(**gather_args)([n, edi])
        if use_edge_features:
            eu = Concatenate(**concat_args)([eu, ed])

        eu = MLP(**edge_mlp_args)(eu)
        # Pool message
        if pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            nu = PoolingLocalEdgesLSTM(**pooling_args)([n, eu, edi])
        else:
            nu = PoolingLocalMessages(**pooling_args)([n, eu, edi])  # Summing for each node connection

        nu = Concatenate(**concat_args)([n, nu])  # Concatenate node features with new edge updates

        n = MLP(**node_mlp_args)(nu)
        n = LayerNormalization(axis=-1)(n)  # Normalize

    # Regression layer on output
    if output_embedding["output_mode"] == 'graph':
        out = PoolingNodes(**pooling_nodes_args)(n)
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be tensor
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model
