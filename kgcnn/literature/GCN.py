import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv import GCN
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn.ops.models import generate_node_embedding, update_model_args, generate_edge_embedding


# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn


def make_gcn(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedding: dict = None,
        # Output
        output_embedding: dict = None,
        output_mlp: dict = None,
        # Model specific
        depth=3,
        gcn_args: dict = None
):
    """Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedding (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"nodes": {"input_dim": 95, "output_dim": 64},
            "edges": {"input_dim": 10, "output_dim": 64},
            "state": {"input_dim": 100, "output_dim": 64}}.
        output_embedding (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_tensor_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classification layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int, optional): Number of convolutions. Defaults to 3.
        gcn_args (dict): Dictionary of arguments for the GCN convolutional unit. Defaults to
            {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum',
            "is_sorted": False, "has_unconnected": "True"}.

    Returns:
        tf.keras.models.Model: Un-compiled GCN model.
    """

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)
    # Make default args
    model_default = {'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 10, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'masked'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'sum',
                                  "is_sorted": False, "has_unconnected": True}
                     }

    # Update model parameter
    input_embedding = update_model_args(model_default['input_embedding'], input_embedding)
    output_embedding = update_model_args(model_default['output_embedding'], output_embedding)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    gcn_args = update_model_args(model_default['gcn_args'], gcn_args)

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    edi = edge_index_input

    # Map to units
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)

    # n-Layer Step
    for i in range(0, depth):
        n = GCN(**gcn_args)([n, ed, edi])

    if output_embedding["output_mode"] == "graph":
        out = PoolingNodes()(n)  # will return tensor
        out = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        out = MLP(**output_mlp)(out)
        out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(
            out)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)

    return model


def make_gcn_node_weights(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedding: dict = None,
        # Output
        output_embedding: dict = None,
        output_mlp: dict = None,
        # Model specific
        depth=3,
        gcn_args: dict = None
):
    """Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedding (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"nodes": {"input_dim": 95, "output_dim": 64},
            "edges": {"input_dim": 10, "output_dim": 64},
            "state": {"input_dim": 100, "output_dim": 64}}.
        output_embedding (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_tensor_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classification layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int, optional): Number of convolutions. Defaults to 3.
        gcn_args (dict): Dictionary of arguments for the GCN convolutional unit. Defaults to
            {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum'}.

    Returns:
        tf.keras.models.Model: Un-compiled GCN model.
    """

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)
    # Make default args
    model_default = {'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 10, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'masked'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'sum'}
                     }

    # Update model parameter
    input_embedding = update_model_args(model_default['input_embedding'], input_embedding)
    output_embedding = update_model_args(model_default['output_embedding'], output_embedding)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    gcn_args = update_model_args(model_default['gcn_args'], gcn_args)

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
    node_weights_input = ks.layers.Input(shape=(None, 1), name='node_weights', dtype="float32", ragged=True)
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    edi = edge_index_input
    nw = node_weights_input

    # Map to units
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)

    # n-Layer Step
    for i in range(0, depth):
        n = GCN(**gcn_args)([n, ed, edi])

    if output_embedding["output_mode"] == "graph":
        out = PoolingWeightedNodes()([n, nw])  # will return tensor
        out = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        out = MLP(**output_mlp)(out)
        out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(
            out)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, node_weights_input], outputs=out)

    return model
