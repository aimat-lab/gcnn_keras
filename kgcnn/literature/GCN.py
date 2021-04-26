import tensorflow.keras as ks

import kgcnn.layers.disjoint.conv
import kgcnn.layers.disjoint.pooling
import kgcnn.layers.ragged.conv
import kgcnn.layers.sparse.conv
from kgcnn.layers.disjoint.casting import CastRaggedToValues, CastValuesToRagged, CastRaggedToDisjoint
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.layers.ragged.mlp import MLPRagged
from kgcnn.layers.ragged.pooling import PoolingNodes
from kgcnn.layers.sparse.casting import CastRaggedToDisjointSparseAdjacency
from kgcnn.ops.models import generate_standard_graph_input, update_model_args


# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn


def make_gcn(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific
        depth=3,
        gcn_args: dict = None
):
    """
    Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
            "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
            "input_type": 'ragged'}.
        output_embedd (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classifcation layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int, optional): Number of convolutions. Defaults to 3.
        gcn_args (dict): Dictionary of arguments for the GCN convolutional unit. Defaults to
            {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum',
            "is_sorted": False, "has_unconnected": "True"}.

    Returns:
        model (tf.keras.models.Model): uncompiled model.

    """

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)
    # Make default args
    model_default = {'input_embedd': {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
                                      "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
                                      "input_type": 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum',
                                  "is_sorted": False, "has_unconnected": True}
                     }

    # Update model parameter
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    gcn_args = update_model_args(model_default['gcn_args'], gcn_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                                     input_edge_shape,
                                                                                                     None,
                                                                                                     **input_embedd)
    # Map to units
    n = kgcnn.layers.ragged.conv.DenseRagged(gcn_args["units"], use_bias=True, activation='linear')(n)
    ed = ed
    edi = edge_index_input
    # edi = ChangeIndexing()([n, edge_index_input])

    # n-Layer Step
    for i in range(0, depth):
        n = kgcnn.layers.ragged.conv.GCN(**gcn_args)([n, ed, edi])

    if output_embedd["output_mode"] == "graph":
        out = PoolingNodes()(n)  # will return tensor
        out = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        out = MLPRagged(**output_mlp)(out)
        out = CastRaggedToDense()(out)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)

    return model


def make_gcn_disjoint(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific
        depth=3,
        gcn_args: dict = None
):
    """
    Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
            "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
            "input_type": 'ragged'}.
        output_embedd (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classifcation layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int, optional): Number of convolutions. Defaults to 3.
        gcn_args (dict): Dictionary of arguments for the GCN convolutional unit. Defaults to
            {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum',
            "is_sorted": False, "has_unconnected": "True"}.

    Returns:
        model (tf.keras.models.Model): uncompiled model.

    """

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)
    # Make default args
    model_default = {'input_embedd': {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
                                      "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
                                      "input_type": 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum',
                                  "is_sorted": False, "has_unconnected": True}
                     }

    # Update model parameter
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    gcn_args = update_model_args(model_default['gcn_args'], gcn_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                                     input_edge_shape,
                                                                                                     None,
                                                                                                     **input_embedd)
    # Map to units
    n, node_len, ed, edge_len, edi = CastRaggedToDisjoint()([n, ed, edge_index_input])
    n = ks.layers.Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # To match units

    # n-Layer Step
    for i in range(0, depth):
        n = kgcnn.layers.disjoint.conv.GCN(**gcn_args)([n, node_len, ed, edge_len, edi])

    if output_embedd["output_mode"] == "graph":
        out = kgcnn.layers.disjoint.pooling.PoolingNodes()([n, node_len])  # will return tensor
        out = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = CastValuesToRagged()([n, node_len])
        out = MLPRagged(**output_mlp)(out)
        out = CastRaggedToDense()(out)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)

    return model


def make_gcn_sparse(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific
        depth=3,
        gcn_args: dict = None
):
    """
    Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
            "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
            "input_type": 'ragged'}.
        output_embedd (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classifcation layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int, optional): Number of convolutions. Defaults to 3.
        gcn_args (dict): Dictionary of arguments for the GCN convolutional unit. Defaults to
            {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum',
            "is_sorted": False, "has_unconnected": "True"}.

    Returns:
        model (tf.keras.models.Model): uncompiled model.

    """

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)
    # Make default args
    model_default = {'input_embedd': {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
                                      "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
                                      "input_type": 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'segment_sum',
                                  "is_sorted": False, "has_unconnected": True}
                     }

    # Update model parameter
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    gcn_args = update_model_args(model_default['gcn_args'], gcn_args)

    # Make input embedding, if no feature dimension
    node_input, nragged, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(
        input_node_shape,
        input_edge_shape,
        None,
        **input_embedd)
    # Map to units
    n, node_len = CastRaggedToValues()(nragged)
    n = ks.layers.Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # To match units
    adj = CastRaggedToDisjointSparseAdjacency()([nragged, edge_index_input, ed])

    # n-Layer Step
    for i in range(0, depth):
        n = kgcnn.layers.sparse.conv.GCN(**gcn_args)([n, adj])

    if output_embedd["output_mode"] == "graph":
        out = kgcnn.layers.disjoint.pooling.PoolingNodes()([n, node_len])  # will return tensor
        out = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = CastValuesToRagged()([n, node_len])
        out = MLPRagged(**output_mlp)(out)
        out = CastRaggedToDense()(out)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)

    return model
