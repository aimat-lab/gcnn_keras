import tensorflow.keras as ks
import pprint

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn.utils.models import generate_node_embedding, update_model_kwargs_logic, generate_edge_embedding


# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn


def make_model(**kwargs):
    """Make GCN model.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: Un-compiled GCN model.
    """
    model_args = kwargs
    model_default = {'name': "GCN",
                     'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                     'input_embedding': {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                         "edge_attributes": {"input_dim": 10, "output_dim": 64}},
                     'output_embedding': 'graph',
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'sum',
                                  "is_sorted": False, "has_unconnected": True},
                     'depth': 3, 'verbose': 1
                     }
    m = update_model_kwargs_logic(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO:kgcnn: Updated functional make model kwargs:")
        pprint.pprint(m)

    # Update model parameter
    inputs = m['inputs']
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    depth = m['depth']
    input_node_shape = inputs[0]['shape']
    input_edge_shape = inputs[1]['shape']
    gcn_args = m['gcn_args']

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # Embedding, if no feature dimension
    n = generate_node_embedding(node_input, input_node_shape, input_embedding[inputs[0]['name']])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding[inputs[1]['name']])
    edi = edge_index_input

    # Model
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # Map to units
    for i in range(0, depth):
        n = GCN(**gcn_args)([n, ed, edi])

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes()(n)  # will return tensor
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = n
        out = MLP(**output_mlp)(out)
        out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(
            out)  # no ragged for distribution supported atm
    else:
        raise ValueError("Unsupported graph embedding for `GCN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)
    return model


def make_model_node_weights(**kwargs):
    """Make GCN model.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: Un-compiled GCN model.
    """
    model_args = kwargs
    model_default = {'name': "GCN_weighted",
                     'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                                {'shape': (None, 1), 'name': "node_weights", 'dtype': 'float32', 'ragged': True}],
                     'input_embedding': {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                         "edge_attributes": {"input_dim": 10, "output_dim": 64}},
                     'output_embedding': 'graph',
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'sum'},
                     'depth': 3, 'verbose': 1
                     }
    m = update_model_kwargs_logic(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # Local variables for model args
    inputs = m['inputs']
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    depth = m['depth']
    input_node_shape = inputs[0]['shape']
    input_edge_shape = inputs[1]['shape']
    gcn_args = m['gcn_args']

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    node_weights_input = ks.layers.Input(**inputs[3])

    # Embedding, if no feature dimension
    n = generate_node_embedding(node_input, input_node_shape, input_embedding[inputs[0]['name']])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding[inputs[1]['name']])
    edi = edge_index_input
    nw = node_weights_input

    # Model
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # Map to units
    for i in range(0, depth):
        n = GCN(**gcn_args)([n, ed, edi])

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingWeightedNodes()([n, nw])  # will return tensor
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = n
        out = MLP(**output_mlp)(out)
        out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(
            out)  # no ragged for distribution supported atm
    else:
        raise ValueError("Unsupported graph embedding for `GCN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, node_weights_input], outputs=out)
    return model
