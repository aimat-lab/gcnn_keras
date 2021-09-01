import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn.utils.models import update_model_kwargs, generate_embedding

# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn

model_default = {'name': "GCN",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                     "edge": {"input_dim": 10, "output_dim": 64}},
                 'output_embedding': 'graph',
                 'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                "activation": ['relu', 'relu', 'sigmoid']},
                 'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'sum',
                              "is_sorted": False, "has_unconnected": True},
                 'depth': 3, 'verbose': 1
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               output_embedding=None,
               output_mlp=None,
               depth=None,
               gcn_args=None,
               **kwargs):
    """Make GCN graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.
        depth (int): Number of graph embedding units or depth of the network.
        gcn_args (dict): Dictionary of layer arguments unpacked in `GCN` convolutional layer.

    Returns:
        tf.keras.models.Model
    """

    input_node_shape = inputs[0]['shape']
    input_edge_shape = inputs[1]['shape']

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # Embedding, if no feature dimension
    n = generate_embedding(node_input, input_node_shape, input_embedding['node'])
    ed = generate_embedding(edge_input, input_edge_shape, input_embedding['edge'])
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


model_default_weighted = {'name': "GCN_weighted",
                          'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                     {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                     {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                                     {'shape': (None, 1), 'name': "node_weights", 'dtype': 'float32', 'ragged': True}],
                          'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                              "edge": {"input_dim": 10, "output_dim": 64}},
                          'output_embedding': 'graph',
                          'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                         "activation": ['relu', 'relu', 'sigmoid']},
                          'gcn_args': {"units": 100, "use_bias": True, "activation": 'relu', "pooling_method": 'sum'},
                          'depth': 3, 'verbose': 1
                          }


@update_model_kwargs(model_default_weighted)
def make_model_weighted(inputs=None,
                        input_embedding=None,
                        output_embedding=None,
                        output_mlp=None,
                        depth=None,
                        gcn_args=None,
                        **kwargs):
    """Make GCN model."""

    input_node_shape = inputs[0]['shape']
    input_edge_shape = inputs[1]['shape']

    if input_edge_shape[-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): ", input_edge_shape)

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    node_weights_input = ks.layers.Input(**inputs[3])

    # Embedding, if no feature dimension
    n = generate_embedding(node_input, input_node_shape, input_embedding['node'])
    ed = generate_embedding(edge_input, input_edge_shape, input_embedding['edge'])
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
