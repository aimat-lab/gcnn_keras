import tensorflow.keras as ks
import pprint
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.connect import AdjacencyPower
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Dense, Activation, Add
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingNodes, PoolingLocalEdges
from kgcnn.layers.pool.topk import PoolingTopK, UnPoolingTopK
from kgcnn.utils.models import generate_embedding, update_model_kwargs

# Graph U-Nets
# by Hongyang Gao, Shuiwang Ji
# https://arxiv.org/pdf/1905.05178.pdf

hyper_model_default = {'name': "Unet",
                       'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                       'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                           "edge": {"input_dim": 5, "output_dim": 64}},
                       'output_embedding': 'graph',
                       'output_mlp': {"use_bias": [True, False], "units": [25, 1], "activation": ['relu', 'sigmoid']},
                       'hidden_dim': {'units': 32, 'use_bias': True, 'activation': 'linear'},
                       'top_k_args': {'k': 0.3, 'kernel_initializer': 'ones'},
                       'activation': 'relu',
                       'use_reconnect': True,
                       'depth': 4,
                       'pooling_args': {"pooling_method": 'segment_mean'},
                       'gather_args': {"node_indexing": 'sample'},
                       'verbose': 1
                       }


@update_model_kwargs(hyper_model_default)
def make_model(inputs=None,
               input_embedding=None,
               output_embedding=None,
               output_mlp=None,
               pooling_args=None,
               gather_args=None,
               top_k_args=None,
               depth=None,
               use_reconnect=None,
               hidden_dim=None,
               activation=None, **kwargs):
    """Make Graph U-Net."""

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # embedding, if no feature dimension
    n = generate_embedding(node_input, inputs[0]['shape'], input_embedding['node'])
    ed = generate_embedding(edge_input, inputs[1]['shape'], input_embedding['edge'])
    edi = edge_index_input

    # Model
    n = Dense(**hidden_dim)(n)
    in_graph = [n, ed, edi]
    graph_list = [in_graph]
    map_list = []

    # U Down
    i_graph = in_graph
    for i in range(0, depth):

        n, ed, edi = i_graph
        # GCN layer
        eu = GatherNodesOutgoing(**gather_args)([n, edi])
        eu = Dense(**hidden_dim)(eu)
        nu = PoolingLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connection
        n = Activation(activation=activation)(nu)

        if use_reconnect:
            ed, edi = AdjacencyPower(n=2)([n, ed, edi])

        # Pooling
        i_graph, i_map = PoolingTopK(**top_k_args)([n, ed, edi])

        graph_list.append(i_graph)
        map_list.append(i_map)

    # U Up
    ui_graph = i_graph
    for i in range(depth, 0, -1):
        o_graph = graph_list[i - 1]
        i_map = map_list[i - 1]
        ui_graph = UnPoolingTopK()(o_graph + i_map + ui_graph)

        n, ed, edi = ui_graph
        # skip connection
        n = Add()([n, o_graph[0]])
        # GCN
        eu = GatherNodesOutgoing(**gather_args)([n, edi])
        eu = Dense(**hidden_dim)(eu)
        nu = PoolingLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connection
        n = Activation(activation=activation)(nu)

        ui_graph = [n, ed, edi]

    # Output embedding choice
    n = ui_graph[0]
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_args)(n)
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    elif output_embedding == 'node':
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `Unet`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model


hyper_model_dataset = {"MUTAG": {'model': {
    'name': "Unet",
    'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 1), 'name': "edge_labels", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
    'input_embedding': {"node": {"input_dim": 60, "output_dim": 128},
                        "edge": {"input_dim": 5, "output_dim": 5}},
    'output_embedding': 'graph',
    'output_mlp': {"use_bias": [True, False], "units": [25, 1], "activation": ['relu', 'sigmoid']},
    'hidden_dim': {'units': 32, 'use_bias': True, 'activation': 'linear'},
    'top_k_args': {'k': 0.3, 'kernel_initializer': 'ones'},
    'activation': 'relu',
    'use_reconnect': True,
    'depth': 4,
    'pooling_args': {"pooling_method": 'segment_mean'},
    'gather_args': {"node_indexing": 'sample'},
    'verbose': 1
},
    'training': {
        'fit': {'batch_size': 32, 'epochs': 500, 'validation_freq': 2, 'verbose': 2},
        'optimizer': {'class_name': 'Adam', "config": {'lr': 5e-4}},
        'callbacks': [{'class_name': 'kgcnn>LinearLearningRateScheduler',
                       "config": {'learning_rate_start': 0.5e-3,
                                  'learning_rate_stop': 1e-5,
                                  'epo_min': 400, 'epo': 500, 'verbose': 0}}]
    }
}
}
