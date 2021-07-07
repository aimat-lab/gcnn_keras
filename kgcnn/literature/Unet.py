import tensorflow.keras as ks
import pprint
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.connect import AdjacencyPower
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Dense, Activation, Add
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingNodes, PoolingLocalEdges
from kgcnn.layers.topk import PoolingTopK, UnPoolingTopK
from kgcnn.utils.models import generate_edge_embedding, update_model_args, generate_node_embedding


# Graph U-Nets
# by Hongyang Gao, Shuiwang Ji
# https://arxiv.org/pdf/1905.05178.pdf


def make_unet(**kwargs):
    """Make Graph U-Net.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: Unet model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_edge_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
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
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # Update model args
    input_node_shape = m['input_node_shape']
    input_edge_shape = m['input_edge_shape']
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    pooling_args = m["pooling_args"]
    gather_args = m['gather_args']
    top_k_args = m['top_k_args']
    depth = m['depth']
    use_reconnect = m['use_reconnect']
    hidden_dim = m['hidden_dim']
    activation = m['activation']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    # embedding, if no feature dimension
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
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
    if output_embedding["output_mode"] == 'graph':
        out = PoolingNodes(**pooling_args)(n)
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model
