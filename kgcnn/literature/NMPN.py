import tensorflow.keras as ks
import pprint
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.set2set import Set2Set
from kgcnn.layers.update import GRUUpdate, TrafoMatMulMessages
from kgcnn.utils.models import generate_edge_embedding, update_model_args, generate_node_embedding


# Neural Message Passing for Quantum Chemistry
# by Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
# http://arxiv.org/abs/1704.01212    


def make_nmpn(**kwargs):
    """Get Message passing model.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: Message Passing model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_edge_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['selu', 'selu', 'sigmoid']},
                     'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "sum",
                                      "init_qstar": "0"},
                     'pooling_args': {'pooling_method': "segment_mean"},
                     'edge_dense': {'use_bias': True, 'activation': 'selu'},
                     'use_set2set': True, 'depth': 3, 'node_dim': 128,
                     'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # local updated model args
    input_node_shape = m['input_node_shape']
    input_edge_shape = m['input_edge_shape']
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    set2set_args = m['set2set_args']
    pooling_args = m['pooling_args']
    edge_dense = m['edge_dense']
    use_set2set = m['use_set2set']
    node_dim = m['node_dim']
    depth = m['depth']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    # embedding, if no feature dimension
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    edi = edge_index_input

    # Model
    n = Dense(node_dim, activation="linear")(n)
    edge_net = Dense(node_dim * node_dim, **edge_dense)(ed)
    gru = GRUUpdate(node_dim)

    for i in range(0, depth):
        eu = GatherNodesOutgoing()([n, edi])
        eu = TrafoMatMulMessages(node_dim, )([edge_net, eu])
        eu = PoolingLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connections
        n = gru([n, eu])

    # Output embedding choice
    if output_embedding["output_mode"] == 'graph':
        if use_set2set:
            # output
            outss = Dense(set2set_args['channels'], activation="linear")(n)
            out = Set2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes(**pooling_args)(n)

        # final dense layers
        main_output = MLP(**output_mlp)(out)
    else:  # Node labeling
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(main_output)
        # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model
