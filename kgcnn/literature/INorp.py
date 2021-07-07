import tensorflow.keras as ks
import pprint
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.keras import Concatenate, Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.set2set import Set2Set
from kgcnn.utils.models import generate_node_embedding, update_model_args, generate_state_embedding, \
    generate_edge_embedding


# 'Interaction Networks for Learning about Objects,Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://github.com/higgsfield/interaction_network_pytorch


def make_inorp(**kwargs):
    """Generate Interaction network.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: Interaction model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_edge_shape': None, 'input_state_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "mean",
                                      "init_qstar": "mean"},
                     'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'edge_mlp_args': {"units": [100, 100, 100, 100, 50],
                                       "activation": ['relu', 'relu', 'relu', 'relu', "linear"]},
                     'pooling_args': {'pooling_method': "segment_mean"},
                     'depth': 3, 'use_set2set': False, 'verbose': 1,
                     'gather_args': {"node_indexing": "sample"}
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # local updated default values
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    depth = m['depth']
    input_node_shape = m['input_node_shape']
    input_edge_shape = m['input_edge_shape']
    input_state_shape = m['input_state_shape']
    gather_args = m['gather_args']
    edge_mlp_args = m['edge_mlp_args']
    node_mlp_args = m['node_mlp_args']
    set2set_args = m['set2set_args']
    pooling_args = m['pooling_args']
    use_set2set = m['use_set2set']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
    env_input = ks.Input(shape=input_state_shape, dtype='float32', name='state_input')

    # embedding, if no feature dimension
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    uenv = generate_state_embedding(env_input, input_state_shape, input_embedding['state'])
    edi = edge_index_input

    # Model
    ev = GatherState(**gather_args)([uenv, n])
    # n-Layer Step
    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing(**gather_args)([n, edi])
        eu2 = GatherNodesOutgoing(**gather_args)([n, edi])
        upd = Concatenate(axis=-1)([eu2, eu1])
        eu = Concatenate(axis=-1)([upd, ed])

        eu = MLP(**edge_mlp_args)(eu)
        # Pool message
        nu = PoolingLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connection
        # Add environment
        nu = Concatenate(axis=-1)(
            [n, nu, ev])  # Concatenate node features with new edge updates
        n = MLP(**node_mlp_args)(nu)

    # Output embedding choice
    if output_embedding["output_mode"] == 'graph':
        if use_set2set:
            # output
            outss = Dense(set2set_args["channels"], activation="linear")(n)
            out = Set2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes(**pooling_args)(n)
        main_output = MLP(**output_mlp)(out)
    else:  # Node labeling
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(main_output)
        # no ragged for distribution atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=main_output)
    return model
