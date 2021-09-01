import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.keras import Concatenate, Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.pool.set2set import PoolingSet2Set
from kgcnn.utils.models import update_model_kwargs, generate_embedding

# 'Interaction Networks for Learning about Objects,Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://github.com/higgsfield/interaction_network_pytorch

hyper_model_default = {'name': "INorp",
                       'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                                  {'shape': [], 'name': "graph_attributes", 'dtype': 'float32', 'ragged': False}],
                       'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                           "edge": {"input_dim": 5, "output_dim": 64},
                                           "graph": {"input_dim": 100, "output_dim": 64}},
                       'output_embedding': 'graph',
                       'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                      "activation": ['relu', 'relu', 'sigmoid']},
                       'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "mean",
                                        "init_qstar": "mean"},
                       'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                       'edge_mlp_args': {"units": [100, 100, 100, 100, 50],
                                         "activation": ['relu', 'relu', 'relu', 'relu', "linear"]},
                       'pooling_args': {'pooling_method': "segment_mean"},
                       'depth': 3, 'use_set2set': False, 'verbose': 1,
                       'gather_args': {}
                       }


@update_model_kwargs(hyper_model_default)
def make_model(inputs=None,
               input_embedding=None,
               output_embedding=None,
               output_mlp=None,
               depth=None,
               gather_args=None,
               edge_mlp_args=None,
               node_mlp_args=None,
               set2set_args=None,
               pooling_args=None,
               use_set2set=None,
               **kwargs
               ):
    """Make INorp graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.
        depth (int): Number of graph embedding units or depth of the network.
        gather_args (dict): Dictionary of layer arguments unpacked in `GatherNodes` layer.
        edge_mlp_args (dict): Dictionary of layer arguments unpacked in `MLP` layer for edge updates.
        node_mlp_args (dict): Dictionary of layer arguments unpacked in `MLP` layer for node updates.
        set2set_args (dict): Dictionary of layer arguments unpacked in `PoolingSet2Set` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in `PoolingLocalEdges`, `PoolingNodes` layer.
        use_set2set (bool): Whether to use `PoolingSet2Set` layer.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    env_input = ks.Input(**inputs[3])

    # embedding, if no feature dimension
    n = generate_embedding(node_input, inputs[0]['shape'], input_embedding['node'])
    ed = generate_embedding(edge_input, inputs[1]['shape'], input_embedding['edge'])
    uenv = generate_embedding(env_input, inputs[3]['shape'], input_embedding['graph'], embedding_rank=0)
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
    if output_embedding == 'graph':
        if use_set2set:
            # output
            outss = Dense(set2set_args["channels"], activation="linear")(n)
            out = PoolingSet2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes(**pooling_args)(n)
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(main_output)
        # no ragged for distribution atm
    else:
        raise ValueError("Unsupported graph embedding for mode `INorp`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=main_output)
    return model
