import tensorflow.keras as ks
import pprint
from kgcnn.utils.models import generate_node_embedding, update_model_args, generate_edge_embedding, \
    generate_state_embedding
from kgcnn.layers.blocks import MEGnetBlock
from kgcnn.layers.keras import Dense, Add, Dropout
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingGlobalEdges, PoolingNodes
from kgcnn.layers.set2set import Set2Set
# from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing

# Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals
# by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong*
# https://github.com/materialsvirtuallab/megnet


def make_megnet(**kwargs):
    """Get Megnet model.

    Args:
        **kwargs

    Returns:
       tf.keras.models.Model: MEGnet model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_edge_shape': None, 'input_state_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, True], "units": [32, 16, 1],
                                    "activation": ['kgcnn>softplus2', 'kgcnn>softplus2', 'linear']},
                     'meg_block_args': {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32],
                                        'env_embed': [64, 32, 32], 'activation': 'kgcnn>softplus2'},
                     'set2set_args': {'channels': 16, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},
                     'node_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                     'edge_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                     'state_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2",
                                       "input_tensor_type": "tensor"},
                     'nblocks': 3, 'has_ff': True, 'dropout': None, 'use_set2set': True,
                     'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # local update default arguments
    input_node_shape = m['input_node_shape']
    input_edge_shape = m['input_edge_shape']
    input_state_shape = m['input_state_shape']
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    meg_block_args = m['meg_block_args']
    set2set_args = m['set2set_args']
    node_ff_args = m['node_ff_args']
    edge_ff_args = m['edge_ff_args']
    state_ff_args = m['state_ff_args']
    use_set2set = m['use_set2set']
    nblocks = m['nblocks']
    has_ff = m['has_ff']
    dropout = m['dropout']

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
    vp = n
    ep = ed
    up = uenv
    vp = MLP(**node_ff_args)(vp)
    ep = MLP(**edge_ff_args)(ep)
    up = MLP(**state_ff_args)(up)
    vp2 = vp
    ep2 = ep
    up2 = up
    for i in range(0, nblocks):
        if has_ff and i > 0:
            vp2 = MLP(**node_ff_args)(vp)
            ep2 = MLP(**edge_ff_args)(ep)
            up2 = MLP(**state_ff_args)(up)

        # MEGnetBlock
        vp2, ep2, up2 = MEGnetBlock(**meg_block_args)(
            [vp2, ep2, edi, up2])

        # skip connection
        if dropout is not None:
            vp2 = Dropout(dropout, name='dropout_atom_%d' % i)(vp2)
            ep2 = Dropout(dropout, name='dropout_bond_%d' % i)(ep2)
            up2 = Dropout(dropout, name='dropout_state_%d' % i)(up2)

        vp = Add()([vp2, vp])
        ep = Add()([ep2, ep])
        up = Add(input_tensor_type="tensor")([up2, up])

    if use_set2set:
        vp = Dense(set2set_args["channels"], activation='linear')(vp)  # to match units
        ep = Dense(set2set_args["channels"], activation='linear')(ep)  # to match units
        vp = Set2Set(**set2set_args)(vp)
        ep = Set2Set(**set2set_args)(ep)
    else:
        vp = PoolingNodes()(vp)
        ep = PoolingGlobalEdges()(ep)

    ep = ks.layers.Flatten()(ep)
    vp = ks.layers.Flatten()(vp)
    final_vec = ks.layers.Concatenate(axis=-1)([vp, ep, up])

    if dropout is not None:
        final_vec = ks.layers.Dropout(dropout, name='dropout_final')(final_vec)

    # final dense layers
    # Only graph embedding for MEGNET
    main_output = MLP(**output_mlp, input_tensor_type="tensor")(final_vec)
    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=main_output)
    return model
