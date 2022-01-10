import tensorflow.keras as ks

from kgcnn.layers.conv.megnet_conv import MEGnetBlock
from kgcnn.layers.geom import NodeDistanceEuclidean, GaussBasisLayer, NodePosition
from kgcnn.layers.modules import DenseEmbedding, LazyAdd, DropoutEmbedding, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingGlobalEdges, PoolingNodes
from kgcnn.layers.pool.set2set import PoolingSet2Set
from kgcnn.utils.models import update_model_kwargs

# from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing

# Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals
# by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong*
# https://github.com/materialsvirtuallab/megnet


model_default = {'name': "Megnet",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                            {'shape': [], 'name': "graph_attributes", 'dtype': 'float32', 'ragged': False}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                     "graph": {"input_dim": 100, "output_dim": 64}},
                 "make_distance": True, "expand_distance": True,
                 'gauss_args': {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
                 'meg_block_args': {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32],
                                    'env_embed': [64, 32, 32], 'activation': 'kgcnn>softplus2'},
                 'set2set_args': {'channels': 16, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},
                 'node_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                 'edge_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                 'state_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                 'nblocks': 3, 'has_ff': True, 'dropout': None, 'use_set2set': True,
                 'verbose': 10,
                 'output_embedding': 'graph',
                 'output_mlp': {"use_bias": [True, True, True], "units": [32, 16, 1],
                                "activation": ['kgcnn>softplus2', 'kgcnn>softplus2', 'linear']}
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               expand_distance=None,
               make_distance=None,
               gauss_args=None,
               meg_block_args=None,
               set2set_args=None,
               node_ff_args=None,
               edge_ff_args=None,
               state_ff_args=None,
               use_set2set=None,
               nblocks=None,
               has_ff=None,
               dropout=None,
               name=None,
               verbose=None,
               output_embedding=None,
               output_mlp=None
               ):
    """Make MegNet graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in `GaussBasisLayer` layer.
        meg_block_args (dict): Dictionary of layer arguments unpacked in `MEGnetBlock` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in `PoolingSet2Set` layer.
        node_ff_args (dict): Dictionary of layer arguments unpacked in `MLP` feed-forward layer.
        edge_ff_args (dict): Dictionary of layer arguments unpacked in `MLP` feed-forward layer.
        state_ff_args (dict): Dictionary of layer arguments unpacked in `MLP` feed-forward layer.
        use_set2set (bool): Whether to use `PoolingSet2Set` layer.
        nblocks (int): Number of graph embedding blocks or depth of the network.
        has_ff (bool): Use feed-forward MLP in each block.
        dropout (int): Dropout to use. Default is None.
        name (str): Name of the model.
        verbose (int): Verbosity level of print.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    env_input = ks.Input(**inputs[3])

    # embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    uenv = OptionalInputEmbedding(**input_embedding['graph'],
                                  use_embedding=len(inputs[3]['shape']) < 1)(env_input)
    edi = edge_index_input

    # Edge distance as Gauss-Basis
    if make_distance:
        x = xyz_input
        pos1, pos2 = NodePosition()([x, edi])
        ed = NodeDistanceEuclidean()([pos1, pos2])
    else:
        ed = xyz_input

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Model
    vp = n
    ep = ed
    up = uenv
    vp = GraphMLP(**node_ff_args)(vp)
    ep = GraphMLP(**edge_ff_args)(ep)
    up = MLP(**state_ff_args)(up)
    vp2 = vp
    ep2 = ep
    up2 = up
    for i in range(0, nblocks):
        if has_ff and i > 0:
            vp2 = GraphMLP(**node_ff_args)(vp)
            ep2 = GraphMLP(**edge_ff_args)(ep)
            up2 = MLP(**state_ff_args)(up)

        # MEGnetBlock
        vp2, ep2, up2 = MEGnetBlock(**meg_block_args)(
            [vp2, ep2, edi, up2])

        # skip connection
        if dropout is not None:
            vp2 = DropoutEmbedding(dropout, name='dropout_atom_%d' % i)(vp2)
            ep2 = DropoutEmbedding(dropout, name='dropout_bond_%d' % i)(ep2)
            up2 = ks.layers.Dropout(dropout, name='dropout_state_%d' % i)(up2)

        vp = LazyAdd()([vp2, vp])
        ep = LazyAdd()([ep2, ep])
        up = ks.layers.Add()([up2, up])

    if use_set2set:
        vp = DenseEmbedding(set2set_args["channels"], activation='linear')(vp)  # to match units
        ep = DenseEmbedding(set2set_args["channels"], activation='linear')(ep)  # to match units
        vp = PoolingSet2Set(**set2set_args)(vp)
        ep = PoolingSet2Set(**set2set_args)(ep)
    else:
        vp = PoolingNodes()(vp)
        ep = PoolingGlobalEdges()(ep)

    ep = ks.layers.Flatten()(ep)
    vp = ks.layers.Flatten()(vp)
    final_vec = ks.layers.Concatenate(axis=-1)([vp, ep, up])

    if dropout is not None:
        final_vec = ks.layers.Dropout(dropout, name='dropout_final')(final_vec)

    # Only graph embedding for Megnet
    if output_embedding != "graph":
        raise ValueError("Unsupported graph embedding for mode `Megnet`.")

    main_output = MLP(**output_mlp)(final_vec)
    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input, env_input], outputs=main_output)
    return model
