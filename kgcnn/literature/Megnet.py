import tensorflow.keras as ks

from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint
from kgcnn.layers.disjoint.conv import MEGnetBlock
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.pooling import PoolingNodes, PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.utils.models import generate_standard_graph_input


# Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals
# by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong*
# https://github.com/materialsvirtuallab/megnet


def make_megnet(
        # Input
        input_node_shape,
        input_edge_shape,
        input_state_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,  # Only graph possible for megnet
        output_mlp: dict = None,
        # Model specs
        meg_block_args: dict = None,
        node_ff_args: dict = None,
        edge_ff_args: dict = None,
        state_ff_args: dict = None,
        set2set_args: dict = None,
        nblocks: int = 3,
        has_ff: bool = True,
        dropout: float = None,
        use_set2set: bool = True,
        ):
    """
    Get Megnet model.
    
    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_embedd (dict): Dictionary of input embedding info. See default values of kgcnn.utils.models.
        
        output_embedd (str): Graph or node embedding of the graph network. Default is {"output_mode": 'graph'}.
        output_mlp (dict): MLP for output arguments. Default is {"mlp_use_bias": [True, True, True],
                            "mlp_units": [32, 16, 1], "mlp_activation": ['softplus2', 'softplus2', 'linear']}

        meg_block_args (dict): MegBlock arguments. Default is
                                {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32], 'env_embed': [64, 32, 32],
                                'activation': 'softplus2', 'is_sorted': False, 'has_unconnected': True}
        node_ff_args (dict): Feed-Forward Layer arguments. Default is "mlp_units": [64, 32],
                            "mlp_activation": ["softplus2", "softplus2"]}
        edge_ff_args (dict): Feed-Forward Layer arguments. Default is "mlp_units": [64, 32],
                            "mlp_activation": ["softplus2", "softplus2"]}
        state_ff_args (dict): Feed-Forward Layer arguments. Default is "mlp_units": [64, 32],
                            "mlp_activation": ["softplus2", "softplus2"]}
        set2set_args (dict): Set2Set Layer Arguments. Default is  {'set2set_dim': 16, 'T': 3, "pooling_method": "sum",
                            "init_qstar": "0"}

        nblocks (int): Number of block. Default is 3.
        has_ff (bool): Feed forward layer. Default is True.
        dropout (float): Use dropout. Default is None.
        use_set2set (bool): Use set2set. Default is True.

    Returns:
        model (tf.keras.models.Model): MEGnet model.
    """
    # Default arguments if None
    input_embedd = {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100, 'input_node_embedd': 64,
                    'input_edge_embedd': 64, 'input_state_embedd': 64,
                    'input_type': 'ragged'} if input_embedd is None else input_embedd
    output_embedd = {"output_mode": 'graph', "output_type": 'padded'} if output_embedd is None else output_embedd
    output_mlp = {"use_bias": [True, True, True], "units": [32, 16, 1],
                  "activation": ['softplus2', 'softplus2', 'linear']} if output_mlp is None else output_mlp
    meg_block_args = {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32], 'env_embed': [64, 32, 32],
                      'activation': 'softplus2', 'is_sorted': False,
                      'has_unconnected': True} if meg_block_args is None else meg_block_args
    set2set_args = {'channels': 16, 'T': 3, "pooling_method": "sum",
                    "init_qstar": "0"} if set2set_args is None else set2set_args
    node_ff_args = {"units": [64, 32],
                    "activation": "softplus2"} if node_ff_args is None else node_ff_args
    edge_ff_args = {"units": [64, 32],
                    "activation":"softplus2"} if edge_ff_args is None else edge_ff_args
    state_ff_args = {"units": [64, 32],
                     "activation": "softplus2"} if state_ff_args is None else state_ff_args

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                                     input_edge_shape,
                                                                                                     input_state_shape,
                                                                                                     **input_embedd)
    n, node_len, ed, edge_len, edi = CastRaggedToDisjoint()([n, ed, edge_index_input])

    # starting
    vp = n
    up = uenv
    ep = ed
    vp = MLP(**node_ff_args)(vp)
    ep = MLP(**edge_ff_args)(ep)
    up = MLP(**state_ff_args)(up)
    ep2 = ep
    vp2 = vp
    up2 = up
    for i in range(0, nblocks):
        if has_ff and i > 0:
            vp2 = MLP(**node_ff_args)(vp)
            ep2 = MLP(**edge_ff_args)(ep)
            up2 = MLP(**state_ff_args)(up)

        # MEGnetBlock
        vp2, ep2, up2 = MEGnetBlock(**meg_block_args)([vp2, ep2, edi, up2, node_len, edge_len])
        # skip connection

        if dropout is not None:
            vp2 = ks.layers.Dropout(dropout, name='dropout_atom_%d' % i)(vp2)
            ep2 = ks.layers.Dropout(dropout, name='dropout_bond_%d' % i)(ep2)
            up2 = ks.layers.Dropout(dropout, name='dropout_state_%d' % i)(up2)

        vp = ks.layers.Add()([vp2, vp])
        up = ks.layers.Add()([up2, up])
        ep = ks.layers.Add()([ep2, ep])

    if use_set2set:
        vp = ks.layers.Dense(set2set_args["channels"], activation='linear')(vp)
        ep = ks.layers.Dense(set2set_args["channels"], activation='linear')(ep)
        vp = Set2Set(**set2set_args)([vp, node_len])
        ep = Set2Set(**set2set_args)([ep, edge_len])
    else:
        vp = PoolingNodes()([vp, node_len])
        ep = PoolingAllEdges()([ep, edge_len])

    ep = ks.layers.Flatten()(ep)
    vp = ks.layers.Flatten()(vp)
    final_vec = ks.layers.Concatenate(axis=-1)([vp, ep, up])

    if dropout is not None:
        final_vec = ks.layers.Dropout(dropout, name='dropout_final')(final_vec)

    # final dense layers 
    main_output = MLP(**output_mlp)(final_vec)

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=main_output)

    return model
