import tensorflow as tf
from kgcnn.layers.conv.megnet_conv import MEGnetBlock
from kgcnn.layers.geom import NodeDistanceEuclidean, GaussBasisLayer, NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import DenseEmbedding, LazyAdd, DropoutEmbedding, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingGlobalEdges, PoolingNodes
from kgcnn.layers.pool.set2set import PoolingSet2Set
from kgcnn.utils.models import update_model_kwargs

# from kgcnn.layers.casting import ChangeTensorType
ks = tf.keras

# Implementation of Megnet in `tf.keras` from paper:
# Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals
# by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong*
# https://github.com/materialsvirtuallab/megnet
# https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294


model_default = {
    "name": "Megnet",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": [], "name": "graph_attributes", "dtype": "float32", "ragged": False}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "make_distance": True, "expand_distance": True,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "meg_block_args": {"node_embed": [64, 32, 32], "edge_embed": [64, 32, 32],
                       "env_embed": [64, 32, 32], "activation": "kgcnn>softplus2"},
    "set2set_args": {"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
    "node_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
    "edge_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
    "state_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
    "nblocks": 3, "has_ff": True, "dropout": None, "use_set2set": True,
    "verbose": 10,
    "output_embedding": "graph",
    "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                   "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               expand_distance: bool = None,
               make_distance: bool = None,
               gauss_args: dict = None,
               meg_block_args: dict = None,
               set2set_args: dict = None,
               node_ff_args: dict = None,
               edge_ff_args: dict = None,
               state_ff_args: dict = None,
               use_set2set: bool = None,
               nblocks: int = None,
               has_ff: bool = None,
               dropout: float = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_mlp: dict = None
               ):
    r"""Make `MegNet <https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Megnet.model_default`.

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices, state_attributes]`
        or `[node_attributes, node_coordinates, edge_indices, state_attributes]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge attributes or distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - state_attributes (tf.Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        meg_block_args (dict): Dictionary of layer arguments unpacked in :obj:`MEGnetBlock` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in `:obj:PoolingSet2Set` layer.
        node_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        edge_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        state_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2Set` layer.
        nblocks (int): Number of graph embedding blocks or depth of the network.
        has_ff (bool): Use feed-forward MLP in each block.
        dropout (int): Dropout to use. Default is None.
        name (str): Name of the model.
        verbose (int): Verbosity level of print.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
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
        raise ValueError("Unsupported output embedding for mode `Megnet`.")

    main_output = MLP(**output_mlp)(final_vec)
    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input, env_input], outputs=main_output)
    return model


model_crystal_default = {
    'name': "Megnet",
    'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
               {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
               {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64', 'ragged': True},
               {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}],
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


@update_model_kwargs(model_crystal_default)
def make_crystal_model(inputs: list = None,
                       input_embedding: dict = None,
                       expand_distance: bool = None,
                       make_distance: bool = None,
                       gauss_args: dict = None,
                       meg_block_args: dict = None,
                       set2set_args: dict = None,
                       node_ff_args: dict = None,
                       edge_ff_args: dict = None,
                       state_ff_args: dict = None,
                       use_set2set: bool = None,
                       nblocks: int = None,
                       has_ff: bool = None,
                       dropout: float = None,
                       name: str = None,
                       verbose: int = None,
                       output_embedding: str = None,
                       output_mlp: dict = None
                       ):
    r"""Make `MegNet <https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Megnet.model_crystal_default`.

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices, state_attributes, edge_image, lattice]`
        or `[node_attributes, node_coordinates, edge_indices, state_attributes, edge_image, lattice]`
        if :obj:`make_distance=True` and :obj:`expand_distance=True` to compute edge distances
        from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - edge_image (tf.RaggedTensor): Indices of the periodic image the sending node is located. The indices
                of and edge are :math:`(i, j)` with :math:`j` being the sending node.
            - state_attributes (tf.Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
                using an embedding layer.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        meg_block_args (dict): Dictionary of layer arguments unpacked in :obj:`MEGnetBlock` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in `:obj:PoolingSet2Set` layer.
        node_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        edge_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        state_ff_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` feed-forward layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2Set` layer.
        nblocks (int): Number of graph embedding blocks or depth of the network.
        has_ff (bool): Use feed-forward MLP in each block.
        dropout (int): Dropout to use. Default is None.
        name (str): Name of the model.
        verbose (int): Verbosity level of print.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    env_input = ks.Input(**inputs[3])
    edge_image = ks.layers.Input(**inputs[4])
    lattice = ks.layers.Input(**inputs[5])

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
        pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
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
        raise ValueError("Unsupported output embedding for mode `Megnet`.")

    main_output = MLP(**output_mlp)(final_vec)
    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input, env_input, edge_image, lattice],
                            outputs=main_output)
    return model
