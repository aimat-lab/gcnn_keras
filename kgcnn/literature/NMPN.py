import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.mpnn_conv import GRUUpdate, TrafoEdgeNetMessages, MatMulMessages
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.pool.set2set import PoolingSet2Set
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer, ShiftPeriodicLattice

ks = tf.keras

# Implementation of NMPN in `tf.keras` from paper:
# Neural Message Passing for Quantum Chemistry
# by Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
# http://arxiv.org/abs/1704.01212    

model_default = {
    "name": "NMPN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "geometric_edge": False, "make_distance": False, "expand_distance": False,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum",
                     "init_qstar": "0"},
    "pooling_args": {"pooling_method": "segment_sum"},
    "edge_mlp": {"use_bias": True, "activation": "swish", "units": [64, 64, 64]},
    "use_set2set": True, "depth": 3, "node_dim": 64,
    "verbose": 10,
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["selu", "selu", "sigmoid"]},
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               geometric_edge: bool = None,
               make_distance: bool = None,
               expand_distance: bool = None,
               gauss_args: dict = None,
               set2set_args: dict = None,
               pooling_args: dict = None,
               edge_mlp: dict = None,
               use_set2set: bool = None,
               node_dim: int = None,
               depth: int = None,
               verbose: int = None,
               name: str = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `NMPN <http://arxiv.org/abs/1704.01212>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.NMPN.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices]`
        or `[node_attributes, edge_distance, edge_indices]` if :obj:`geometric_edge=True`
        or `[node_attributes, node_coordinates, edge_indices]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge attributes or distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        geometric_edge (bool): Whether the edges are geometric, like distance or coordinates.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2Set` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`, `PoolingLocalEdges` layers.
        edge_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge matrix.
        use_set2set (bool): Whether to use :obj:`PoolingSet2Set` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])  # Or coordinates
    edge_index_input = ks.layers.Input(**inputs[2])
    edi = edge_index_input

    # embedding, if no feature dimension
    n0 = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    if not geometric_edge:
        ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    else:
        ed = edge_input

    # If coordinates are in place of edges
    if make_distance:
        pos1, pos2 = NodePosition()([ed, edi])
        ed = NodeDistanceEuclidean()([pos1, pos2])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Make hidden dimension
    n = DenseEmbedding(node_dim, activation="linear")(n0)

    # Make edge networks.
    edge_net_in = GraphMLP(**edge_mlp)(ed)
    edge_net_in = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_in)
    edge_net_out = GraphMLP(**edge_mlp)(ed)
    edge_net_out = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_out)

    # Gru for node updates
    gru = GRUUpdate(node_dim)

    for i in range(0, depth):
        n_in = GatherNodesOutgoing()([n, edi])
        n_out = GatherNodesIngoing()([n, edi])
        m_in = MatMulMessages()([edge_net_in, n_in])
        m_out = MatMulMessages()([edge_net_out, n_out])
        eu = LazyConcatenate(axis=-1)([m_in, m_out])
        eu = PoolingLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connections
        n = gru([n, eu])

    n = LazyConcatenate(axis=-1)([n0, n])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            out = DenseEmbedding(set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2Set(**set2set_args)(out)
        else:
            out = PoolingNodes(**pooling_args)(n)
        out = ks.layers.Flatten()(out)  # Flatten() required for to Set2Set output.
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `NMPN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)
    return model


model_crystal_default = {
    "name": "NMPN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "geometric_edge": False, "make_distance": False, "expand_distance": False,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum",
                     "init_qstar": "0"},
    "pooling_args": {"pooling_method": "segment_sum"},
    "edge_mlp": {"use_bias": True, "activation": "swish", "units": [64, 64, 64]},
    "use_set2set": True, "depth": 3, "node_dim": 64,
    "verbose": 10,
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["selu", "selu", "sigmoid"]},
}


@update_model_kwargs(model_crystal_default)
def make_crystal_model(inputs: list = None,
                       input_embedding: dict = None,
                       geometric_edge: bool = None,
                       make_distance: bool = None,
                       expand_distance: bool = None,
                       gauss_args: dict = None,
                       set2set_args: dict = None,
                       pooling_args: dict = None,
                       edge_mlp: dict = None,
                       use_set2set: bool = None,
                       node_dim: int = None,
                       depth: int = None,
                       verbose: int = None,
                       name: str = None,
                       output_embedding: str = None,
                       output_to_tensor: bool = None,
                       output_mlp: dict = None
                       ):
    r"""Make `NMPN <http://arxiv.org/abs/1704.01212>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.NMPN.model_crystal_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_image, lattice]`
        or `[node_attributes, edge_distance, edge_indices, edge_image, lattice]` if :obj:`geometric_edge=True`
        or `[node_attributes, node_coordinates, edge_indices, edge_image, lattice]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge attributes or distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - edge_image (tf.RaggedTensor): Indices of the periodic image the sending node is located. The indices
                of and edge are :math:`(i, j)` with :math:`j` being the sending node.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        geometric_edge (bool): Whether the edges are geometric, like distance or coordinates.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2Set` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`, `PoolingLocalEdges` layers.
        edge_mlp (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge matrix.
        use_set2set (bool): Whether to use :obj:`PoolingSet2Set` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])  # Or coordinates
    edge_index_input = ks.layers.Input(**inputs[2])
    edge_image = ks.layers.Input(**inputs[3])
    lattice = ks.layers.Input(**inputs[4])

    edi = edge_index_input

    # embedding, if no feature dimension
    n0 = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    if not geometric_edge:
        ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    else:
        ed = edge_input

    # If coordinates are in place of edges
    if make_distance:
        x = ed
        pos1, pos2 = NodePosition()([x, edi])
        pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
        ed = NodeDistanceEuclidean()([pos1, pos2])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Make hidden dimension
    n = DenseEmbedding(node_dim, activation="linear")(n0)

    # Make edge networks.
    edge_net_in = GraphMLP(**edge_mlp)(ed)
    edge_net_in = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_in)
    edge_net_out = GraphMLP(**edge_mlp)(ed)
    edge_net_out = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_out)

    # Gru for node updates
    gru = GRUUpdate(node_dim)

    for i in range(0, depth):
        n_in = GatherNodesOutgoing()([n, edi])
        n_out = GatherNodesIngoing()([n, edi])
        m_in = MatMulMessages()([edge_net_in, n_in])
        m_out = MatMulMessages()([edge_net_out, n_out])
        eu = LazyConcatenate(axis=-1)([m_in, m_out])
        eu = PoolingLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connections
        n = gru([n, eu])

    n = LazyConcatenate(axis=-1)([n0, n])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            out = DenseEmbedding(set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2Set(**set2set_args)(out)
        else:
            out = PoolingNodes(**pooling_args)(n)
        out = ks.layers.Flatten()(out)  # Flatten() required for to Set2Set output.
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `NMPN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, edge_image, lattice], outputs=out)
    return model
