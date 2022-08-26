import tensorflow as tf
from kgcnn.layers.conv.cgcnn_conv import CGCNNLayer
from kgcnn.layers.geom import DisplacementVectorsASU, DisplacementVectorsUnitCell, FracToRealCoordinates, \
    EuclideanNorm, GaussBasisLayer, NodePosition
from kgcnn.layers.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn.layers.modules import OptionalInputEmbedding, LazySubtract, DenseEmbedding
from kgcnn.layers.mlp import MLP
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of CGCNN in `tf.keras` from paper:
# Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties
# Tian Xie and Jeffrey C. Grossman
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301


model_crystal_default = {
    'name': 'CGCNN',
    'inputs': [
        {'shape': (None,), 'name': 'node_number', 'dtype': 'int64', 'ragged': True},
        {'shape': (None, 3), 'name': 'node_frac_coordinates', 'dtype': 'float64', 'ragged': True},
        {'shape': (None, 2), 'name': 'edge_indices', 'dtype': 'int64', 'ragged': True},
        {'shape': (3, 3), 'name': 'lattice_matrix', 'dtype': 'float64', 'ragged': False},
        {'shape': (None, 3), 'name': 'cell_translations', 'dtype': 'float32', 'ragged': True},
        # For `representation="asu"`:
        # {'shape': (None, 1), 'name': 'multiplicities', 'dtype': 'float32', 'ragged': True},
        # {'shape': (None, 4, 4), 'name': 'symmops', 'dtype': 'float64', 'ragged': True},
    ],
    'input_embedding': {'node': {'input_dim': 95, 'output_dim': 64}},
    'representation': 'unit',  # None, 'asu' or 'unit'
    'expand_distance': True,
    'make_distances': True,
    'gauss_args': {'bins': 40, 'distance': 8, 'offset': 0.0, 'sigma': 0.4},
    'depth': 3,
    "verbose": 10,
    'conv_layer_args': {
        'units': 64,
        'activation_s': 'softplus',
        'activation_out': 'softplus',
        'batch_normalization': True,
    },
    'node_pooling_args': {'pooling_method': 'mean'},
    "output_embedding": "graph",
    'output_mlp': {'use_bias': [True, False], 'units': [64, 1],
                   'activation': ['softplus', 'linear']},
}


@update_model_kwargs(model_crystal_default)
def make_crystal_model(inputs: list = None,
                       representation: str = None,
                       make_distances: bool = None,
                       input_embedding: dict = None,
                       conv_layer_args: dict = None,
                       expand_distance: bool = None,
                       depth: int = None,
                       name: str = None,
                       verbose: int = None,
                       gauss_args: dict = None,
                       node_pooling_args: dict = None,
                       output_mlp: dict = None,
                       output_embedding: str = None,
                       ):
    """Make `CGCNN <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_ graph network
    via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.CGCNN.model_crystal_default`.

    Inputs:
        list: `[node_attributes, node_frac_coordinates, bond_indices, lattice, cell_translations]`
        if :obj:`representation='unit'` and :obj:`make_distances=True`
        or `[node_attributes, node_frac_coordinates, bond_indices, lattice, cell_translations, multiplicities, symmops]`
        if :obj:`representation='asu'` and :obj:`make_distances=True`
        or `[node_attributes, edge_distance, bond_indices]`
        if :obj:`make_distances=False`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_frac_coordinates (tf.RaggedTensor): Fractional coordinates of shape `(batch, None, 3)`.
            - bond_indices (tf.RaggedTensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - cell_translations (tf.RaggedTensor): Indices of the periodic image the sending node is located.
                The indices of an edge are :math:`(i, j)` with :math:`j` being the sending node.
            - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, None, D)` expanded
                in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
                with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - multiplicities (tf.RaggedTensor): Multiplicities of asymmetric unit cell of shape `(batch, None, 1)`.
            - symmops (tf.RaggedTensor): Symmetry operations for atoms of shape `(batch, None, 4, 4)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        make_distances (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        representation (str): The representation of unit cell. Can be either `None`, 'asu' or 'unit'. Default is 'unit'.
        conv_layer_args (dict):
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    atom_attributes = ks.layers.Input(**inputs[0])
    edge_indices = ks.layers.Input(**inputs[2])

    if make_distances:

        frac_coords = ks.layers.Input(**inputs[1])
        lattice_matrix = ks.layers.Input(**inputs[3])

        if representation == 'unit':
            cell_translations = ks.layers.Input(**inputs[4])
            displacement_vectors = DisplacementVectorsUnitCell()([frac_coords, edge_indices, cell_translations])

        elif representation == 'asu':
            cell_translations = ks.layers.Input(**inputs[4])
            multiplicities = ks.layers.Input(**inputs[5])
            symmops = ks.layers.Input(**inputs[6])

            displacement_vectors = DisplacementVectorsASU()([frac_coords, edge_indices, symmops, cell_translations])
        else:
            x_in, x_out = NodePosition()([frac_coords, edge_indices])
            displacement_vectors = LazySubtract()([x_out, x_in])

        displacement_vectors = FracToRealCoordinates()([displacement_vectors, lattice_matrix])

        edge_distances = EuclideanNorm(axis=2, keepdims=True)(displacement_vectors)

    else:
        edge_distances_input = ks.layers.Input(**inputs[1])
        edge_distances = edge_distances_input

    if expand_distance:
        edge_distances = GaussBasisLayer(**gauss_args)(edge_distances)

    # embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(atom_attributes)

    n = DenseEmbedding(conv_layer_args["units"], activation='linear')(n)
    for _ in range(depth):
        n = CGCNNLayer(**conv_layer_args)([n, edge_distances, edge_indices])

    if representation == 'asu':
        out = PoolingWeightedNodes(**node_pooling_args)([n, multiplicities])
    else:
        out = PoolingNodes(**node_pooling_args)(n)

    out = MLP(**output_mlp)(out)

    # Only graph embedding for CGCNN.
    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `CGCNN`.")

    if make_distances:
        if representation == 'unit':
            model = ks.models.Model(
                inputs=[atom_attributes, frac_coords, edge_indices, lattice_matrix, cell_translations],
                outputs=out, name=name)
        elif representation == 'asu':
            model = ks.models.Model(
                inputs=[atom_attributes, frac_coords, edge_indices, lattice_matrix, cell_translations,
                        multiplicities, symmops], outputs=out, name=name)
        else:
            model = ks.models.Model(
                inputs=[atom_attributes, frac_coords, edge_indices, lattice_matrix],
                outputs=out, name=name)
    else:
        model = ks.models.Model(
            inputs=[atom_attributes, edge_distances_input, edge_indices],
            outputs=out, name=name)
    return model
