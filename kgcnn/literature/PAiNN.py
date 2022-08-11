import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.painn_conv import PAiNNUpdate, EquivariantInitialize
from kgcnn.layers.conv.painn_conv import PAiNNconv
from kgcnn.layers.geom import NodeDistanceEuclidean, BesselBasisLayer, EdgeDirectionNormalized, CosCutOffEnvelope, \
    NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.norm import GraphLayerNormalization, GraphBatchNormalization
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of PAiNN in `tf.keras` from paper:
# Equivariant message passing for the prediction of tensorial properties and molecular spectra
# Kristof T. Schuett, Oliver T. Unke and Michael Gastegger
# https://arxiv.org/pdf/2102.03150.pdf

model_default = {
    "name": "PAiNN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
    "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    "pooling_args": {"pooling_method": "sum"},
    "conv_args": {"units": 128, "cutoff": None, "conv_pool": "sum"},
    "update_args": {"units": 128},
    "equiv_normalization": False, "node_normalization": False,
    "depth": 3,
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               bessel_basis: dict = None,
               depth: int = None,
               pooling_args: dict = None,
               conv_args: dict = None,
               update_args: dict = None,
               equiv_normalization: bool = None,
               node_normalization: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.PAiNN.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, bond_indices]`
        or `[node_attributes, node_coordinates, bond_indices, equiv_initial]` if a custom equivariant initialization is
        chosen other than zero.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Atomic coordinates of shape `(batch, None, 3)`.
            - bond_indices (tf.RaggedTensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - equiv_initial (tf.RaggedTensor): Equivariant initialization `(batch, None, 3, F)`. Optional.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        bessel_basis (dict): Dictionary of layer arguments unpacked in final :obj:`BesselBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        conv_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNconv` layer.
        update_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNUpdate` layer.
        equiv_normalization (bool): Whether to apply :obj:`GraphLayerNormalization` to equivariant tensor update.
        node_normalization (bool): Whether to apply :obj:`GraphBatchNormalization` to node tensor update.
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
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    z = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)

    if len(inputs) > 3:
        equiv_input = ks.layers.Input(**inputs[3])
    else:
        equiv_input = EquivariantInitialize(dim=3)(z)

    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    pos1, pos2 = NodePosition()([x, edi])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        if equiv_normalization:
            dv = GraphLayerNormalization(axis=2)(dv)
        if node_normalization:
            ds = GraphBatchNormalization(axis=-1)(ds)
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        if equiv_normalization:
            dv = GraphLayerNormalization(axis=2)(dv)
        if node_normalization:
            ds = GraphBatchNormalization(axis=-1)(ds)
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])

    n = z
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)(n)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `PAiNN`")

    if len(inputs) > 3:
        model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input, equiv_input], outputs=out)
    else:
        model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input], outputs=out)
    return model


model_crystal_default = {
    "name": "PAiNN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64', 'ragged': True},
        {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
    "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    "pooling_args": {"pooling_method": "sum"},
    "conv_args": {"units": 128, "cutoff": None, "conv_pool": "sum"},
    "update_args": {"units": 128},
    "equiv_normalization": False, "node_normalization": False,
    "depth": 3,
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_crystal_default)
def make_crystal_model(inputs: list = None,
                       input_embedding: dict = None,
                       bessel_basis: dict = None,
                       depth: int = None,
                       pooling_args: dict = None,
                       conv_args: dict = None,
                       update_args: dict = None,
                       equiv_normalization: bool = None,
                       node_normalization: bool = None,
                       name: str = None,
                       verbose: int = None,
                       output_embedding: str = None,
                       output_to_tensor: bool = None,
                       output_mlp: dict = None
                       ):
    r"""Make `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.PAiNN.model_crystal_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, bond_indices, edge_image, lattice]`
        or `[node_attributes, node_coordinates, bond_indices, edge_image, lattice, equiv_initial]` if a custom
        equivariant initialization is chosen other than zero.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Atomic coordinates of shape `(batch, None, 3)`.
            - bond_indices (tf.RaggedTensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - equiv_initial (tf.RaggedTensor): Equivariant initialization `(batch, None, 3, F)`. Optional.
            - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
            - edge_image (tf.RaggedTensor): Indices of the periodic image the sending node is located. The indices
                of and edge are :math:`(i, j)` with :math:`j` being the sending node.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        bessel_basis (dict): Dictionary of layer arguments unpacked in final :obj:`BesselBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        conv_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNconv` layer.
        update_args (dict): Dictionary of layer arguments unpacked in :obj:`PAiNNUpdate` layer.
        equiv_normalization (bool): Whether to apply :obj:`GraphLayerNormalization` to equivariant tensor update.
        node_normalization (bool): Whether to apply :obj:`GraphBatchNormalization` to node tensor update.
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
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    edge_image = ks.layers.Input(**inputs[3])
    lattice = ks.layers.Input(**inputs[4])
    z = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)

    if len(inputs) > 5:
        equiv_input = ks.layers.Input(**inputs[5])
    else:
        equiv_input = EquivariantInitialize(dim=3)(z)

    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    pos1, pos2 = NodePosition()([x, edi])
    pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        if equiv_normalization:
            dv = GraphLayerNormalization(axis=2)(dv)
        if node_normalization:
            ds = GraphBatchNormalization(axis=-1)(ds)
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        if equiv_normalization:
            dv = GraphLayerNormalization(axis=2)(dv)
        if node_normalization:
            ds = GraphBatchNormalization(axis=-1)(ds)
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])

    n = z
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)(n)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `PAiNN`")

    if len(inputs) > 5:
        model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input, edge_image, lattice, equiv_input],
                                outputs=out)
    else:
        model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input, edge_image, lattice], outputs=out)
    return model
