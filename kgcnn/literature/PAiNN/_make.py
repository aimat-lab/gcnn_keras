import keras_core as ks
from ._layers import PAiNNUpdate, EquivariantInitialize
from ._layers import PAiNNconv
from kgcnn.layers.casting import (CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint,
                                  CastDisjointToGraphState, CastDisjointToBatchedAttributes, CastGraphStateToDisjoint)
from kgcnn.layers.geom import NodeDistanceEuclidean, BesselBasisLayer, EdgeDirectionNormalized, CosCutOffEnvelope, \
    NodePosition, ShiftPeriodicLattice
from keras_core.layers import Add
from kgcnn.layers.modules import Embedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.norm import GraphLayerNormalization, GraphBatchNormalization
from kgcnn.models.utils import update_model_kwargs
from keras_core.backend import backend as backend_to_use
from kgcnn.layers.scale import get as get_scaler

# To be updated if model is changed in a significant way.
__model_version__ = "2023-10-04"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'PAiNN' is not supported." % backend_to_use())

# Implementation of PAiNN in `tf.keras` from paper:
# Equivariant message passing for the prediction of tensorial properties and molecular spectra
# Kristof T. Schuett, Oliver T. Unke and Michael Gastegger
# https://arxiv.org/pdf/2102.03150.pdf

model_default = {
    "name": "PAiNN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 128},
    "equiv_initialize_kwargs": {"dim": 3, "method": "zeros"},
    "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    "pooling_args": {"pooling_method": "scatter_sum"},
    "conv_args": {"units": 128, "cutoff": None, "conv_pool": "scatter_sum"},
    "update_args": {"units": 128},
    "equiv_normalization": False, "node_normalization": False,
    "depth": 3,
    "verbose": 10,
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               cast_disjoint_kwargs: dict = None,
               input_node_embedding: dict = None,
               equiv_initialize_kwargs: dict = None,
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
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    r"""Make `PAiNN <https://arxiv.org/pdf/2102.03150.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.PAiNN.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, bond_indices, total_nodes, total_edges]`
        or `[node_attributes, node_coordinates, bond_indices, total_nodes, total_edges, equiv_initial]`
        if a custom equivariant initialization is chosen other than zero.

            - node_attributes (Tensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (Tensor): Atomic coordinates of shape `(batch, None, 3)`.
            - bond_indices (Tensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - equiv_initial (Tensor): Equivariant initialization `(batch, None, 3, F)`. Optional.
            - total_nodes(Tensor): Number of Nodes in graph if not same sized graphs of shape `(batch, )` .
            - total_edges(Tensor): Number of Edges in graph if not same sized graphs of shape `(batch, )` .

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        equiv_initialize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`EquivariantInitialize` layer.
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
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    batched_nodes, batched_x, batched_indices, total_nodes, total_edges = model_inputs[:5]
    z, edi, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])
    x, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_x, total_nodes])

    if len(model_inputs) > 5:
        v, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([model_inputs[6], total_edges])
    else:
        v = EquivariantInitialize(**equiv_initialize_kwargs)(z)

    use_node_embedding = len(inputs[0]['shape']) < 2
    # Optional Embedding.
    if use_node_embedding:
        z = Embedding(**input_node_embedding)(z)

    pos1, pos2 = NodePosition()([x, edi])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        z = Add()([z, ds])
        v = Add()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = Add()([z, ds])
        v = Add()([v, dv])

        if equiv_normalization:
            v = GraphLayerNormalization(axis=2)([v, batch_id_edge, count_edges])
        if node_normalization:
            z = GraphBatchNormalization(axis=-1)([z, batch_id_node, count_nodes])

    n = z
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
        out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
        if output_to_tensor:
            out = CastDisjointToBatchedAttributes(**cast_disjoint_kwargs)([batched_nodes, out, batch_id_node, node_id])
        else:
            out = CastDisjointToGraphState(**cast_disjoint_kwargs)(out)
    else:
        raise ValueError("Unsupported output embedding for mode `PAiNN`")

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, batched_nodes])
        else:
            out = scaler(out)

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

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
    "equiv_initialize_kwargs": {"dim": 3, "method": "zeros"},
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
                       equiv_initialize_kwargs: dict = None,
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
        equiv_initialize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`EquivariantInitialize` layer.
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
        equiv_input = EquivariantInitialize(**equiv_initialize_kwargs)(z)

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
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])

        if equiv_normalization:
            v = GraphLayerNormalization(axis=2)(v)
        if node_normalization:
            z = GraphBatchNormalization(axis=-1)(z)

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

    model.__kgcnn_model_version__ = __model_version__
    return model
