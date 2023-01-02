import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.geom import NodeDistanceEuclidean, EdgeAngle, BesselBasisLayer, NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import Dense, OptionalInputEmbedding, LazyConcatenate, LazySubtract, LazyAdd
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.conv.dimenet_conv import SphericalBasisLayer, EmbeddingDimeBlock
from kgcnn.layers.conv.mxmnet_conv import MXMGlobalMP, MXMLocalMP

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of MXMNet in `tf.keras` from paper:
# Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures
# by Shuo Zhang, Yang Liu, Lei Xie (2020)
# https://arxiv.org/abs/2011.07457
# https://github.com/zetayue/MXMNet


model_default = {
    "name": "MXMNet",
    "inputs": [{"shape": (None, ), "name": "node_number", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, ), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": [None, 2], "name": "angle_indices_1", "dtype": "int64", "ragged": True},
               {"shape": [None, 2], "name": "angle_indices_2", "dtype": "int64", "ragged": True},
               {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 32,
                                 "embeddings_initializer": {
                                     "class_name": "RandomUniform",
                                     "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}},
                        "edge": {"input_dim": 32, "output_dim": 32}},
    "bessel_basis_local": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},
    "bessel_basis_global": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},  # Should match range_indices
    "spherical_basis_local": {"num_spherical": 7, "num_radial": 6, "cutoff": 5.0, "envelope_exponent": 5},
    "mlp_rbf_kwargs": {"units": 32, "activation": "swish"},
    "mlp_sbf_kwargs": {"units": 32, "activation": "swish"},
    "global_mp_kwargs": {"units": 32},
    "local_mp_kwargs": {"units": 32, "output_units": 1, "output_kernel_initializer": "zeros"},
    "use_edge_attributes": False,
    "depth": 3,
    "verbose": 10,
    "node_pooling_args": {"pooling_method": "sum"},
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True], "units": [1],
                   "activation": ["linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               name: str = None,
               bessel_basis_local: dict = None,
               bessel_basis_global: dict = None,
               spherical_basis_local: dict = None,
               use_edge_attributes: bool = None,
               mlp_rbf_kwargs: dict = None,
               mlp_sbf_kwargs: dict = None,
               global_mp_kwargs: dict = None,
               local_mp_kwargs: dict = None,
               verbose: int = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               node_pooling_args: dict = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `MXMNet <https://arxiv.org/abs/2011.07457>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.MXMNet.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, edge_attributes, edge_indices, angle_indices_1, angle_indices_2,
         range_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for (local) edges of shape `(batch, None, 2)`.
            - range_indices (tf.RaggedTensor): Index list for (global) range-like edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - angle_indices_1 (tf.RaggedTensor): Index list of angles referring to (local) edge connections of
              shape `(batch, None, 2)`. Angles of edge pairing (ij, jk).
            - angle_indices_2 (tf.RaggedTensor): Index list of angles referring to (local) edge connections of
              shape `(batch, None, 2)`.  Angles of edge pairing (ij, ik).

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        bessel_basis_local: Dictionary of layer arguments unpacked in local `:obj:BesselBasisLayer` layer.
        bessel_basis_global: Dictionary of layer arguments unpacked in global `:obj:BesselBasisLayer` layer.
        spherical_basis_local: Dictionary of layer arguments unpacked in `:obj:SphericalBasisLayer` layer.
        use_edge_attributes: Whether to add edge attributes. Default is False.
        mlp_rbf_kwargs: Dictionary of layer arguments unpacked in `:obj:MLP` layer for RBF feed-forward.
        mlp_sbf_kwargs: Dictionary of layer arguments unpacked in `:obj:MLP` layer for SBF feed-forward.
        global_mp_kwargs: Dictionary of layer arguments unpacked in `:obj:MXMGlobalMP` layer.
        local_mp_kwargs: Dictionary of layer arguments unpacked in `:obj:MXMLocalMP` layer.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_input = ks.layers.Input(**inputs[2])
    edge_index_input = ks.layers.Input(**inputs[3])
    angle_index_input_1 = ks.layers.Input(**inputs[4])
    angle_index_input_2 = ks.layers.Input(**inputs[5])
    range_index_input = ks.layers.Input(**inputs[6])

    # Rename to short names and make embedding, if no feature dimension.
    x = xyz_input
    n = EmbeddingDimeBlock(**input_embedding["node"])(node_input) if len(inputs[0]["shape"]) < 2 else node_input
    ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[2]['shape']) < 2)(edge_input)
    ei_l = edge_index_input
    ri_g = range_index_input
    ai_1 = angle_index_input_1
    ai_2 = angle_index_input_2

    # Calculate distances and spherical and bessel basis for local edges including angles.
    # For the first version, we restrict ourselves to 2-hop angles.
    pos1_l, pos2_l = NodePosition()([x, ei_l])
    d_l = NodeDistanceEuclidean()([pos1_l, pos2_l])
    rbf_l = BesselBasisLayer(**bessel_basis_local)(d_l)
    v12_l = LazySubtract()([pos1_l, pos2_l])
    a_l_1 = EdgeAngle()([v12_l, ai_1])
    a_l_2 = EdgeAngle(vector_scale=[1.0, -1.0])([v12_l, ai_2])
    sbf_l_1 = SphericalBasisLayer(**spherical_basis_local)([d_l, a_l_1, ai_1])
    sbf_l_2 = SphericalBasisLayer(**spherical_basis_local)([d_l, a_l_2, ai_2])

    # Calculate distance and bessel basis for global (range) edges.
    pos1_g, pos2_g = NodePosition()([x, ri_g])
    d_g = NodeDistanceEuclidean()([pos1_g, pos2_g])
    rbf_g = BesselBasisLayer(**bessel_basis_global)(d_g)

    if use_edge_attributes:
        rbf_l = LazyConcatenate()([rbf_l, ed])

    rbf_l = GraphMLP(**mlp_rbf_kwargs)(rbf_l)
    sbf_l_1 = GraphMLP(**mlp_sbf_kwargs)(sbf_l_1)
    sbf_l_2 = GraphMLP(**mlp_sbf_kwargs)(sbf_l_2)
    rbf_g = GraphMLP(**mlp_rbf_kwargs)(rbf_g)

    # Model
    h = n
    nodes_list = []
    for i in range(0, depth):
        h = MXMGlobalMP(**global_mp_kwargs)([h, rbf_g, ri_g])
        h, t = MXMLocalMP(**local_mp_kwargs)([h, rbf_l, sbf_l_1, sbf_l_2, ei_l, ai_1, ai_2])
        nodes_list.append(t)

    # Output embedding choice
    out = LazyAdd()(nodes_list)
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)(out)
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `MXMNet`")

    model = ks.models.Model(
        inputs=[node_input, xyz_input, edge_input, edge_index_input, angle_index_input_1, angle_index_input_2,
                range_index_input],
        outputs=out,
        name=name
    )

    model.__kgcnn_model_version__ = __model_version__
    return model
