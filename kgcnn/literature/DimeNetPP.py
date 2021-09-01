import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.conv.dimenet_conv import DimNetInteractionPPBlock, DimNetOutputBlock
from kgcnn.layers.embedding import EmbeddingDimeBlock
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.geom import SphericalBasisLayer, NodeDistance, EdgeAngle, BesselBasisLayer
from kgcnn.layers.keras import Dense, Concatenate, Add
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.utils.models import update_model_kwargs

# Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules
# Johannes Klicpera, Shankari Giri, Johannes T. Margraf, Stephan Günnemann
# https://arxiv.org/abs/2011.14115

model_default = {"name": "DimeNetPP",
                 "inputs": [{"shape": [None], "name": "node_attributes", "dtype": "float32", "ragged": True},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                            {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True}],
                 "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                              "embeddings_initializer": {"class_name": "RandomUniform",
                                                                         "config": {"minval": -1.7320508075688772,
                                                                                    "maxval": 1.7320508075688772}}}},
                 "output_embedding": "graph",
                 "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
                 "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
                 "cutoff": 5.0, "envelope_exponent": 5,
                 "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
                 "num_targets": 12, "extensive": True, "output_init": "zeros",
                 "activation": "swish", "verbose": 1,
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               output_embedding=None,
               emb_size=None,
               out_emb_size=None,
               int_emb_size=None,
               basis_emb_size=None,
               num_blocks=None,
               num_spherical=None,
               num_radial=None,
               cutoff=None,
               envelope_exponent=None,
               num_before_skip=None,
               num_after_skip=None,
               num_dense_output=None,
               num_targets=None,
               activation=None,
               extensive=None,
               output_init=None,
               **kwargs):
    """Make DimeNetPP graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Note: DimeNetPP does require a large amount of memory for this implementation, which increase quickly with
        the number of connections in a batch.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of `DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in `SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output `DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    angle_index_input = ks.layers.Input(**inputs[3])

    # Atom embedding
    # n = generate_node_embedding(node_input, input_node_shape, input_embedding["nodes"])
    if len(inputs[0]["shape"]) == 1:
        n = EmbeddingDimeBlock(**input_embedding["node"])(node_input)
    else:
        n = node_input

    x = xyz_input
    edi = bond_index_input
    adi = angle_index_input

    # Calculate distances
    d = NodeDistance()([x, edi])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    a = EdgeAngle()([x, edi, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = Concatenate(axis=-1)([n_pairs, rbf_emb])
    x = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = Add()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])
        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        main_output = PoolingNodes(pooling_method="sum")(ps)
    else:
        main_output = PoolingNodes(pooling_method="mean")(ps)

    if output_embedding != "graph":
        raise ValueError("Unsupported graph embedding for mode `DimeNetPP`.")

    model = tf.keras.models.Model(inputs=[node_input, xyz_input, bond_index_input, angle_index_input],
                                  outputs=main_output)

    return model
