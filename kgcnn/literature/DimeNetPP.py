import tensorflow as tf
import tensorflow.keras as ks
import pprint

from kgcnn.utils.models import update_model_kwargs_logic
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.geom import SphericalBasisLayer, NodeDistance, EdgeAngle, BesselBasisLayer
from kgcnn.layers.keras import Dense, Concatenate, Add
from kgcnn.layers.conv.dimenet_conv import DimNetInteractionPPBlock, DimNetOutputBlock
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.layers.embedding import EmbeddingDimeBlock

# Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules
# Johannes Klicpera, Shankari Giri, Johannes T. Margraf, Stephan Günnemann
# https://arxiv.org/abs/2011.14115


def make_dimnet_pp(**kwargs):
    """Make DimeNet++ network.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: DimeNet++ model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 128,
                                                   'embeddings_initializer': {'class_name': 'RandomUniform',
                                                   'config': {'minval': -1.7320508075688772,
                                                              'maxval': 1.7320508075688772}}}},
                     'emb_size': 128, 'out_emb_size': 256, 'int_emb_size': 64, 'basis_emb_size': 8,
                     'num_blocks': 4, 'num_spherical': 7, 'num_radial': 6,
                     'cutoff': 5.0, 'envelope_exponent': 5,
                     'num_before_skip': 1, 'num_after_skip': 2, 'num_dense_output': 3,
                     'num_targets': 12, 'extensive': True, 'output_init': 'zeros',
                     'activation': 'swish', 'verbose': 1,
                     }
    m = update_model_kwargs_logic(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO:kgcnn: Updated functional make model kwargs:")
        pprint.pprint(m)

    # Update model parameters
    input_node_shape = m['input_node_shape']
    input_embedding = m['input_embedding']
    emb_size = m['emb_size']
    out_emb_size = m['out_emb_size']
    int_emb_size = m['int_emb_size']
    basis_emb_size = m['basis_emb_size']
    num_blocks = m['num_blocks']
    num_spherical = m['num_spherical']
    num_radial = m['num_radial']
    cutoff = m['cutoff']
    envelope_exponent = m['envelope_exponent']
    num_before_skip = m['num_before_skip']
    num_after_skip = m['num_after_skip']
    num_dense_output = m['num_dense_output']
    num_targets = m['num_targets']
    activation = m['activation']
    extensive = m['extensive']
    output_init = m['output_init']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    xyz_input = ks.layers.Input(shape=[None, 3], name='xyz_input', dtype="float32", ragged=True)
    bond_index_input = ks.layers.Input(shape=[None, 2], name='bond_index_input', dtype="int64", ragged=True)
    angle_index_input = ks.layers.Input(shape=[None, 2], name='angle_index_input', dtype="int64", ragged=True)

    # Atom embedding
    # n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    if len(input_node_shape) == 1:
        n = EmbeddingDimeBlock(**input_embedding['nodes'])(node_input)
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

    model = tf.keras.models.Model(inputs=[node_input, xyz_input, bond_index_input, angle_index_input],
                                  outputs=main_output)

    return model
