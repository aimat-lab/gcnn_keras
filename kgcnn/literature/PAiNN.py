import tensorflow.keras as ks
import pprint
import tensorflow as tf
from kgcnn.utils.models import update_model_args, generate_node_embedding
from kgcnn.layers.keras import Add
from kgcnn.layers.geom import NodeDistance, BesselBasisLayer, EdgeDirectionNormalized
from kgcnn.layers.conv import PAiNNconv
from kgcnn.layers.update import PAiNNUpdate
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers.casting import ChangeTensorType

# First Version not (fully tested and a few things will be changed)
# Equivariant message passing for the prediction of tensorial properties and molecular spectra
# Kristof T. Schuett, Oliver T. Unke and Michael Gastegger
# https://arxiv.org/pdf/2102.03150.pdf


def make_painn(**kwargs):
    """Get PAiNN model.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: PAiNN keras model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_equiv_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 128}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True], "units": [128, 1],
                                    "activation": ['swish', 'linear']},
                     'bessel_basis': {'num_radial': 20, 'cutoff': 5.0, 'envelope_exponent': 5},
                     'pooling_args': {'pooling_method': 'sum'},
                     'depth': 3,
                     'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # local variables
    input_node_shape = m['input_node_shape']
    input_equiv_shape = m['input_equiv_shape']
    input_embedding = m['input_embedding']
    bessel_basis = m['bessel_basis']
    depth = m['depth']
    output_embedding = m['output_embedding']
    pooling_args = m['pooling_args']
    output_mlp = m['output_mlp']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    equiv_input = ks.layers.Input(shape=input_equiv_shape, name='equiv_input', dtype="float32", ragged=True)
    xyz_input = ks.layers.Input(shape=[None, 3], name='xyz_input', dtype="float32", ragged=True)
    bond_index_input = ks.layers.Input(shape=[None, 2], name='bond_index_input', dtype="int64", ragged=True)

    # Embedding
    z = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    rij = EdgeDirectionNormalized()([x, edi])
    d = NodeDistance()([x, edi])
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(units=128)([z, v, rbf, rij, edi])
        z = Add()([z, ds])
        v = Add()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(units=128)([z, v])
        z = Add()([z, ds])
        v = Add()([v, dv])
    n = z
    # Output embedding choice
    if output_embedding["output_mode"] == 'graph':
        out = PoolingNodes(**pooling_args)(n)
        main_output = MLP(**output_mlp)(out)
    else:  # Node labeling
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(main_output)
        # no ragged for distribution atm

    model = tf.keras.models.Model(inputs=[node_input, equiv_input, xyz_input, bond_index_input],
                                  outputs=main_output)

    return model