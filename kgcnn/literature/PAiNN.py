import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.painn_conv import PAiNNUpdate, EquivariantInitialize
from kgcnn.layers.conv.painn_conv import PAiNNconv
from kgcnn.layers.geom import NodeDistance, BesselBasisLayer, EdgeDirectionNormalized
from kgcnn.layers.keras import Add
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.utils.models import update_model_kwargs, generate_embedding

# Equivariant message passing for the prediction of tensorial properties and molecular spectra
# Kristof T. Schuett, Oliver T. Unke and Michael Gastegger
# https://arxiv.org/pdf/2102.03150.pdf

model_default = {"name": "PAiNN",
                 "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
                            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
                 "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
                 "output_embedding": "graph",
                 "output_mlp": {"use_bias": [True, True], "units": [128, 1],
                                "activation": ["swish", "linear"]},
                 "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                 "pooling_args": {"pooling_method": "sum"},
                 "conv_args": {"units": 128, "cutoff": None},
                 "update_args": {"units": 128},
                 "depth": 3,
                 "verbose": 1
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               bessel_basis=None,
               depth=None,
               output_embedding=None,
               pooling_args=None,
               output_mlp=None,
               conv_args=None,
               update_args=None, **kwargs):
    """Make PAiNN graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        bessel_basis (dict): Dictionary of layer arguments unpacked in final `BesselBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        pooling_args (dict): Dictionary of layer arguments unpacked in `PoolingNodes` layer.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.
        conv_args (dict): Dictionary of layer arguments unpacked in `PAiNNconv` layer.
        update_args (dict): Dictionary of layer arguments unpacked in `PAiNNUpdate` layer.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    z = generate_embedding(node_input, inputs[0]["shape"], input_embedding["node"])

    if len(inputs) > 3:
        equiv_input = ks.layers.Input(**inputs[3])
    else:
        equiv_input = EquivariantInitialize(dim=3)(z)

    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    rij = EdgeDirectionNormalized()([x, edi])
    d = NodeDistance()([x, edi])
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, rij, edi])
        z = Add()([z, ds])
        v = Add()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = Add()([z, ds])
        v = Add()([v, dv])
    n = z
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)(n)
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(main_output)
        # no ragged for distribution atm
    else:
        raise ValueError("Unsupported graph embedding for mode `PAiNN`")

    if len(inputs) > 3:
        model = tf.keras.models.Model(inputs=[node_input, xyz_input, bond_index_input, equiv_input],
                                      outputs=main_output)
    else:
        model = tf.keras.models.Model(inputs=[node_input, xyz_input, bond_index_input],
                                      outputs=main_output)
    return model
