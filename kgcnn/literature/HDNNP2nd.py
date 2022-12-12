import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.conv.wacsf_conv import wACSFAng, wACSFRad
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.mlp import RelationalMLP

ks = tf.keras

# Implementation of HDNNP in `tf.keras` from paper:
# Atom-centered symmetry functions for constructing high-dimensional neural network potentials
# by Jörg Behler (2011)
# https://aip.scitation.org/doi/abs/10.1063/1.3553717


model_default = {
    "name": "HDNNP2nd",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True}],
    "w_acsf_ang_kwargs": {},
    "w_acsf_rad_kwargs": {},
    "mlp_kwargs": {"units": [64, 64, 64],
                   "num_relations": 96,
                   "activation": ["swish", "swish", "linear"]},
    "node_pooling_args": {"pooling_method": "sum"},
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": False,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               node_pooling_args: dict = None,
               name: str = None,
               verbose: int = None,
               w_acsf_ang_kwargs: dict = None,
               w_acsf_rad_kwargs: dict = None,
               mlp_kwargs: dict = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make 2nd generation `HDNNP <https://arxiv.org/abs/1706.08566>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.HDNNP2nd.model_default`.
    Uses weighted `wACSF <https://arxiv.org/abs/1712.05861>`_ .

    Inputs:
        list: `[node_number, node_coordinates, edge_indices, angle_indices_nodes]`

            - node_number (tf.RaggedTensor): Atomic number of shape `(batch, None)` .
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - angle_indices_nodes (tf.RaggedTensor): Index list for angles of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        w_acsf_ang_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`wACSFAng` layer.
        w_acsf_rad_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`wACSFRad` layer.
        mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalMLP` layer.
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
    edge_index_input = ks.layers.Input(**inputs[2])
    angle_index_input = ks.layers.Input(**inputs[3])

    # ACSF representation.
    rep_rad = wACSFRad(**w_acsf_rad_kwargs)([node_input, xyz_input, edge_index_input])
    rep_ang = wACSFAng(**w_acsf_ang_kwargs)([node_input, xyz_input, angle_index_input])
    rep = LazyConcatenate()([rep_rad, rep_ang])

    # learnable NN.
    n = RelationalMLP(**mlp_kwargs)([rep, node_input])

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)(n)
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `HDNNP2nd`")

    model = ks.models.Model(
        inputs=[node_input, xyz_input, edge_index_input, angle_index_input], outputs=out, name=name)
    return model
