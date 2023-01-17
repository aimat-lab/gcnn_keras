import tensorflow as tf
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.conv.acsf_conv import ACSFG2, ACSFG4
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.conv.hdnnp_conv import CENTCharge, ElectrostaticEnergyCharge
from kgcnn.layers.mlp import RelationalMLP
from kgcnn.layers.norm import GraphBatchNormalization

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023.01.17"

# Implementation of HDNNP in `tf.keras` from paper:
# A fourth-generation high-dimensional neural network potential with accurate electrostatics including
# non-local charge transfer
# by Tsz Wai Ko, Jonas A. Finkler, Stefan Goedecker and Jörg Behler  (2021)
# https://www.nature.com/articles/s41467-020-20427-2


model_default_behler = {
    "name": "HDNNP4th",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
               {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False}],
    "g2_kwargs": {"eta": [0.0, 0.3], "rs": [0.0, 3.0], "rc": 10.0, "elements": [1, 6, 16]},
    "g4_kwargs": {"eta": [0.0, 0.3], "lamda": [-1.0, 1.0], "rc": 6.0,
                  "zeta": [1.0, 8.0], "elements": [1, 6, 16], "multiplicity": 2.0},
    "normalize_kwargs": {},
    "mlp_charge_kwargs": {"units": [64, 64, 1],
                          "num_relations": 96,
                          "activation": ["swish", "swish", "linear"]},
    "mlp_local_kwargs": {"units": [64, 64, 1],
                         "num_relations": 96,
                         "activation": ["swish", "swish", "linear"]},
    "cent_kwargs": {},
    "electrostatic_kwargs": {},
    "node_pooling_args": {"pooling_method": "sum"},
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": False,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default_behler)
def make_model_behler(inputs: list = None,
                      node_pooling_args: dict = None,
                      name: str = None,
                      verbose: int = None,
                      normalize_kwargs: dict = None,
                      g2_kwargs: dict = None,
                      g4_kwargs: dict = None,
                      mlp_charge_kwargs: dict = None,
                      mlp_local_kwargs: dict = None,
                      cent_kwargs: dict = None,
                      electrostatic_kwargs: dict = None,
                      output_embedding: str = None,
                      use_output_mlp: bool = None,
                      output_to_tensor: bool = None,
                      output_mlp: dict = None
                      ):
    r"""Make 4th generation `HDNNP <https://www.nature.com/articles/s41467-020-20427-2>`_ graph network via
    functional API.

    Default parameters can be found in :obj:`kgcnn.literature.HDNNP4th.model_default_behler` .

    Inputs:
        list: `[node_number, node_coordinates, edge_indices, angle_indices_nodes, total_charge]`

            - node_number (tf.RaggedTensor): Atomic number of shape `(batch, None)` .
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)` .
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)` .
            - angle_indices_nodes (tf.RaggedTensor): Index list for angles of shape `(batch, None, 3)` .
            - total_charge (tf.Tensor): Total charge of each molecule of shape `(batch, 1)` .

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        g2_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFG2` layer.
        g4_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFG4` layer.
        normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphBatchNormalization` layer.
        mlp_charge_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalMLP` layer.
        mlp_local_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalMLP` layer.
        electrostatic_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ElectrostaticEnergyCharge` layer.
        cent_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`CENTCharge` layer.
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
    total_charge_input = ks.layers.Input(**inputs[4])

    # ACSF representation.
    rep_g2 = ACSFG2(**ACSFG2.make_param_table(**g2_kwargs))([node_input, xyz_input, edge_index_input])
    rep_g4 = ACSFG4(**ACSFG4.make_param_table(**g4_kwargs))([node_input, xyz_input, angle_index_input])
    rep = LazyConcatenate()([rep_g2, rep_g4])

    # Normalization
    if normalize_kwargs:
        rep = GraphBatchNormalization(**normalize_kwargs)(rep)

    # learnable NN.
    chi = RelationalMLP(**mlp_charge_kwargs)([rep, node_input])
    q_local = CENTCharge(**cent_kwargs)([node_input, chi, xyz_input, total_charge_input])
    eng_elec = ElectrostaticEnergyCharge(**electrostatic_kwargs)([node_input, q_local, xyz_input, edge_index_input])

    rep_charge = LazyConcatenate()([rep, q_local])
    local_node_energy = RelationalMLP(**mlp_local_kwargs)([rep_charge, node_input])
    eng_short = PoolingNodes(**node_pooling_args)(local_node_energy)

    out = ks.layers.Add()([eng_short, eng_elec])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    else:
        raise ValueError("Unsupported output embedding for mode `HDNNP4th`")

    model = ks.models.Model(
        inputs=[node_input, xyz_input, edge_index_input, angle_index_input, total_charge_input], outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__
    return model
