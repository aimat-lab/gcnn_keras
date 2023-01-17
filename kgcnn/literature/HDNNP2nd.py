import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.conv.wacsf_conv import wACSFAng, wACSFRad
from kgcnn.layers.conv.acsf_conv import ACSFG2, ACSFG4
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from kgcnn.layers.mlp import RelationalMLP
from kgcnn.layers.norm import GraphBatchNormalization

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023.01.17"

# Implementation of HDNNP in `tf.keras` from paper:
# Atom-centered symmetry functions for constructing high-dimensional neural network potentials
# by Jörg Behler (2011)
# https://aip.scitation.org/doi/abs/10.1063/1.3553717


model_default_weighted = {
    "name": "HDNNP2nd",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True}],
    "w_acsf_ang_kwargs": {},
    "w_acsf_rad_kwargs": {},
    "normalize_kwargs": None,
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


@update_model_kwargs(model_default_weighted)
def make_model_weighted(inputs: list = None,
                        node_pooling_args: dict = None,
                        name: str = None,
                        verbose: int = None,
                        w_acsf_ang_kwargs: dict = None,
                        w_acsf_rad_kwargs: dict = None,
                        normalize_kwargs: dict = None,
                        mlp_kwargs: dict = None,
                        output_embedding: str = None,
                        use_output_mlp: bool = None,
                        output_to_tensor: bool = None,
                        output_mlp: dict = None
                        ):
    r"""Make 2nd generation `HDNNP <https://arxiv.org/abs/1706.08566>`_ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.HDNNP2nd.model_default_weighted`.
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
        normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphBatchNormalization` layer.
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

    # Normalization
    if normalize_kwargs:
        rep = GraphBatchNormalization(**normalize_kwargs)(rep)

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
    model.__kgcnn_model_version__ = __model_version__
    return model


model_default_behler = {
    "name": "HDNNP2nd",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True}],
    "g2_kwargs": {"eta": [0.0, 0.3], "rs": [0.0, 3.0], "rc": 10.0, "elements": [1, 6, 16]},
    "g4_kwargs": {"eta": [0.0, 0.3], "lamda": [-1.0, 1.0], "rc": 6.0,
                  "zeta": [1.0, 8.0], "elements": [1, 6, 16], "multiplicity": 2.0},
    "normalize_kwargs": {},
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


@update_model_kwargs(model_default_behler)
def make_model_behler(inputs: list = None,
                      node_pooling_args: dict = None,
                      name: str = None,
                      verbose: int = None,
                      normalize_kwargs: dict = None,
                      g2_kwargs: dict = None,
                      g4_kwargs: dict = None,
                      mlp_kwargs: dict = None,
                      output_embedding: str = None,
                      use_output_mlp: bool = None,
                      output_to_tensor: bool = None,
                      output_mlp: dict = None
                      ):
    r"""Make 2nd generation `HDNNP <https://arxiv.org/abs/1706.08566>`_ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.HDNNP2nd.model_default_behler`.

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
        g2_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFG2` layer.
        g4_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFG4` layer.
        normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphBatchNormalization` layer.
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
    rep_g2 = ACSFG2(**ACSFG2.make_param_table(**g2_kwargs))([node_input, xyz_input, edge_index_input])
    rep_g4 = ACSFG4(**ACSFG4.make_param_table(**g4_kwargs))([node_input, xyz_input, angle_index_input])
    rep = LazyConcatenate()([rep_g2, rep_g4])

    # Normalization
    if normalize_kwargs:
        rep = GraphBatchNormalization(**normalize_kwargs)(rep)

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

    model.__kgcnn_model_version__ = __model_version__
    return model


model_atom_wise_default = {
    "name": "HDNNP2nd",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_representation", "dtype": "float32", "ragged": True}],
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


@update_model_kwargs(model_atom_wise_default)
def make_model_atom_wise(inputs: list = None,
                         node_pooling_args: dict = None,
                         name: str = None,
                         verbose: int = None,
                         mlp_kwargs: dict = None,
                         output_embedding: str = None,
                         use_output_mlp: bool = None,
                         output_to_tensor: bool = None,
                         output_mlp: dict = None
                         ):
    r"""Make 2nd generation `HDNNP <https://arxiv.org/abs/1706.08566>`_ network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.HDNNP2nd.model_atom_wise_default`.

    Inputs:
        list: `[node_number, node_representation]`

            - node_number (tf.RaggedTensor): Atomic number of shape `(batch, None)` .
            - node_representation (tf.RaggedTensor): Node (atomic) features of shape `(batch, None, F)`

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
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
    rep_input = ks.layers.Input(**inputs[1])

    # learnable NN.
    n = RelationalMLP(**mlp_kwargs)([rep_input, node_input])

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
        inputs=[node_input, rep_input], outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__
    return model


# For default, the weighted ACSF are used, since they do should in principle work for all elements.
make_model = make_model_weighted

