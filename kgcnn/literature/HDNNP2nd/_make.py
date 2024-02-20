import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint_weighted, model_disjoint_behler, model_disjoint_atom_wise
from kgcnn.layers.modules import Input
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-06"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'HDNNP2nd' is not supported." % backend_to_use())

# Implementation of HDNNP in `keras` from paper:
# Atom-centered symmetry functions for constructing high-dimensional neural network potentials
# by Jörg Behler (2011)
# https://aip.scitation.org/doi/abs/10.1063/1.3553717


model_default_weighted = {
    "name": "HDNNP2nd",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
        {"shape": (), "name": "total_angles", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "cast_disjoint_kwargs": {},
    "has_charge_input": False,
    "w_acsf_ang_kwargs": {},
    "w_acsf_rad_kwargs": {},
    "normalize_kwargs": None,
    "const_normalize_kwargs": None,
    "mlp_kwargs": {"units": [64, 64, 64],
                   "num_relations": 96,
                   "activation": ["swish", "swish", "linear"]},
    "node_pooling_args": {"pooling_method": "sum"},
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "predict_dipole": False,
    "use_output_mlp": False,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]},
    "output_tensor_type": "padded",
    "output_scaling": None
}


@update_model_kwargs(model_default_weighted, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model_weighted(inputs: list = None,
                        input_tensor_type: str = None,
                        cast_disjoint_kwargs: dict = None,
                        has_charge_input: bool = False,
                        node_pooling_args: dict = None,
                        name: str = None,
                        verbose: int = None,
                        w_acsf_ang_kwargs: dict = None,
                        w_acsf_rad_kwargs: dict = None,
                        normalize_kwargs: dict = None,
                        const_normalize_kwargs: dict = None,
                        mlp_kwargs: dict = None,
                        output_embedding: str = None,
                        use_output_mlp: bool = None,
                        output_to_tensor: bool = None,
                        predict_dipole: bool = None,
                        output_mlp: dict = None,
                        output_scaling: dict = None,
                        output_tensor_type: str = None
                        ):
    r"""Make 2nd generation `HDNNP <https://arxiv.org/abs/1706.08566>`__ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.HDNNP2nd.model_default_weighted` .
    Uses weighted `wACSF <https://arxiv.org/abs/1712.05861>`__ .

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[node_number, coordinates, edge_indices, angle_indices, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Requires node number for atom-wise neural networks.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        has_charge_input (bool): Whether the model needs total charge as input. Default is False.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        w_acsf_ang_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`wACSFAng` layer.
        w_acsf_rad_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`wACSFRad` layer.
        mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalMLP` layer.
        normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphBatchNormalization` layer.
        const_normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFConstNormalization` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        predict_dipole (bool): Whether to predict additional dipole based on charges. Default is False.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0, 1, 2] + ([None] if has_charge_input else []),
        index_assignment=[None, None, 0, 0] + ([None] if has_charge_input else [])
    )

    if has_charge_input:
        n, x, disjoint_indices, ang_ind, tot_charge, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angle = dj
    else:
        n, x, disjoint_indices, ang_ind, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angle = dj
        tot_charge = None

    out = model_disjoint_weighted(
        [n, x, disjoint_indices, ang_ind, tot_charge, batch_id_node, count_nodes],
        node_pooling_args=node_pooling_args,
        w_acsf_ang_kwargs=w_acsf_ang_kwargs,
        w_acsf_rad_kwargs=w_acsf_rad_kwargs,
        normalize_kwargs=normalize_kwargs,
        const_normalize_kwargs=const_normalize_kwargs,
        mlp_kwargs=mlp_kwargs,
        output_embedding=output_embedding,
        use_output_mlp=use_output_mlp,
        output_mlp=output_mlp,
        predict_dipole=predict_dipole
    )

    if not isinstance(out, list):
        out = [out]

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        # We will only apply scale to first output, i.e. energy.
        out_scaled = out[0]
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out_scaled = scaler([out_scaled, n, batch_id_node])
        else:
            out_scaled = scaler(out_scaled)
        out[0] = out_scaled

    # Output embedding choice
    out = [template_cast_output(
        [out_to_cast, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    ) for out_to_cast in out]

    if len(out) == 1:
        out = out[0]

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model_weighted.__doc__ = make_model_weighted.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)

model_default_behler = {
    "name": "HDNNP2nd",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
        {"shape": (), "name": "total_angles", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "cast_disjoint_kwargs": {},
    "has_charge_input": False,
    "g2_kwargs": {"eta": [0.0, 0.3], "rs": [0.0, 3.0], "rc": 10.0, "elements": [1, 6, 16]},
    "g4_kwargs": {"eta": [0.0, 0.3], "lamda": [-1.0, 1.0], "rc": 6.0,
                  "zeta": [1.0, 8.0], "elements": [1, 6, 16], "multiplicity": 2.0},
    "normalize_kwargs": {},
    "const_normalize_kwargs": None,
    "mlp_kwargs": {"units": [64, 64, 64],
                   "num_relations": 96,
                   "activation": ["swish", "swish", "linear"]},
    "node_pooling_args": {"pooling_method": "sum"},
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": False,
    "predict_dipole": False,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]},
    "output_tensor_type": "padded",
    "output_scaling": None
}


@update_model_kwargs(model_default_behler, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model_behler(inputs: list = None,
                      input_tensor_type: str = None,
                      cast_disjoint_kwargs: dict = None,
                      has_charge_input: bool = None,
                      node_pooling_args: dict = None,
                      name: str = None,
                      verbose: int = None,
                      normalize_kwargs: dict = None,
                      const_normalize_kwargs: dict = None,
                      g2_kwargs: dict = None,
                      g4_kwargs: dict = None,
                      mlp_kwargs: dict = None,
                      output_embedding: str = None,
                      use_output_mlp: bool = None,
                      predict_dipole: bool = None,
                      output_to_tensor: bool = None,
                      output_mlp: dict = None,
                      output_scaling: dict = None,
                      output_tensor_type: str = None
                      ):
    r"""Make 2nd generation `HDNNP <https://arxiv.org/abs/1706.08566>`__ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.HDNNP2nd.model_default_behler` .

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[node_number, coordinates, edge_indices, angle_indices, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Requires node number for atom-wise neural networks.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        has_charge_input (bool): Whether the model needs total charge as input. Default is False.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        g2_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFG2` layer.
        g4_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFG4` layer.
        normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphBatchNormalization` layer.
        const_normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`ACSFConstNormalization` layer.
        mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalMLP` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        predict_dipole (bool): Whether to predict additional dipole based on charges. Default is False.
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0, 1, 2] + ([None] if has_charge_input else []),
        index_assignment=[None, None, 0, 0] + ([None] if has_charge_input else [])
    )

    if has_charge_input:
        n, x, disjoint_indices, ang_index, tot_charge, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angle = dj
    else:
        n, x, disjoint_indices, ang_index, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angle = dj
        tot_charge = None

    out = model_disjoint_behler(
        [n, x, disjoint_indices, ang_index, tot_charge, batch_id_node, count_nodes],
        node_pooling_args=node_pooling_args,
        normalize_kwargs=normalize_kwargs,
        const_normalize_kwargs=const_normalize_kwargs,
        g2_kwargs=g2_kwargs,
        g4_kwargs=g4_kwargs,
        mlp_kwargs=mlp_kwargs,
        output_embedding=output_embedding,
        use_output_mlp=use_output_mlp,
        output_mlp=output_mlp,
        predict_dipole=predict_dipole
    )

    if not isinstance(out, list):
        out = [out]

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        # We will only apply scale to first output, i.e. energy.
        out_scaled = out[0]
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out_scaled = scaler([out_scaled, n, batch_id_node])
        else:
            out_scaled = scaler(out_scaled)
        out[0] = out_scaled

    # Output embedding choice
    out = [template_cast_output(
        [out_to_cast, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    ) for out_to_cast in out]

    if len(out) == 1:
        out = out[0]

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model_behler.__doc__ = make_model_behler.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)

model_default_atom_wise = {
    "name": "HDNNP2nd",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_representation", "dtype": "float32"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
    ],
    "input_tensor_type": "padded",
    "has_charge_input": False,
    "cast_disjoint_kwargs": {},
    "mlp_kwargs": {"units": [64, 64, 64],
                   "num_relations": 96,
                   "activation": ["swish", "swish", "linear"]},
    "node_pooling_args": {"pooling_method": "sum"},
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "predict_dipole": False,
    "use_output_mlp": False,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]},
    "output_tensor_type": "padded",
    "output_scaling": None
}


@update_model_kwargs(model_default_atom_wise, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model_atom_wise(inputs: list = None,
                         input_tensor_type: str = None,
                         cast_disjoint_kwargs: dict = None,
                         has_charge_input: bool = None,
                         node_pooling_args: dict = None,
                         name: str = None,
                         verbose: int = None,
                         mlp_kwargs: dict = None,
                         output_embedding: str = None,
                         predict_dipole: bool = None,
                         use_output_mlp: bool = None,
                         output_to_tensor: bool = None,
                         output_mlp: dict = None,
                         output_scaling: dict = None,
                         output_tensor_type: str = None
                         ):
    r"""Make 2nd generation `HDNNP <https://arxiv.org/abs/1706.08566>`__ network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.HDNNP2nd.model_default_atom_wise` .

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[node_number, node_representation, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Requires node number for atom-wise neural networks.
    The representation are given directly to the model as they are expected to be pre-computed.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        has_charge_input (bool): Whether the model needs total charge as input. Default is False.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalMLP` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        predict_dipole (bool): Whether to predict additional dipole based on charges. Default is False.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 0] + ([None] if has_charge_input else []),
        index_assignment=[None, None] + ([None] if has_charge_input else [])
    )

    if has_charge_input:
        n, x, tot_charge, batch_id_node, node_id, count_nodes = dj
    else:
        n, x, batch_id_node, node_id, count_nodes = dj
        tot_charge = None

    batch_id_edge, edge_id, count_edges = None, None, None

    out = model_disjoint_atom_wise(
        [n, x, tot_charge, batch_id_node, count_nodes],
        node_pooling_args=node_pooling_args,
        mlp_kwargs=mlp_kwargs,
        output_embedding=output_embedding,
        use_output_mlp=use_output_mlp,
        output_mlp=output_mlp,
        predict_dipole=predict_dipole
    )

    if not isinstance(out, list):
        out = [out]

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        # We will only apply scale to first output, i.e. energy.
        out_scaled = out[0]
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out_scaled = scaler([out_scaled, n, batch_id_node])
        else:
            out_scaled = scaler(out_scaled)
        out[0] = out_scaled

    # Output embedding choice
    out = [template_cast_output(
        [out_to_cast, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    ) for out_to_cast in out]

    if len(out) == 1:
        out = out[0]

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model_atom_wise.__doc__ = make_model_atom_wise.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)

# For default, the weighted ACSF are used, since they do should in principle work for all elements.
make_model = make_model_weighted
model_default = model_default_weighted
