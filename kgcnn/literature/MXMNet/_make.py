import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint
from kgcnn.layers.modules import Input
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-09"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'MXMNet' is not supported." % backend_to_use())

# Implementation of MXMNet in `tf.keras` from paper:
# Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures
# by Shuo Zhang, Yang Liu, Lei Xie (2020)
# https://arxiv.org/abs/2011.07457
# https://github.com/zetayue/MXMNet


model_default = {
    "name": "MXMNet",
    "inputs": [
        {"shape": (None, ), "name": "node_number", "dtype": "float32"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 64), "name": "edge_attributes", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (None, 2), "name": "range_indices", "dtype": "int64"},
        {"shape": [None, 2], "name": "angle_indices_1", "dtype": "int64"},
        {"shape": [None, 2], "name": "angle_indices_2", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
        {"shape": (), "name": "total_ranges", "dtype": "int64"},
        {"shape": (), "name": "total_angles_1", "dtype": "int64"},
        {"shape": (), "name": "total_angles_2", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {
        "input_dim": 95, "output_dim": 32,
        "embeddings_initializer": {
            "class_name": "RandomUniform",
            "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}
    },
    "input_edge_embedding": {"input_dim": 32, "output_dim": 32},
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
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True], "units": [1],
                   "activation": ["linear"]},
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_to_tensor": None  # deprecated
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
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
               output_mlp: dict = None,
               output_scaling: dict = None,
               output_tensor_type: str = None,
               ):
    r"""Make `MXMNet <https://arxiv.org/abs/2011.07457>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.MXMNet.model_default` .

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are
    :obj:`[nodes, coordinates, edge_attributes, edge_indices, range_indices, angle_indices_1, angle_indices_2, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Note that you must supply angle indices as index pairs that refer to two edges or two range connections.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of embedding arguments for nodes unpacked in :obj:`Embedding` layers.
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
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor` .
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
        mask_assignment=[0, 0, 1, 1, 2, 3, 4],
        index_assignment=[None, None, None, 0, 0, 3, 3]
    )

    n, x, ed, edi, rgi, adi1, adi2 = dj[:7]
    batch_id_node, batch_id_edge, batch_id_ranges, batch_id_angles_1, batch_id_angles_2 = dj[7:12]
    node_id, edge_id, range_id, angle_id1, angle_id2 = dj[12:17]
    count_nodes, count_edges, count_ranges, count_angles1, count_angles2 = dj[17:]

    out = model_disjoint(
        dj,
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[2]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        bessel_basis_local=bessel_basis_local,
        spherical_basis_local=spherical_basis_local,
        bessel_basis_global=bessel_basis_global,
        use_edge_attributes=use_edge_attributes,
        mlp_rbf_kwargs=mlp_rbf_kwargs,
        mlp_sbf_kwargs=mlp_sbf_kwargs,
        depth=depth,
        global_mp_kwargs=global_mp_kwargs,
        local_mp_kwargs=local_mp_kwargs,
        node_pooling_args=node_pooling_args,
        output_embedding=output_embedding,
        use_output_mlp=use_output_mlp,
        output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        if scaler.extensive:
            # Node information must be numbers, or we need an additional input.
            out = scaler([out, n, batch_id_node])
        else:
            out = scaler(out)

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
