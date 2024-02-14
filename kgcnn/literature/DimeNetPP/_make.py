import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint, model_disjoint_crystal
from kgcnn.layers.modules import Input
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from kgcnn.models.utils import update_model_kwargs
from keras.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-04"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'DimeNetPP' is not supported." % backend_to_use())

# Implementation of DimeNet++ in `keras` from paper:
# Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules
# Johannes Klicpera, Shankari Giri, Johannes T. Margraf, Stephan Günnemann
# https://arxiv.org/abs/2011.14115
# Original code: https://github.com/gasteigerjo/dimenet

model_default = {
    "name": "DimeNetPP",
    "inputs": [
        {"shape": [None], "name": "node_number", "dtype": "int64"},
        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
        {"shape": [None, 2], "name": "angle_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
        {"shape": (), "name": "total_angles", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {
        "input_dim": 95, "output_dim": 128, "embeddings_initializer": {
            "class_name": "RandomUniform",
            "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}
    },
    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
    "cutoff": 5.0, "envelope_exponent": 5,
    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
    "num_targets": 64, "extensive": True, "output_init": "zeros",
    "activation": "swish", "verbose": 10,
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, False],
                   "units": [64, 12], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               emb_size: int = None,
               out_emb_size: int = None,
               int_emb_size: int = None,
               basis_emb_size: int = None,
               num_blocks: int = None,
               num_spherical: int = None,
               num_radial: int = None,
               cutoff: float = None,
               envelope_exponent: int = None,
               num_before_skip: int = None,
               num_after_skip: int = None,
               num_dense_output: int = None,
               num_targets: int = None,
               activation: str = None,
               extensive: bool = None,
               output_init: str = None,
               verbose: int = None,
               name: str = None,
               output_embedding: str = None,
               output_tensor_type: str = None,
               use_output_mlp: bool = None,
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    """Make `DimeNetPP <https://arxiv.org/abs/2011.14115>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DimeNetPP.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, angle_indices...]`
    with '...' indicating mask or ID tensors following the template below.
    Note that you must supply angle indices as index pairs that refer to two edges.

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
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of :obj:`DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in :obj:`SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output :obj:`DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final :obj:`MLP`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation. Note that DimeNetPP originally defines the output dimension
            via `num_targets`. But this can be set to `out_emb_size` and the `output_mlp` be used for more
            specific control.
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
        mask_assignment=[0, 0, 1, 2],
        index_assignment=[None, None, 0, 2]
    )

    n, x, edi, adi, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angles = dj

    out = model_disjoint(
        [n, x, edi, adi, batch_id_node, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        emb_size=emb_size,
        out_emb_size=out_emb_size,
        int_emb_size=int_emb_size,
        basis_emb_size=basis_emb_size,
        num_blocks=num_blocks,
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_dense_output=num_dense_output,
        num_targets=num_targets,
        activation=activation,
        extensive=extensive,
        output_init=output_init,
        use_output_mlp=use_output_mlp,
        output_embedding=output_embedding,
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

model_crystal_default = {
    "name": "DimeNetPP",
    "inputs": [
        {"shape": [None], "name": "node_number", "dtype": "int64", "ragged": True},
        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
        {'shape': (None, 3), 'name': "edge_image", 'dtype': 'int64', 'ragged': True},
        {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
    ],
    "input_tensor_type": "ragged",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {
        "input_dim": 95, "output_dim": 128, "embeddings_initializer": {
            "class_name": "RandomUniform",
            "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}
    },
    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
    "cutoff": 5.0, "envelope_exponent": 5,
    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
    "num_targets": 64, "extensive": True, "output_init": "zeros",
    "activation": "swish", "verbose": 10,
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_tensor_type": "padded",
    "output_scaling": None,
    "output_mlp": {"use_bias": [True, False],
                   "units": [64, 12], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_crystal_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_crystal_model(inputs: list = None,
                       input_tensor_type: str = None,
                       cast_disjoint_kwargs: dict = None,
                       input_embedding: dict = None,
                       input_node_embedding: dict = None,
                       emb_size: int = None,
                       out_emb_size: int = None,
                       int_emb_size: int = None,
                       basis_emb_size: int = None,
                       num_blocks: int = None,
                       num_spherical: int = None,
                       num_radial: int = None,
                       cutoff: float = None,
                       envelope_exponent: int = None,
                       num_before_skip: int = None,
                       num_after_skip: int = None,
                       num_dense_output: int = None,
                       num_targets: int = None,
                       activation: str = None,
                       extensive: bool = None,
                       output_init: str = None,
                       verbose: int = None,
                       name: str = None,
                       output_embedding: str = None,
                       output_tensor_type: str = None,
                       use_output_mlp: bool = None,
                       output_mlp: dict = None,
                       output_scaling: dict = None
                       ):
    """Make `DimeNetPP <https://arxiv.org/abs/2011.14115>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DimeNetPP.model_crystal_default`.

    .. note::

        DimeNetPP does require a large amount of memory for this implementation, which increase quickly with
        the number of connections in a batch. Use ragged input or dataloader if possible.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, coordinates, edge_indices, angle_indices, image_translation, lattice, ...]`
    with '...' indicating mask or ID tensors following the template below.
    Note that you must supply angle indices as index pairs that refer to two edges.

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
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of :obj:`DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in :obj:`SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output :obj:`DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final :obj:`MLP`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation. Note that DimeNetPP originally defines the output dimension
            via `num_targets`. But this can be set to `out_emb_size` and the `output_mlp` be used for more
            specific control.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    disjoint_inputs = template_cast_list_input(
        model_inputs, input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        index_assignment=[None, None, 0, 2, None, None],
        mask_assignment=[0, 0, 1, 2, 1, None]
    )
    n, x, edi, angi, img, lattice, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angles = disjoint_inputs

    # Wrapp disjoint model
    out = model_disjoint_crystal(
        [n, x, edi, angi, img, lattice, batch_id_node, batch_id_edge, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        emb_size=emb_size,
        out_emb_size=out_emb_size,
        int_emb_size=int_emb_size,
        basis_emb_size=basis_emb_size,
        num_blocks=num_blocks,
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_dense_output=num_dense_output,
        num_targets=num_targets,
        activation=activation,
        extensive=extensive,
        output_init=output_init,
        use_output_mlp=use_output_mlp,
        output_embedding=output_embedding,
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


make_crystal_model.__doc__ = make_crystal_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
