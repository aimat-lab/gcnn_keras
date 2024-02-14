import keras as ks
from kgcnn.layers.scale import get as get_scaler
from ._model import model_disjoint
from kgcnn.layers.modules import Input
from kgcnn.models.utils import update_model_kwargs
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from keras.backend import backend as backend_to_use


# Keep track of model version from commit date in literature.
__kgcnn_model_version__ = "2023-12-04"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'GNNFilm' is not supported." % backend_to_use())

# Implementation of GNNFilm in `keras` from paper:
# GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation
# Marc Brockschmidt
# https://arxiv.org/abs/1906.12192


model_default = {
    "name": "GNNFilm",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "int64"},
        {"shape": (None, ), "name": "edge_relations", "dtype": "int64"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,  # deprecated
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "dense_relation_kwargs": {"units": 64, "num_relations": 20},
    "dense_modulation_kwargs": {"units": 64, "num_relations": 20, "activation": "sigmoid"},
    "activation_kwargs": {"activation": "swish"},
    "depth": 3,
    "verbose": 10,
    "node_pooling_kwargs": {},
    "output_embedding": 'graph',
    "output_scaling": None,
    "output_tensor_type": "padded",
    "output_to_tensor": None,  # deprecated
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"}
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               depth: int = None,
               dense_relation_kwargs: dict = None,
               dense_modulation_kwargs: dict = None,
               activation_kwargs: dict = None,
               name: str = None,
               verbose: int = None,
               node_pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_scaling: dict = None,
               output_tensor_type: str = None,
               output_mlp: dict = None
               ):
    r"""Make `GNNFilm <https://arxiv.org/abs/1906.12192>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.RGCN.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, edge_relations, edge_indices, ...]`
    with '...' indicating mask or ID tensors following the template below.
    The edge relations do not have a feature dimension and specify the relation of each edge of type 'int'.
    Edges are actually edge single weight values which are entries of the pre-scaled adjacency matrix.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        dense_relation_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalDense` layer.
        dense_modulation_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalDense` layer.
        activation_kwargs (dict):  Dictionary of layer arguments unpacked in :obj:`Activation` layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
        node_pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layer.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers. Default is None.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    dj_inputs = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 1, 1],
        index_assignment=[None, None, 0]
    )

    n, er, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = dj_inputs

    out = model_disjoint(
        [n, er, disjoint_indices, batch_id_node, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        depth=depth,
        dense_modulation_kwargs=dense_modulation_kwargs,
        dense_relation_kwargs=dense_relation_kwargs,
        activation_kwargs=activation_kwargs,
        output_embedding=output_embedding,
        output_mlp=output_mlp,
        node_pooling_kwargs=node_pooling_kwargs
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
        out = scaler(out)

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __kgcnn_model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)
    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
