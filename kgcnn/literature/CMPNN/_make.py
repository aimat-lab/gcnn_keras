import keras as ks
from typing import Union
from kgcnn.layers.scale import get as get_scaler
from kgcnn.models.utils import update_model_kwargs
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from keras.backend import backend as backend_to_use
from kgcnn.layers.modules import Input
from ._model import model_disjoint

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023-10-30"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]

if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'CMPNN' is not supported." % backend_to_use())

# Implementation of CMPNN in `keras` from paper:
# Communicative Representation Learning on Attributed Molecular Graphs
# Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu and Yuedong Yang
# https://www.ijcai.org/proceedings/2020/0392.pdf


model_default = {
    "name": "CMPNN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None,), "name": "edge_number", "dtype": "int64"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"},
        {"shape": (), "name": "total_reverse", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "cast_disjoint_kwargs": {},
    'input_embedding': None,  # deprecated
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 20, "output_dim": 64},
    "node_initialize": {"units": 300, "activation": "relu"},
    "edge_initialize":  {"units": 300, "activation": "relu"},
    "edge_dense": {"units": 300, "activation": "linear"},
    "node_dense": {"units": 300, "activation": "linear"},
    "edge_activation": {"activation": "relu"},
    "verbose": 10,
    "depth": 5,
    "dropout": {"rate": 0.1},
    "use_final_gru": True,
    "pooling_gru": {"units": 300},
    "pooling_kwargs": {"pooling_method": "sum"},
    "output_embedding": "graph",
    "output_scaling": None,
    "output_tensor_type": "padded",
    "output_to_tensor": None,  # deprecated
    "output_mlp": {"use_bias": [True, True, False], "units": [300, 100, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(name: str = None,
               inputs: list = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               edge_initialize: dict = None,
               node_initialize: dict = None,
               edge_dense: dict = None,
               node_dense: dict = None,
               edge_activation: dict = None,
               depth: int = None,
               dropout: Union[dict, None] = None,
               verbose: int = None,
               use_final_gru: bool = True,
               pooling_gru: dict = None,
               pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_tensor_type: str = None,
               output_mlp: dict = None,
               output_scaling: dict = None
               ):
    r"""Make `CMPNN <https://www.ijcai.org/proceedings/2020/0392.pdf>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.CMPNN.model_default` .

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, edges, edge_indices, reverse_indices, ...]`
    with '...' indicating mask or id tensors following the template below.
    Here, reverse indices are in place of angle indices and refer to edges.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        name (str): Name of the model. Should be "CMPNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_embedding (dict): Deprecated in favour of input_node_embedding etc.
        input_node_embedding (dict): Dictionary of arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        node_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first node embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge communicate.
        node_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for node communicate.
        edge_activation (dict): Dictionary of layer arguments unpacked in :obj:`Activation` layer for edge communicate.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level for print information.
        dropout (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            :obj:`AggregateLocalEdges` layers.
        use_final_gru (bool): Whether to use GRU for final readout.
        pooling_gru (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodesGRU`.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): WDeprecated in favour of `output_tensor_type` .
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Kwargs for scaling layer, if scaling layer is to be used.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    di = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 1, 1, 2],
        index_assignment=[None, None, 0, 2]
    )

    n, ed, edi, e_pairs, batch_id_node, batch_id_edge, _,  node_id, edge_id, _, count_nodes, count_edges, _ = di

    # Wrapping disjoint model.
    out = model_disjoint(
        [n, ed, edi, e_pairs, batch_id_node, node_id, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        node_initialize=node_initialize,
        edge_initialize=edge_initialize,
        depth=depth,
        pooling_kwargs=pooling_kwargs,
        edge_dense=edge_dense,
        edge_activation=edge_activation,
        dropout=dropout,
        node_dense=node_dense,
        output_embedding=output_embedding,
        use_final_gru=use_final_gru,
        output_mlp=output_mlp,
        pooling_gru=pooling_gru
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
    model.__kgcnn_model_version__ = __model_version__

    if output_scaling is not None:
        def set_scale(*args, **kwargs):
            scaler.set_scale(*args, **kwargs)

        setattr(model, "set_scale", set_scale)

    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
