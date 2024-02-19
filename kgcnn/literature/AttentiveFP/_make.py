import keras as ks
from kgcnn.layers.scale import get as get_scaler
from kgcnn.models.utils import update_model_kwargs
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
from keras.backend import backend as backend_to_use
from kgcnn.layers.modules import Input
from ._model import model_disjoint

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023.11.15"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]

if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'AttentiveFP' is not supported." % backend_to_use())

# Implementation of AttentiveFP in `keras` from paper:
# Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
# Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li,
# Xiaomin Luo, Kaixian Chen, Hualiang Jiang*, and Mingyue Zheng*
# Cite this: J. Med. Chem. 2020, 63, 16, 8749–8760
# Publication Date:August 13, 2019
# https://doi.org/10.1021/acs.jmedchem.9b00959


model_default = {
    "name": "AttentiveFP",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None,), "name": "edge_number", "dtype": "int64"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    "cast_disjoint_kwargs": {},
    "input_embedding": None,  # deprecated
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
    "attention_args": {"units": 32},
    "depthmol": 2,
    "depthato": 2,
    "dropout": 0.1,
    "verbose": 10,
    "output_embedding": "graph",
    "output_scaling": None,
    "output_to_tensor": True,  # deprecated
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]}
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               cast_disjoint_kwargs: dict = None,
               input_tensor_type: str = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               input_embedding: dict = None,
               depthmol: int = None,
               depthato: int = None,
               dropout: float = None,
               attention_args: dict = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_tensor_type: str = None,
               output_scaling: dict = None,
               output_mlp: dict = None
               ):
    r"""Make `AttentiveFP <https://doi.org/10.1021/acs.jmedchem.9b00959>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.AttentiveFP.model_default`.

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, edges, edge_indices, ...]`
    with '...' indicating mask or id tensors following the template below:

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`Input`. Order must match model definition.
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layers if used.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        input_embedding (dict): Deprecated in favour of `input_node_embedding` etc.
        input_node_embedding (dict): Dictionary of arguments for nodes unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of arguments for edge unpacked in :obj:`Embedding` layers.
        depthato (int): Number of graph embedding units or depth of the network.
        depthmol (int): Number of graph embedding units or depth of the graph embedding.
        dropout (float): Dropout to use.
        attention_args (dict): Dictionary of layer arguments unpacked in :obj:`AttentiveHeadFP` layer. Units parameter
            is also used in GRU-update and :obj:`PoolingNodesAttentive`.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Deprecated in favour of `output_tensor_type` .
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_scaling (dict): Dictionary of layer arguments unpacked in scaling layers.

    Returns:
        :obj:`keras.models.Model`
    """
    # Make input
    model_inputs = [Input(**x) for x in inputs]

    di_inputs = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 1, 1],
        index_assignment=[None, None, 0]
    )

    n, ed, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = di_inputs

    # Wrapping disjoint model.
    out = model_disjoint(
        [n, ed, disjoint_indices, batch_id_node, count_nodes],
        use_node_embedding=("int" in inputs[0]['dtype']) if input_node_embedding is not None else False,
        use_edge_embedding=("int" in inputs[1]['dtype']) if input_edge_embedding is not None else False,
        input_node_embedding=input_node_embedding,
        input_edge_embedding=input_edge_embedding,
        depthmol=depthmol,
        depthato=depthato,
        dropout=dropout,
        attention_args=attention_args,
        output_embedding=output_embedding,
        output_mlp=output_mlp
    )

    if output_scaling is not None:
        scaler = get_scaler(output_scaling["name"])(**output_scaling)
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
