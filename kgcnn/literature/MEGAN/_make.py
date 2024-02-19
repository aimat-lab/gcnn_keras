import keras as ks
from ._model import MEGAN
from kgcnn.models.utils import update_model_kwargs
from kgcnn.layers.modules import Input
from keras.backend import backend as backend_to_use
from kgcnn.models.casting import (template_cast_output, template_cast_list_input,
                                  template_cast_list_input_docs, template_cast_output_docs)
import kgcnn.ops.activ

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-08"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'MEGAN' is not supported." % backend_to_use())


# Implementation of INorp in `tf.keras` from paper:
# 'Interaction Networks for Learning about Objects, Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://arxiv.org/abs/1612.00222
# https://github.com/higgsfield/interaction_network_pytorch


model_default = {
    "name": "MEGAN",
    "inputs": [
        {'shape': (None, 128), 'name': "node_attributes", 'dtype': 'float32'},
        {'shape': (None, 64), 'name': "edge_attributes", 'dtype': 'float32'},
        {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64'},
        {'shape': (1,), 'name': "graph_labels", 'dtype': 'float32'},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "input_tensor_type": "padded",
    'input_embedding': None,
    "cast_disjoint_kwargs": {},
    "units": [128],
    "activation": {"class_name": "function", "config": "kgcnn>leaky_relu2"},
    "use_bias":  True,
    "dropout_rate":  0.0,
    "use_edge_features":  True,
    "input_node_embedding":  None,
    # node/edge importance related arguments
    "importance_units":  [],
    "importance_channels":  2,
    "importance_activation":  "sigmoid",  # do not change
    "importance_dropout_rate":  0.0,  # do not change
    "importance_factor":  0.0,
    "importance_multiplier":  10.0,
    "sparsity_factor":  0.0,
    "concat_heads":  True,
    # mlp tail end related arguments
    "final_units":  [1],
    "final_dropout_rate":  0.0,
    "final_activation":  'linear',
    "final_pooling":  'sum',
    "regression_limits":  None,
    "regression_reference": None,
    "return_importances": True,
    "output_tensor_type": "padded",
    'output_embedding': 'graph',
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(inputs: list = None,
               name: str = None,
               input_tensor_type: str = None,
               cast_disjoint_kwargs: dict = None,
               units: list = None,
               activation: str = None,
               use_bias: bool = None,
               dropout_rate: float = None,
               use_edge_features: bool = None,
               input_embedding: dict = None,  # deprecated
               input_node_embedding: dict = None,
               # node/edge importance related arguments
               importance_units: list = None,
               importance_channels: int = None,
               importance_activation: str = None,  # do not change
               importance_dropout_rate: float = None,  # do not change
               importance_factor: float = None,
               importance_multiplier: float = None,
               sparsity_factor: float = None,
               concat_heads: bool = None,
               # mlp tail end related arguments
               final_units: list = None,
               final_dropout_rate: float = None,
               final_activation: str = None,
               final_pooling: str = None,
               regression_limits: tuple = None,
               regression_reference: float = None,
               return_importances: bool = True,
               output_embedding: str = None,
               output_tensor_type: str = None
               ):
    r"""Functional model definition of MEGAN. Please check documentation of :obj:`kgcnn.literature.MEGAN` .

    **Model inputs**:
    Model uses the list template of inputs and standard output template.
    The supported inputs are  :obj:`[nodes, edges, edge_indices, graph_labels...]`
    with '...' indicating mask or ID tensors following the template below.
    Graph labels are used to generate explanations but not to influence model output.

    %s

    **Model outputs**:
    The standard output template:

    %s

    Args:
        name: Name of the model.
        inputs (list): List of dictionaries unpacked in :obj:`keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input type of graph tensor. Default is "padded".
        cast_disjoint_kwargs (dict): Dictionary of arguments for casting layer.
        units: A list of ints where each element configures an additional attention layer. The numeric
                value determines the number of hidden units to be used in the attention heads of that layer
        activation: The activation function to be used within the attention layers of the network
        use_bias: Whether the layers of the network should use bias weights at all
        dropout_rate: The dropout rate to be applied after *each* of the attention layers of the network.
        input_node_embedding: Dictionary of embedding kwargs for input embedding layer.
        use_edge_features: Whether edge features should be used. Generally the network supports the
            usage of edge features, but if the input data does not contain edge features, this should be
            set to False.
        importance_units: A list of ints where each element configures another dense layer in the
            subnetwork that produces the node importance tensor from the main node embeddings. The
            numeric value determines the number of hidden units in that layer.
        importance_channels: The int number of explanation channels to be produced by the network. This
            is the value referred to as "K". Note that this will also determine the number of attention
            heads used within the attention subnetwork.
        importance_factor: The weight of the explanation-only train step. If this is set to exactly
            zero then the explanation train step will not be executed at all (less computationally
            expensive)
        importance_multiplier: An additional hyperparameter of the explanation-only train step. This
            is essentially the scaling factor that is applied to the values of the dataset such that
            the target values can reasonably be approximated by a sum of [0, 1] importance values.
        sparsity_factor: The coefficient for the sparsity regularization of the node importance
            tensor.
        concat_heads: Whether to concat the heads of the attention subnetwork. The default is True. In
            that case the output of each individual attention head is concatenated and the concatenated
            vector is then used as the input of the next attention layer's heads. If this is False, the
            vectors are average pooled instead.
        final_units: A list of ints where each element configures another dense layer in the MLP
            at the tail end of the network. The numeric value determines the number of the hidden units
            in that layer. Note that the final element in this list has to be the same as the dimension
            to be expected for the samples of the training dataset!
        final_dropout_rate: The dropout rate to be applied after *every* layer of the final MLP.
        final_activation: The activation to be applied at the very last layer of the MLP to produce the
            actual output of the network.
        final_pooling: The pooling method to be used during the global pooling phase in the network.
        regression_limits: A tuple where the first value is the lower limit for the expected value range
            of the regression task and teh second value the upper limit.
        regression_reference: A reference value which is inside the range of expected values (best if
            it was in the middle, but does not have to). Choosing different references will result
            in different explanations.
        return_importances: Whether the importance / explanation tensors should be returned as an output
            of the model. If this is True, the output of the model will be a 3-tuple:
            (output, node importances, edge importances), otherwise it is just the output itself
        output_tensor_type (str): Output type of graph tensors such as nodes or edges. Default is "padded".
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".

    Returns:
        :obj:`keras.models.Model`
    """
    model_inputs = [Input(**x) for x in inputs]

    di_inputs = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        mask_assignment=[0, 1, 1, None],
        index_assignment=[None, None, 0, None]
    )

    n, ed, disjoint_indices, gs, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = di_inputs

    # Wrapping disjoint model.
    out = MEGAN(
        units=units,
        activation=activation,
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        use_edge_features=use_edge_features,
        input_node_embedding=input_node_embedding,
        importance_units=importance_units,
        importance_channels=importance_channels,
        importance_activation=importance_activation,  # do not change
        importance_dropout_rate=importance_dropout_rate,  # do not change
        importance_factor=importance_factor,
        importance_multiplier=importance_multiplier,
        sparsity_factor=sparsity_factor,
        concat_heads=concat_heads,
        final_units=final_units,
        final_dropout_rate=final_dropout_rate,
        final_activation=final_activation,
        final_pooling=final_pooling,
        regression_limits=regression_limits,
        regression_reference=regression_reference,
        return_importances=return_importances
    )([n, ed, disjoint_indices, gs, batch_id_node, count_nodes])

    # Output embedding choice
    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding, output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type, cast_disjoint_kwargs=cast_disjoint_kwargs,
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__

    return model


make_model.__doc__ = make_model.__doc__ % (template_cast_list_input_docs, template_cast_output_docs)
