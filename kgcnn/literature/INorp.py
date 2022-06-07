import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.modules import LazyConcatenate, DenseEmbedding, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.pool.set2set import PoolingSet2Set
from kgcnn.utils.models import update_model_kwargs
ks = tf.keras

# Implementation of INorp in `tf.keras` from paper:
# 'Interaction Networks for Learning about Objects, Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://arxiv.org/abs/1612.00222
# https://github.com/higgsfield/interaction_network_pytorch

hyper_model_default = {'name': "INorp",
                       'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                                  {'shape': [], 'name': "graph_attributes", 'dtype': 'float32', 'ragged': False}],
                       'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                           "edge": {"input_dim": 5, "output_dim": 64},
                                           "graph": {"input_dim": 100, "output_dim": 64}},
                       'set2set_args': {"channels": 32, "T": 3, "pooling_method": "mean",
                                        "init_qstar": "mean"},
                       'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                       'edge_mlp_args': {"units": [100, 100, 100, 100, 50],
                                         "activation": ['relu', 'relu', 'relu', 'relu', "linear"]},
                       'pooling_args': {'pooling_method': "segment_mean"},
                       'depth': 3, 'use_set2set': False, 'verbose': 10,
                       'gather_args': {},
                       'output_embedding': 'graph', "output_to_tensor": True,
                       'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                      "activation": ['relu', 'relu', 'sigmoid']}
                       }


@update_model_kwargs(hyper_model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               gather_args: dict = None,
               edge_mlp_args: dict = None,
               node_mlp_args: dict = None,
               set2set_args: dict = None,
               pooling_args: dict = None,
               use_set2set: dict = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `INorp <https://arxiv.org/abs/1612.00222>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.INorp.hyper_model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, state_attributes]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - state_attributes (tf.Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gather_args (dict): Dictionary of layer arguments unpacked in :obj:`GatherNodes` layer.
        edge_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for edge updates.
        node_mlp_args (dict): Dictionary of layer arguments unpacked in :obj:`MLP` layer for node updates.
        set2set_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingSet2Set` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingLocalEdges`, :obj:`PoolingNodes`
            layer.
        use_set2set (bool): Whether to use :obj:`PoolingSet2Set` layer.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    env_input = ks.Input(**inputs[3])

    # embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    uenv = OptionalInputEmbedding(**input_embedding['graph'],
                                  use_embedding=len(inputs[3]['shape']) < 1)(env_input)
    edi = edge_index_input

    # Model
    ev = GatherState(**gather_args)([uenv, n])
    # n-Layer Step
    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing(**gather_args)([n, edi])
        eu2 = GatherNodesOutgoing(**gather_args)([n, edi])
        upd = LazyConcatenate(axis=-1)([eu2, eu1])
        eu = LazyConcatenate(axis=-1)([upd, ed])

        eu = GraphMLP(**edge_mlp_args)(eu)
        # Pool message
        nu = PoolingLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connection
        # Add environment
        nu = LazyConcatenate(axis=-1)(
            [n, nu, ev])  # LazyConcatenate node features with new edge updates
        n = GraphMLP(**node_mlp_args)(nu)

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            n = DenseEmbedding(set2set_args["channels"], activation="linear")(n)
            out = PoolingSet2Set(**set2set_args)(n)
        else:
            out = PoolingNodes(**pooling_args)(n)
        out = ks.layers.Flatten()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `INorp`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=out)
    return model
