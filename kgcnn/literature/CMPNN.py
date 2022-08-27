import tensorflow as tf
from typing import Union
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate, ActivationEmbedding, LazyAdd, DropoutEmbedding, \
    OptionalInputEmbedding, LazySubtract, LazyMultiply
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.conv.dmpnn_conv import DMPNNGatherEdgesPairs
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of CMPNN in `tf.keras` from paper:
# Communicative Representation Learning on Attributed Molecular Graphs
# Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu and Yuedong Yang
# https://www.ijcai.org/proceedings/2020/0392.pdf

model_default = {
    "name": "CMPNN",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
    'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
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
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [300, 100, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
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
               output_mlp: dict = None
               ):
    r"""Make `CMPNN <https://www.ijcai.org/proceedings/2020/0392.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.CMPNN.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_pairs]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_pairs (tf.RaggedTensor): Pair mappings for reverse edge for each edge `(batch, None, 1)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "CMPNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        node_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first node embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge communicate.
        node_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for node communicate.
        edge_activation (dict): Dictionary of layer arguments unpacked in :obj:`Activation` layer for edge communicate.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level for print information.
        dropout (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            :obj:`PoolingLocalEdges` layers.
        use_final_gru (bool): Whether to use GRU for final readout.
        pooling_gru (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodesGRU`.
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
    edge_pair_input = ks.layers.Input(**inputs[3])
    ed_pairs = edge_pair_input

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    h0 = DenseEmbedding(**node_initialize)(n)
    he0 = DenseEmbedding(**edge_initialize)(ed)

    # Model Loop
    h = h0
    he = he0
    for i in range(depth - 1):
        # Node message/update
        m_pool = PoolingLocalEdges(**pooling_kwargs)([h, he, edi])
        m_max = PoolingLocalEdges(pooling_method="segment_max")([h, he, edi])
        m = LazyMultiply()([m_pool, m_max])
        # In paper there is a potential COMMUNICATE() here but in reference code just add() operation.
        h = LazyAdd()([h, m])

        # Edge message/update
        h_out = GatherNodesOutgoing()([h, edi])
        e_rev = DMPNNGatherEdgesPairs()([he, ed_pairs])
        he = LazySubtract()([h_out, e_rev])
        he = DenseEmbedding(**edge_dense)(he)
        he = LazyAdd()([he, he0])
        he = ActivationEmbedding(**edge_activation)(he)
        if dropout:
            he = DropoutEmbedding(**dropout)(he)

    # Last step
    m_pool = PoolingLocalEdges(**pooling_kwargs)([h, he, edi])
    m_max = PoolingLocalEdges(pooling_method="segment_max")([h, he, edi])
    m = LazyMultiply()([m_pool, m_max])
    h_final = LazyConcatenate()([m, h, h0])
    h_final = DenseEmbedding(**node_dense)(h_final)

    n = h_final
    if output_embedding == 'graph':
        if use_final_gru:
            out = ks.layers.GRU(**pooling_gru)(n)
        else:
            out = PoolingNodes(**pooling_kwargs)(n)
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `CMPNN`")

    model = ks.models.Model(
        inputs=[node_input, edge_input, edge_index_input, edge_pair_input],
        outputs=out,
        name=name
    )
    return model
