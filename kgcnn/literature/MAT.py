import tensorflow as tf
from typing import Union
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.layers.casting import ChangeTensorType, CastEdgeIndicesToDenseAdjacency
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of CMPNN in `tf.keras` from paper:
# Communicative Representation Learning on Attributed Molecular Graphs
# Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu and Yuedong Yang
# https://www.ijcai.org/proceedings/2020/0392.pdf

model_default = {
    "name": "MAT",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "max_atoms": None,
    "verbose": 10,
    "depth": 5,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [64, 64, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               max_atoms: int = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `CMPNN <https://www.ijcai.org/proceedings/2020/0392.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.CMPNN.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, edge_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "MAT".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level for print information.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_input = ks.layers.Input(**inputs[2])
    edge_index_input = ks.layers.Input(**inputs[3])

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[2]['shape']) < 2)(edge_input)
    edi = edge_index_input

    # Cast to dense Tensor with padding for MAT.
    n, n_mask = ChangeTensorType(output_tensor_type="padded")(n)
    xyz, xyz_mask = ChangeTensorType(output_tensor_type="padded")(xyz_input)
    a, a_mask = CastEdgeIndicesToDenseAdjacency(n_max=max_atoms)([ed, edi])

    # Model Loop
    for i in range(depth):
        # Model
        pass

    out = n
    if output_embedding == 'graph':
        pass
    elif output_embedding == 'node':
        pass
    else:
        raise ValueError("Unsupported graph embedding for mode `MAT`")

    model = ks.models.Model(
        inputs=[node_input, xyz_input, edge_input, edge_index_input],
        outputs=out,
        name=name
    )
    return model

# test = make_model()