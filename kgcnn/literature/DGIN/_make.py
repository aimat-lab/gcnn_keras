import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState
from kgcnn.layers.modules import Dense, LazyConcatenate, Activation, LazyAdd, Dropout, \
    OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.aggr import AggregateLocalEdges
from ...layers.pooling import PoolingNodes
from ._dgin_conv import DMPNNPPoolingEdgesDirected, GIN_D
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023.03.22"

# Implementation of DGIN in `tf.keras` from paper:
# Analyzing Learned Molecular Representations for Property Prediction
# by Oliver Wieder, Mélaine Kuenemann, Marcus Wieder, Thomas Seidel,
# Christophe Meyer, Sharon D Bryant and Thierry Langer
# https://pubmed.ncbi.nlm.nih.gov/34684766/

model_default = {
    "name": "DGIN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "gin_mlp": {"units": [64,64], "use_bias": True, "activation": ["relu","linear"],
                "use_normalization": True, "normalization_technique": "graph_batch"},
    "gin_args": {},
    "last_mlp": {"use_bias": [True, True], "units": [64, 64],
                 "activation": ["relu", "relu"]},
    "pooling_args": {"pooling_method": "sum"},
    "use_graph_state": False,
    "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
    "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
    "edge_activation": {"activation": "relu"},
    "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
    "verbose": 10,
    "depthDMPNN": 4,"depthGIN": 4,
    "dropoutDMPNN": {"rate": 0.15}, "dropoutGIN":  {"rate": 0.15},
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "linear"}
}

@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               pooling_args: dict = None,
               edge_initialize: dict = None,
               edge_dense: dict = None,
               edge_activation: dict = None,
               node_dense: dict = None,
               dropoutDMPNN: dict = None,
               dropoutGIN: dict = None,
               depthDMPNN: int = None,
               depthGIN: int = None,
               gin_args: dict = None,
               gin_mlp: dict = None,
               last_mlp: dict = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `DGIN <https://pubmed.ncbi.nlm.nih.gov/34684766/>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DGIN.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_pairs]` or
        `[node_attributes, edge_attributes, edge_indices, edge_pairs, state_attributes]` if `use_graph_state=True`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_pairs (tf.RaggedTensor): Pair mappings for reverse edge for each edge `(batch, None, 1)`.
            - state_attributes (tf.Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "DGIN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            :obj:`AggregateLocalEdges` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge embedding.
        edge_activation (dict): Edge Activation after skip connection.
        node_dense (dict): Dense kwargs for node embedding layer.
        depthDMPNN (int): Number of graph embedding units or depth of the DMPNN subnetwork.
        depthGIN (int): Number of graph embedding units or depth of the GIN subnetwork.
        dropoutDMPNN (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        dropoutGIN (float): dropout rate.
        verbose (int): Level for print information.
        use_graph_state (bool): Whether to use graph state information. Default is False.
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
    graph_state_input = ks.layers.Input(**inputs[4]) if use_graph_state else None

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(
        **input_embedding["node"],
        use_embedding=len(inputs[0]["shape"]) < 2)(node_input)
    ed = OptionalInputEmbedding(
        **input_embedding["edge"],
        use_embedding=len(inputs[1]["shape"]) < 2)(edge_input)
    graph_state = OptionalInputEmbedding(
        **input_embedding["graph"],
        use_embedding=len(inputs[4]["shape"]) < 1)(graph_state_input) if use_graph_state else None
    edi = edge_index_input
    ed_pairs = edge_pair_input

    # Make first edge hidden h0 step 1
    h_n0 = GatherNodesOutgoing()([n, edi])
    h0 = LazyConcatenate(axis=-1)([h_n0, ed])
    h0 = Dense(**edge_initialize)(h0) 
    h0 = Activation(**edge_activation)(h0) # relu equation 1

    # One Dense layer for all message steps this is not the case in DGIN they are independents!
    edge_dense_all = Dense(**edge_dense) # see equation 3 comments

    # Model Loop steps 2 & 3
    h = h0
    for i in range(depthDMPNN):
        # equation 2
        m_vw = DMPNNPPoolingEdgesDirected()([n, h, edi, ed_pairs]) # ed_pairs for Directed Pooling!
        # equation 3
        h = edge_dense_all(m_vw)  # do one per layer ...
        #h = Dense(**edge_dense)(m_vw)
        h = LazyAdd()([h, h0])
        # remark : dropout before Activation in DGIN code
        h = Activation(**edge_activation)(h)
        if dropoutDMPNN is not None:
            h = Dropout(**dropoutDMPNN)(h)

    # equation 4 & 5
    m_v = AggregateLocalEdges(**pooling_args)([n, h, edi])
    m_v = LazyConcatenate(axis=-1)([n, m_v]) # 
    # equation 5b: hv = Dense(**node_dense)(mv) removed based on the paper

    # GIN_D part (this projection is normaly not done in DGIN but we need to get the correct "dim")
    # not sure of this code :
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    if n_units != m_v.shape[-1]:
        h_v = Dense(n_units, use_bias=True, activation='linear')(m_v)
    h_v_0 = h_v

    list_embeddings = [h_v_0] # empty in the paper
    for i in range(depthGIN):
        # not sure of the mv, mv ... here but why not ;-)
        h_v = GIN_D(**gin_args)([h_v, edi, h_v_0]) # equation 6 & 7a  mv is new the new nodes values and we do pooling on ed via edi
        h_v = GraphMLP(**gin_mlp)(h_v)  # equation 7b
        list_embeddings.append(h_v)
    
    # Output embedding choice look like it takes only the last h_v in the paper not all ???
    if output_embedding == 'graph': 
        out = [PoolingNodes()(x) for x in list_embeddings]  # will return tensor equation 8
        out = [MLP(**last_mlp)(x) for x in out] # MLP apply per depthGIN
        if dropoutGIN is not None:
            out = [ks.layers.Dropout(**dropoutGIN)(x) for x in out]
        out = ks.layers.Add()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if use_graph_state:
            graph_state_node = GatherState()([graph_state, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `DMPNN`.")

    if use_graph_state:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, edge_pair_input, graph_state_input],
            outputs=out, name=name)
    else:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, edge_pair_input],
            outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    return model
