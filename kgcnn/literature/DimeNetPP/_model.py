from keras.layers import Add, Subtract, Concatenate, Dense
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, BesselBasisLayer, EdgeAngle, ShiftPeriodicLattice, \
    SphericalBasisLayer
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from ._layers import DimNetInteractionPPBlock, EmbeddingDimeBlock, DimNetOutputBlock


def model_disjoint(
        inputs,
        use_node_embedding,
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
        use_output_mlp: bool = None,
        output_embedding: str = None,
        output_mlp: dict = None
):
    n, x, edi, adi, batch_id_node, count_nodes = inputs

    # Atom embedding
    if use_node_embedding:
        n = EmbeddingDimeBlock(**input_node_embedding)(n)

    # Calculate distances
    pos1, pos2 = NodePosition()([x, edi])
    d = NodeDistanceEuclidean()([pos1, pos2])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    v12 = Subtract()([pos1, pos2])
    a = EdgeAngle()([v12, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = Dense(emb_size, use_bias=True, activation=activation,
                    kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = Concatenate(axis=-1)([n_pairs, rbf_emb])
    x = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = Add()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])

        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        out = PoolingNodes(pooling_method="sum")([count_nodes, ps, batch_id_node])
    else:
        out = PoolingNodes(pooling_method="mean")([count_nodes, ps, batch_id_node])

    if use_output_mlp:
        out = MLP(**output_mlp)(out)

    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `DimeNetPP`. ")

    return out


def model_disjoint_crystal(
        inputs,
        use_node_embedding,
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
        use_output_mlp: bool = None,
        output_embedding: str = None,
        output_mlp: dict = None
    ):

    n, x, edi, adi, edge_image, lattice, batch_id_node, batch_id_edge, count_nodes = inputs

    # Atom embedding
    if use_node_embedding:
        n = EmbeddingDimeBlock(**input_node_embedding)(n)

    # Calculate distances
    pos1, pos2 = NodePosition()([x, edi])
    pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice, batch_id_edge])
    d = NodeDistanceEuclidean()([pos1, pos2])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    v12 = Subtract()([pos1, pos2])
    a = EdgeAngle()([v12, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = Dense(emb_size, use_bias=True, activation=activation,
                    kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = Concatenate(axis=-1)([n_pairs, rbf_emb])
    x = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = Add()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])
        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        out = PoolingNodes(pooling_method="sum")([count_nodes, ps, batch_id_node])
    else:
        out = PoolingNodes(pooling_method="mean")([count_nodes, ps, batch_id_node])

    if use_output_mlp:
        out = MLP(**output_mlp)(out)

    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `DimeNetPP`. ")

    return out
