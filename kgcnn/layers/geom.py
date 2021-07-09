import numpy as np
import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.polynom import spherical_bessel_jn_zeros, spherical_bessel_jn_normalization_prefactor, \
    tf_spherical_bessel_jn, tf_spherical_harmonics_yl


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='NodeDistance')
class NodeDistance(GraphBaseLayer):
    """Compute geometric node distances similar to edges.

    A distance based edge is defined by edge or bond index in index list of shape (batch, [N], 2) with last dimension
    of ingoing and outgoing.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(NodeDistance, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(NodeDistance, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [position, edge_index]

                - position (tf.RaggedTensor): Node positions of shape (batch, [N], 3)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Gathered node distances as edges that match the number of indices of shape (batch, [M], 1)
        """
        dyn_inputs = inputs
        # We cast to values here
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge_index, edge_part = dyn_inputs[1].values, dyn_inputs[1].row_lengths()

        indexlist = partition_row_indexing(edge_index, node_part, edge_part, partition_type_target="row_splits",
                                           partition_type_index="row_length", to_indexing='batch',
                                           from_indexing=self.node_indexing)
        # For ragged tensor we can now also try:
        # out = tf.gather(nod, tensor_index[:, :, 0], batch_dims=1)
        xi = tf.gather(node, indexlist[:, 0], axis=0)
        xj = tf.gather(node, indexlist[:, 1], axis=0)

        out = tf.expand_dims(tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.math.square(xi - xj), axis=-1))), axis=-1)

        out = tf.RaggedTensor.from_row_lengths(out, edge_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update config."""
        config = super(NodeDistance, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ScalarProduct')
class ScalarProduct(GraphBaseLayer):
    """Compute geometric scalar product for edges.

    A distance based edge or node coordinates are defined by (batch, [N], ..., D) with last dimension D.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(ScalarProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(ScalarProduct, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [vec1, vec1]

                - vec1 (tf.RaggedTensor): Positions of shape (batch, [N], ..., D)
                - vec1 (tf.RaggedTensor): Positions of shape (batch, [N], ..., D)

        Returns:
            tf.RaggedTensor: Scalar product of shape (batch, [N], ...)
        """
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]) and all([x.ragged_rank == 1 for x in inputs]):
            v1 = inputs[0].values
            v2 = inputs[1].values
            out = tf.reduce_sum(v1*v2, axis=-1)
            out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
        else:
            out = tf.reduce_sum(inputs[0]*inputs[1], axis=-1)
        return out

    def get_config(self):
        """Update config."""
        config = super(ScalarProduct, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EuclideanNorm')
class EuclideanNorm(GraphBaseLayer):
    """Compute geometric norm for edges or nodes.

    A distance based edge or node coordinates are defined by (batch, [N], ..., D) with last dimension D.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(EuclideanNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(EuclideanNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): coord

                - coord (tf.RaggedTensor): Positions of shape (batch, [N], ..., D)

        Returns:
            tf.RaggedTensor: Scalar product of shape (batch, [N], ...)
        """
        # axis = [i for i in range(len(inputs.shape))][self.axis]
        if isinstance(inputs, tf.RaggedTensor) and inputs.ragged_rank == 1:
            v = inputs.values
            out = tf.reduce_sum(tf.square(v), axis=-1)
            out = tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)
        else:  # Try general tensor approach
            out = tf.reduce_sum(tf.square(inputs), axis=-1)
        return out

    def get_config(self):
        """Update config."""
        config = super(EuclideanNorm, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EdgeDirectionNormalized')
class EdgeDirectionNormalized(GraphBaseLayer):
    r"""Compute the normalized geometric edge direction similar to edges.
    Will return :math:`\frac{r_{ij}}{||{r_ij}||} `.

    A distance based edge is defined by edge or bond index in index list of shape (batch, [N], 2) with last dimension
    of ingoing and outgoing.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(EdgeDirectionNormalized, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(EdgeDirectionNormalized, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [position, edge_index]

                - position (tf.RaggedTensor): Node positions of shape (batch, [N], 3)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Gathered node distances as edges that match the number of indices of shape (batch, [M], 3)
        """
        assert all([isinstance(x, tf.RaggedTensor) for x in inputs]) and all([x.ragged_rank == 1 for x in inputs])
        dyn_inputs = inputs
        # We cast to values here
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge_index, edge_part = dyn_inputs[1].values, dyn_inputs[1].row_lengths()

        indexlist = partition_row_indexing(edge_index, node_part, edge_part, partition_type_target="row_splits",
                                           partition_type_index="row_length", to_indexing='batch',
                                           from_indexing=self.node_indexing)
        # For ragged tensor we can now also try:
        # out = tf.gather(nod, tensor_index[:, :, 0], batch_dims=1)
        xi = tf.gather(node, indexlist[:, 0], axis=0)
        xj = tf.gather(node, indexlist[:, 1], axis=0)

        xij = xi - xj
        out = tf.expand_dims(tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.math.square(xij), axis=-1))), axis=-1)
        out = xij/out
        out = tf.RaggedTensor.from_row_lengths(out, edge_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update config."""
        config = super(EdgeDirectionNormalized, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='NodeAngle')
class NodeAngle(GraphBaseLayer):
    """Compute geometric node angles.

    The geometric angle is computed between i<-j,j<-k for index tuple (i,j,k) in (batch, None, 3) last dimension.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(NodeAngle, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(NodeAngle, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [position, node_index]

                - position (tf.RaggedTensor): Node positions of shape (batch, [N], 3)
                - node_index (tf.RaggedTensor): Node indices of shape (batch, [M], 3) referring to nodes

        Returns:
            tf.RaggedTensor: Gathered node angles between edges that match the indices. Shape is (batch, [M], 1)
        """
        dyn_inputs = inputs
        # We cast to values here
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge_index, edge_part = dyn_inputs[1].values, dyn_inputs[1].row_lengths()

        indexlist = partition_row_indexing(edge_index, node_part, edge_part, partition_type_target="row_splits",
                                           partition_type_index="row_length", to_indexing='batch',
                                           from_indexing=self.node_indexing)
        # For ragged tensor we can now also try:
        # out = tf.gather(nod, tensor_index[:, :, 0], batch_dims=1)
        xi = tf.gather(node, indexlist[:, 0], axis=0)
        xj = tf.gather(node, indexlist[:, 1], axis=0)
        xk = tf.gather(node, indexlist[:, 2], axis=0)
        v1 = xi - xj
        v2 = xj - xk
        x = tf.reduce_sum(v1 * v2, axis=-1)
        y = tf.linalg.cross(v1, v2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        angle = tf.expand_dims(angle, axis=-1)

        out = tf.RaggedTensor.from_row_lengths(angle, edge_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update config."""
        config = super(NodeAngle, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EdgeAngle')
class EdgeAngle(GraphBaseLayer):
    """Compute geometric edge angles.

    The geometric angle is computed between edge tuple of index (i,j), where i,j refer to two edges.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(EdgeAngle, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(EdgeAngle, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [position, edge_index, angle_index]

                - position (tf.RaggedTensor): Node positions of shape (batch, [N], 3)
                - edge_index (tf.RaggedTensor): Edge indices of shape (batch, [M], 2) referring to nodes.
                - angle_index (tf.RaggedTensor): Angle indices of shape (batch, [K], 2) referring to edges.

        Returns:
            tf.RaggedTensor: Gathered edge angles between edges that match the indices. Shape is (batch, [K], 1)
        """
        dyn_inputs = inputs
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge_index, edge_part = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        angle_index, angle_part = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

        indexlist = partition_row_indexing(edge_index, node_part, edge_part, partition_type_target="row_splits",
                                           partition_type_index="row_length", to_indexing='batch',
                                           from_indexing=self.node_indexing)

        indexlist2 = partition_row_indexing(angle_index, edge_part, angle_part, partition_type_target="row_splits",
                                            partition_type_index="row_length", to_indexing='batch',
                                            from_indexing=self.node_indexing)

        # For ragged tensor we can now also try:
        # out = tf.gather(nod, tensor_index[:, :, 0], batch_dims=1)
        xi = tf.gather(node, indexlist[:, 0], axis=0)
        xj = tf.gather(node, indexlist[:, 1], axis=0)
        vs = xi - xj
        v1 = tf.gather(vs, indexlist2[:, 0], axis=0)
        v2 = tf.gather(vs, indexlist2[:, 1], axis=0)
        x = tf.reduce_sum(v1 * v2, axis=-1)
        y = tf.linalg.cross(v1, v2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        angle = tf.expand_dims(angle, axis=-1)

        out = tf.RaggedTensor.from_row_lengths(angle, angle_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update config."""
        config = super(EdgeAngle, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='BesselBasisLayer')
class BesselBasisLayer(GraphBaseLayer):
    r"""
    Expand a distance into a Bessel Basis with :math:`l=m=0`, according to Klicpera et al. 2020

    Args:
        num_radial (int): Number of radial radial basis functions
        cutoff (float): Cutoff distance c
        envelope_exponent (int): Degree of the envelope to smoothen at cutoff. Default is 5.
    """

    def __init__(self, num_radial,
                 cutoff,
                 envelope_exponent=5,
                 **kwargs):
        super(BesselBasisLayer, self).__init__(**kwargs)
        # Layer variables
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope_exponent = envelope_exponent

        # Initialize frequencies at canonical positions
        def freq_init(shape, dtype):
            return tf.constant(np.pi * np.arange(1, shape + 1, dtype=np.float32), dtype=dtype)

        self.frequencies = self.add_weight(name="frequencies", shape=self.num_radial,
                                           dtype=tf.float32, initializer=freq_init, trainable=True)

    @tf.function
    def envelope(self, inputs):
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / inputs + a * inputs ** (p - 1) + b * inputs ** p + c * inputs ** (p + 1)
        return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape (batch, [K], 1)

        Returns:
            tf.RaggedTensor: Expanded distance. Shape is (batch, [K], #Radial)
        """
        dyn_inputs = [inputs]
        # We cast to values here
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits

        d_scaled = node * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        out = d_cutoff * tf.sin(self.frequencies * d_scaled)
        out = tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update config."""
        config = super(BesselBasisLayer, self).get_config()
        config.update({"num_radial": self.num_radial, "cutoff": self.cutoff,
                       "envelope_exponent": self.envelope_exponent})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SphericalBasisLayer')
class SphericalBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Bessel Basis with :math:`l=m=0`, according to Klicpera et al. 2020

    Args:
        num_spherical (int): Number of spherical basis functions
        num_radial (int): Number of radial basis functions
        cutoff (float): Cutoff distance c
        envelope_exponent (int): Degree of the envelope to smoothen at cutoff. Default is 5.
    """

    def __init__(self, num_spherical,
                 num_radial,
                 cutoff,
                 envelope_exponent=5,
                 **kwargs):
        super(SphericalBasisLayer, self).__init__(**kwargs)

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope_exponent = envelope_exponent

        # retrieve formulas
        self.bessel_n_zeros = spherical_bessel_jn_zeros(num_spherical, num_radial)
        self.bessel_norm = spherical_bessel_jn_normalization_prefactor(num_spherical, num_radial)

    @tf.function
    def envelope(self, inputs):
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1 / inputs + a * inputs ** (p - 1) + b * inputs ** p + c * inputs ** (p + 1)
        return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [distance, angles, angle_index]

                - distance (tf.RaggedTensor): Edge distance of shape (batch, [M], 1)
                - angles (tf.RaggedTensor): Angle list of shape (batch, [K], 1)
                - angle_index (tf.RaggedTensor): Angle indices referring to edges of shape (batch, [K], 2)

        Returns:
            tf.RaggedTensor: Expanded angle/distance basis. Shape is (batch, [K], #Radial * #Spherical)
        """
        dyn_inputs = inputs
        edge, edge_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        angles, angle_part = dyn_inputs[1].values, dyn_inputs[1].row_splits
        angle_index, angle_index_part = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

        indexlist = partition_row_indexing(angle_index, edge_part, angle_index_part,
                                           partition_type_target="row_splits", partition_type_index="row_length",
                                           to_indexing='batch', from_indexing=self.node_indexing)

        d = edge
        id_expand_kj = indexlist

        d_scaled = d[:, 0] * self.inv_cutoff
        rbf = []
        for n in range(self.num_spherical):
            for k in range(self.num_radial):
                rbf += [self.bessel_norm[n, k] * tf_spherical_bessel_jn(d_scaled * self.bessel_n_zeros[n][k], n)]
        rbf = tf.stack(rbf, axis=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf
        rbf_env = tf.gather(rbf_env, id_expand_kj[:, 1])

        cbf = [tf_spherical_harmonics_yl(angles[:, 0], n) for n in range(self.num_spherical)]
        cbf = tf.stack(cbf, axis=1)
        cbf = tf.repeat(cbf, self.num_radial, axis=1)
        out = rbf_env * cbf

        out = tf.RaggedTensor.from_row_splits(out, angle_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update config."""
        config = super(SphericalBasisLayer, self).get_config()
        config.update({"num_radial": self.num_radial, "cutoff": self.cutoff,
                       "envelope_exponent": self.envelope_exponent, "num_spherical": self.num_spherical})
        return config
