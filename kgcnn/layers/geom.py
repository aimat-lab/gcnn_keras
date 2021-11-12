import numpy as np
import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesSelection, GatherNodesOutgoing, GatherState
from kgcnn.ops.partition import partition_row_indexing
from kgcnn.layers.keras import Subtract, Multiply
from kgcnn.ops.polynom import spherical_bessel_jn_zeros, spherical_bessel_jn_normalization_prefactor, \
    tf_spherical_bessel_jn, tf_spherical_harmonics_yl
from kgcnn.ops.axis import get_positive_axis


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='NodePosition')
class NodePosition(GraphBaseLayer):
    """Get node position for edges. Directly calls `GatherNodesSelection` with provided index tensor.
    Returns separate node position tensor for each of the indices. Index selection must be provided
    in the constructor. Defaults to first two indices of an edge.

    A distance based edge is defined by two bond indices of the index list of shape (batch, [M], 2) with last dimension
    of incoming and outgoing node (message passing framework).
    """

    def __init__(self, selection_index: list = None, **kwargs):
        """Initialize layer instance of `NodePosition`.

        Args:
            selection_index (list): List of positions (last dimension of the index tensor) to return node coordinates.
                Default is [0, 1].
        """
        super(NodePosition, self).__init__(**kwargs)
        if selection_index is None:
            selection_index = [0, 1]
        self.selection_index = selection_index
        self.layer_gather = GatherNodesSelection(self.selection_index, **self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(NodePosition, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass of `NodePosition`.

        Args:
            inputs (list): [position, edge_index]

                - position (tf.RaggedTensor): Node positions of shape (batch, [N], 3).
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2).

        Returns:
            list: List of node positions (ragged) tensors for each of the `selection_index`. Position tensors have
                shape (batch, [M], 3).
        """
        return self.layer_gather(inputs, **kwargs)

    def get_config(self):
        """Update config for `NodePosition`."""
        config = super(NodePosition, self).get_config()
        config.update({"selection_index": self.selection_index})
        return config


class ShiftPeriodicLattice(GraphBaseLayer):
    """Node position periodic.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(ShiftPeriodicLattice, self).__init__(**kwargs)
        self.layer_state = GatherState()

    def build(self, input_shape):
        """Build layer."""
        super(ShiftPeriodicLattice, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [position, edge_index]

                - position (tf.RaggedTensor): Positions of shape (batch, [M], 3)
                - edge_image (tf.RaggedTensor): Position in which image to shift of shape (batch, [M], 3)
                - lattice (tf.tensor): Lattice vector matrix of shape (batch, 3, 3)

        Returns:
            tf.RaggedTensor: Gathered node position number of indices of shape (batch, [M], 1)
        """
        lattice_rep = self.layer_state([inputs[2], inputs[1]])  # Should be (batch, None, 3, 3)
        if all([isinstance(x, tf.RaggedTensor) for x in inputs[:2]]):  # Possibly faster
            if all([x.ragged_rank == 1 for x in inputs[:2]]):
                x = inputs[0]
                xj = x.values
                xj = xj + tf.reduce_sum(tf.cast(lattice_rep.values, dtype=xj.dtype) * tf.expand_dims(
                    tf.cast(inputs[1].values, dtype=xj.dtype), axis=-1), axis=1)
                return tf.RaggedTensor.from_row_splits(xj, inputs[1].row_splits, validate=self.ragged_validate)
        else:
            raise NotImplementedError("ERROR:kgcnn: Not implemented for arbitrary ragged_rank.")


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EuclideanNorm')
class EuclideanNorm(GraphBaseLayer):
    """Compute euclidean norm for edges or nodes vectors. This amounts for a  certain dimension to

    :math:`||x||_2 = \sqrt{\sum_i x_i^2}`

    Vector based edge or node coordinates are defined by (batch, [N], ..., D) with last dimension D.
    You can choose to collapse or keep the dimension.
    """

    def __init__(self, axis: int = -1, keepdims: bool = False, invert_norm: bool = False, **kwargs):
        """Initialize layer."""
        super(EuclideanNorm, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
        self.invert_norm = invert_norm

    def build(self, input_shape):
        """Build layer."""
        super(EuclideanNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass for `EuclideanNorm`.

        Args:
            inputs (tf.RaggedTensor): Positions of shape (batch, [N], ..., D, ...)

        Returns:
            tf.RaggedTensor: Euclidean norm computed for specific axis of shape (batch, [N], ...)
        """
        # Possibly faster
        if isinstance(inputs, tf.RaggedTensor):
            axis = get_positive_axis(self.axis, inputs.shape.rank)
            if inputs.ragged_rank == 1 and axis > 1:
                out = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(inputs.values), axis=axis - 1, keepdims=self.keepdims)))
                if self.invert_norm:
                    out = tf.math.divide_no_nan(tf.constant(1.0, dtype=out.dtype), out)
                return tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)
        # Default
        out = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(inputs), axis=self.axis, keepdims=self.keepdims)))
        if self.invert_norm:
            out = tf.math.divide_no_nan(tf.constant(1.0, dtype=out.dtype), out)
        return out

    def get_config(self):
        """Update config."""
        config = super(EuclideanNorm, self).get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims, "invert_norm": self.invert_norm})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ScalarProduct')
class ScalarProduct(GraphBaseLayer):
    """Compute geometric scalar product for edge or node coordinates.

    A distance based edge or node coordinates are defined by (batch, [N], ..., D) with last dimension D.
    """

    def __init__(self, axis=-1, **kwargs):
        """Initialize layer."""
        super(ScalarProduct, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        """Build layer."""
        super(ScalarProduct, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [vec1, vec1]

                - vec1 (tf.RaggedTensor): Positions of shape (batch, [N], ..., D, ...)
                - vec1 (tf.RaggedTensor): Positions of shape (batch, [N], ..., D, ...)

        Returns:
            tf.RaggedTensor: Scalar product of shape (batch, [N], ...)
        """
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):  # Possibly faster
            axis = get_positive_axis(self.axis, inputs[0].shape.rank)
            axis2 = get_positive_axis(self.axis, inputs[1].shape.rank)
            assert axis2 == axis
            if all([x.ragged_rank == 1 for x in inputs]) and axis > 1:
                v1, v2 = inputs[0].values, inputs[1].values
                out = tf.reduce_sum(v1 * v2, axis=axis - 1)
                return tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
        # Default
        out = tf.reduce_sum(inputs[0] * inputs[1], axis=self.axis)
        return out

    def get_config(self):
        """Update config."""
        config = super(ScalarProduct, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='NodeDistanceEuclidean')
class NodeDistanceEuclidean(GraphBaseLayer):
    r"""Compute euclidean distance between two node coordinate tensors. Let the :math:`\vec{x}_1` and :math:`\vec{x}_2`
    be two nodes, then the output is given by :math:`|| \vec{x}_1 - \vec{x}_2 ||_2`. Calls :obj:`EuclideanNorm` on
    the difference of the inputs. The number of node positions must not be connected to the number of nodes, but can
    also match edges.
    """

    def __init__(self, **kwargs):
        """Initialize layer instance of `NodeDistanceEuclidean`."""
        super(NodeDistanceEuclidean, self).__init__(**kwargs)
        self.layer_subtract = Subtract()
        self.layer_euclidean_norm = EuclideanNorm(axis=2, keepdims=True)

    def build(self, input_shape):
        """Build layer."""
        super(NodeDistanceEuclidean, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [position_start, position_stop]

                - position_start (tf.RaggedTensor): Node positions of shape (batch, [M], 3)
                - position_stop (tf.RaggedTensor): Node positions of shape (batch, [M], 3)

        Returns:
            tf.RaggedTensor: Distances as edges that match the number of indices of shape (batch, [M], 1)
        """
        diff = self.layer_subtract(inputs)
        return self.layer_euclidean_norm(diff)



@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EdgeDirectionNormalized')
class EdgeDirectionNormalized(GraphBaseLayer):
    r"""Compute the normalized geometric direction between two point coordinates for e.g. a geometric edge.

    :math:`\frac{\vec{r}_{ij}}{||{r_ij}||} = \frac{\vec{r}_{i} - \vec{r}_{j}}{||\vec{r}_{i} - \vec{r}_{j}||}`.

    Note that the difference is defined here as :math:`\vec{r}_{i} - \vec{r}_{j}`.
    As the first index defines the incoming edge.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(EdgeDirectionNormalized, self).__init__(**kwargs)
        self.layer_subtract = Subtract()
        self.layer_euclidean_norm = EuclideanNorm(axis=2, keepdims=True, invert_norm=True)
        self.layer_multiply = Multiply()

    def build(self, input_shape):
        super(EdgeDirectionNormalized, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [position_1, position_2]

                - position_1 (tf.RaggedTensor): Stop node positions of shape (batch, [N], 3)
                - position_2 (tf.RaggedTensor): Start node positions of shape (batch, [N], 3)

        Returns:
            tf.RaggedTensor: Normalized vector distance of shape (batch, [M], 3).
        """
        diff = self.layer_subtract(inputs)
        norm = self.layer_euclidean_norm(diff)
        return self.layer_multiply([diff, norm])

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
        self.lay_gather = GatherNodesSelection([0, 1, 2], **self._kgcnn_info)

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
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):  # Possibly faster
            if all([x.ragged_rank == 1 for x in inputs]):
                rxi, rxj, rxk = self.lay_gather(inputs)
                xi, xj, xk = rxi.values, rxj.values, rxk.values
                v1 = xi - xj
                v2 = xj - xk
                x = tf.reduce_sum(v1 * v2, axis=-1)
                y = tf.linalg.cross(v1, v2)
                y = tf.norm(y, axis=-1)
                angle = tf.math.atan2(y, x)
                angle = tf.expand_dims(angle, axis=-1)
                return tf.RaggedTensor.from_row_splits(angle, rxi.row_splits, validate=self.ragged_validate)
        # Default
        xi, xj, xk = self.lay_gather(inputs)
        v1 = xi - xj
        v2 = xj - xk
        x = tf.reduce_sum(v1 * v2, axis=-1)
        y = tf.linalg.cross(v1, v2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        out = tf.expand_dims(angle, axis=-1)
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
        self.lay_gather_x = GatherNodesSelection([0, 1], **self._kgcnn_info)
        self.lay_gather_v = GatherNodesSelection([0, 1], **self._kgcnn_info)

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
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):  # Possibly faster
            if all([x.ragged_rank == 1 for x in inputs]):
                rxi, rxj = self.lay_gather_x([inputs[0], inputs[1]])
                vs = tf.RaggedTensor.from_row_splits(rxi.values - rxj.values, rxi.row_splits,
                                                     validate=self.ragged_validate)
                rv1, rv2 = self.lay_gather_v([vs, inputs[2]])
                v1, v2 = rv1.values, rv2.values
                x = tf.reduce_sum(v1 * v2, axis=-1)
                y = tf.linalg.cross(v1, v2)
                y = tf.norm(y, axis=-1)
                angle = tf.math.atan2(y, x)
                angle = tf.expand_dims(angle, axis=-1)
                return tf.RaggedTensor.from_row_splits(angle, rv1.row_splits, validate=self.ragged_validate)
        # Default
        xi, xj = self.lay_gather_x([inputs[0], inputs[1]])
        vs = xi - xj
        v1, v2 = self.lay_gather_v([vs, inputs[2]])
        x = tf.reduce_sum(v1 * v2, axis=-1)
        y = tf.linalg.cross(v1, v2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        angle = tf.expand_dims(angle, axis=-1)
        return angle

    def get_config(self):
        """Update config."""
        config = super(EdgeAngle, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GaussBasisLayer')
class GaussBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Gauss Basis with :math:`\sgima`, according to Schuett et al."""

    def __init__(self, bins=20, distance=4.0, sigma=0.4, offset=0.0,
                 **kwargs):
        super(GaussBasisLayer, self).__init__(**kwargs)
        # Layer variables
        self.bins = int(bins)
        self.distance = float(distance)
        self.offset = float(offset)
        self.sigma = float(sigma)
        self.gamma = 1 / sigma / sigma * (-1) / 2

        # Note: For arbitrary axis the code must be adapted.

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape (batch, [K], 1)

        Returns:
            tf.RaggedTensor: Expanded distance. Shape is (batch, [K], #bins)
        """
        gbs = tf.range(0, self.bins, 1, dtype=self.dtype) / float(self.bins) * self.distance
        # Possibly faster RaggedRank==1
        if isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                edge, ege_part = inputs.values, inputs.row_splits
                out = edge - self.offset
                out = tf.square(out - gbs) * self.gamma
                out = tf.exp(out)
                return tf.RaggedTensor.from_row_splits(out, ege_part, validate=self.ragged_validate)
        # Default
        out = inputs - self.offset
        out = tf.square(out - gbs) * self.gamma
        out = tf.exp(out)
        return out

    def get_config(self):
        """Update config."""
        config = super(GaussBasisLayer, self).get_config()
        config.update({"bins": self.bins, "distance": self.distance, "offset": self.offset, "sigma": self.sigma})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='BesselBasisLayer')
class BesselBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Bessel Basis with :math:`l=m=0`, according to Klicpera et al. 2020

    Args:
        num_radial (int): Number of radial radial basis functions
        cutoff (float): Cutoff distance c
        envelope_exponent (int): Degree of the envelope to smoothen at cutoff. Default is 5.
    """

    def __init__(self, num_radial,
                 cutoff,
                 envelope_exponent=5,
                 envelope_type="poly",
                 **kwargs):
        super(BesselBasisLayer, self).__init__(**kwargs)
        # Layer variables
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope_exponent = envelope_exponent
        self.envelope_type = str(envelope_type)

        if self.envelope_type not in ["poly"]:
            raise ValueError("Unknown envelope type %s in BesselBasisLayer" % self.envelope_type)

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
        # Possibly faster RaggedRank==1
        if isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                node, node_part = inputs.values, inputs.row_splits
                d_scaled = node * self.inv_cutoff
                d_cutoff = self.envelope(d_scaled)
                out = d_cutoff * tf.sin(self.frequencies * d_scaled)
                return tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)
        # Default
        d_scaled = inputs * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        out = d_cutoff * tf.sin(self.frequencies * d_scaled)
        return out

    def get_config(self):
        """Update config."""
        config = super(BesselBasisLayer, self).get_config()
        config.update({"num_radial": self.num_radial, "cutoff": self.cutoff,
                       "envelope_exponent": self.envelope_exponent, "envelope_type": self.envelope_type})
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

        self.layer_gather_out = GatherNodesOutgoing(**self._kgcnn_info)

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

        d = edge
        d_scaled = d[:, 0] * self.inv_cutoff
        rbf = []
        for n in range(self.num_spherical):
            for k in range(self.num_radial):
                rbf += [self.bessel_norm[n, k] * tf_spherical_bessel_jn(d_scaled * self.bessel_n_zeros[n][k], n)]
        rbf = tf.stack(rbf, axis=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf
        ragged_rbf_env = tf.RaggedTensor.from_row_splits(rbf_env, edge_part, validate=self.ragged_validate)
        rbf_env = self.layer_gather_out([ragged_rbf_env, inputs[2]]).values
        # rbf_env = tf.gather(rbf_env, id_expand_kj[:, 1])

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


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='CosCutOffEnvelope')
class CosCutOffEnvelope(GraphBaseLayer):
    r"""Calculate cos-cutoff envelope according to Behler et al. https://aip.scitation.org/doi/10.1063/1.3553717
    :math:`f_c(R_{ij}) = 0.5 [ \cos{\frac{\pi R_{ij}}{R_c}} + 1]`

    Args:
        cutoff (float): Cutoff distance :math:`R_c`.
    """

    def __init__(self,
                 cutoff,
                 **kwargs):
        super(CosCutOffEnvelope, self).__init__(**kwargs)
        self.cutoff = float(np.abs(cutoff)) if cutoff is not None else 1e8

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape (batch, [M], 1)

        Returns:
            tf.RaggedTensor: Cutoff envelope of shape (batch, [M], 1)
        """
        if isinstance(inputs, tf.RaggedTensor):   # Possibly faster for ragged_rank == 1
            if inputs.ragged_rank == 1:
                values = inputs.values
                fc = tf.clip_by_value(values, -self.cutoff, self.cutoff)
                fc = (tf.math.cos(fc * np.pi / self.cutoff) + 1) * 0.5
                # fc = tf.where(tf.abs(values) < self.cutoff, fc, tf.zeros_like(fc))
                return tf.RaggedTensor.from_row_splits(fc, inputs.row_splits, validate=self.ragged_validate)
        # Try tf.cos directly, works also for ragged
        fc = tf.clip_by_value(inputs, -self.cutoff, self.cutoff)
        fc = (tf.math.cos(fc * np.pi / self.cutoff) + 1) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    def get_config(self):
        """Update config."""
        config = super(CosCutOffEnvelope, self).get_config()
        config.update({"cutoff": self.cutoff})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='CosCutOff')
class CosCutOff(GraphBaseLayer):
    r"""Apply cos-cutoff according to Behler et al. https://aip.scitation.org/doi/10.1063/1.3553717
    :math:`f_c(R_{ij}) = 0.5 [ \cos{\frac{\pi R_{ij}}{R_c}} + 1]`

    Args:
        cutoff (float): Cutoff distance :math:`R_c`.
    """

    def __init__(self,
                 cutoff,
                 **kwargs):
        super(CosCutOff, self).__init__(**kwargs)
        self.cutoff = float(np.abs(cutoff)) if cutoff is not None else 1e8

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: distance
                - distance (tf.RaggedTensor): Edge distance of shape (batch, [M], D)

        Returns:
            tf.RaggedTensor: Cutoff applied to input of shape (batch, [M], D)
        """
        if isinstance(inputs, tf.RaggedTensor):   # Possibly faster for ragged_rank == 1
            if inputs.ragged_rank == 1:
                values = inputs.values
                fc = tf.clip_by_value(values, -self.cutoff, self.cutoff)
                fc = (tf.math.cos(fc * np.pi / self.cutoff) + 1) * 0.5
                # fc = tf.where(tf.abs(values) < self.cutoff, fc, tf.zeros_like(fc))
                out = fc * values
                return tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)
        # Default
        # Try tf.cos directly, works also for ragged
        fc = tf.clip_by_value(inputs, -self.cutoff, self.cutoff)
        fc = (tf.math.cos(fc * np.pi / self.cutoff) + 1) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        out = fc * inputs
        return out

    def get_config(self):
        """Update config."""
        config = super(CosCutOff, self).get_config()
        config.update({"cutoff": self.cutoff})
        return config