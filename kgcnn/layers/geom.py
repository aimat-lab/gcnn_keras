import math
import numpy as np
import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesSelection, GatherState
from kgcnn.layers.modules import LazySubtract, LazyMultiply, LazyAdd
from kgcnn.ops.axis import get_positive_axis
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='NodePosition')
class NodePosition(GraphBaseLayer):
    r"""Get node position for directed edges via node indices.

    Directly calls :obj:`GatherNodesSelection` with provided index tensor.
    Returns separate node position tensor for each of the indices. Index selection must be provided
    in the constructor. Defaults to first two indices of an edge. This layer simply implements:

    .. code-block:: python

        GatherNodesSelection([0,1])([position, indices])

    A distance based edge is defined by two bond indices of the index list of shape `(batch, [M], 2)`
    with last dimension of incoming and outgoing node (message passing framework).
    Example usage:

    .. code-block:: python

        position = tf.ragged.constant([[[0.0, -1.0, 0.0],[1.0, 1.0, 0.0]]], ragged_rank=1)
        indices = tf.ragged.constant([[[0,1],[1,0]]], ragged_rank=1)
        x_in, x_out = NodePosition()([position, indices])
        print(x_in - x_out)
    """

    def __init__(self, selection_index: list = None, **kwargs):
        r"""Initialize layer instance of :obj:`NodePosition`.

        Args:
            selection_index (list): List of positions (last dimension of the index tensor) to return node coordinates.
                Default is [0, 1].
        """
        super(NodePosition, self).__init__(**kwargs)
        if selection_index is None:
            selection_index = [0, 1]
        self.selection_index = selection_index
        self.layer_gather = GatherNodesSelection(self.selection_index)

    def build(self, input_shape):
        """Build layer."""
        super(NodePosition, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass of :obj:`NodePosition`.

        Args:
            inputs (list): [position, edge_index]

                - position (tf.RaggedTensor): Node positions of shape `(batch, [N], 3)`.
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [M], 2)`.

        Returns:
            list: List of node positions (ragged) tensors for each of the :obj:`selection_index`. Position tensors have
            shape `(batch, [M], 3)`.
        """
        return self.layer_gather(inputs, **kwargs)

    def get_config(self):
        """Update config for `NodePosition`."""
        config = super(NodePosition, self).get_config()
        config.update({"selection_index": self.selection_index})
        return config


class ShiftPeriodicLattice(GraphBaseLayer):
    r"""Shift position tensor by multiples of the lattice constant of a periodic lattice in 3D.

    Let an atom have position :math:`\vec{x}_0` in the unit cell and be in a periodic lattice with lattice vectors
    :math:`\mathbf{a} = (\vec{a}_1, \vec{a}_2, \vec{a}_3)` and further be located in its image with indices
    :math:`\vec{n} = (n_1, n_2, n_3)`, then this layer is supposed to return:

    .. math::
        \vec{x} = \vec{x_0} + n_1\vec{a}_1 + n_2\vec{a}_2 + n_3\vec{a}_3 = \vec{x_0} + \vec{n} \mathbf{a}

    The layer expects ragged tensor input for :math:`\vec{x_0}` and :math:`\vec{n}` with multiple positions and their
    images but a single (tensor) lattice matrix per sample.

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
            inputs (list): [position, edge_image, lattice]

                - position (tf.RaggedTensor): Positions of shape `(batch, [M], 3)`
                - edge_image (tf.RaggedTensor): Position in which image to shift of shape `(batch, [M], 3)`
                - lattice (tf.tensor): Lattice vector matrix of shape `(batch, 3, 3)`

        Returns:
            tf.RaggedTensor: Gathered node position number of indices of shape `(batch, [M], 1)`
        """
        inputs_ragged = self.assert_ragged_input_rank(inputs[:2])
        lattice_rep = self.layer_state([inputs[2], inputs_ragged[1]], **kwargs)  # Should be (batch, None, 3, 3)
        x = inputs_ragged[0]
        ei = inputs_ragged[1]
        x_val = x.values
        # 1. Implementation: Manual multiplication.
        x_val = x_val + tf.reduce_sum(tf.cast(lattice_rep.values, dtype=x_val.dtype) * tf.expand_dims(
            tf.cast(ei.values, dtype=x_val.dtype), axis=-1), axis=1)
        # 2. Implementation: Matrix multiplication.
        # xv = xv + ks.batch_dot(tf.cast(ei.values, dtype=x_val.dtype), tf.cast(lattice_rep.values, dtype=x_val.dtype))
        return tf.RaggedTensor.from_row_splits(x_val, ei.row_splits, validate=self.ragged_validate)


@ks.utils.register_keras_serializable(package='kgcnn', name='EuclideanNorm')
class EuclideanNorm(GraphBaseLayer):
    r"""Compute euclidean norm for edge or node vectors.

    This amounts for a specific :obj:`axis` along which to sum the coordinates:

    .. math::
        ||\mathbf{x}||_2 = \sqrt{\sum_i x_i^2}

    Vector based edge or node coordinates are defined by `(batch, [N], ..., D)` with last dimension `D`.
    You can choose to collapse or keep this dimension with :obj:`keepdims` and to optionally invert the resulting norm
    with :obj:`invert_norm` layer arguments.
    """

    def __init__(self, axis: int = -1, keepdims: bool = False, invert_norm: bool = False, **kwargs):
        """Initialize layer.

        Args:
            axis (int): Axis of coordinates. Defaults to -1.
            keepdims (bool): Whether to keep the axis for sum. Defaults to False.
            invert_norm (bool): Whether to invert the results. Defaults to False.
        """
        super(EuclideanNorm, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
        self.invert_norm = invert_norm

    def build(self, input_shape):
        """Build layer."""
        super(EuclideanNorm, self).build(input_shape)

    @staticmethod
    def _compute_euclidean_norm(inputs, axis: int = -1, keepdims: bool = False, invert_norm: bool = False):
        """Function to compute euclidean norm for inputs.

        Args:
            inputs (tf.Tensor, tf.RaggedTensor): Tensor input to compute norm for.
            axis (int): Axis of coordinates. Defaults to -1.
            keepdims (bool): Whether to keep the axis for sum. Defaults to False.
            invert_norm (bool): Whether to invert the results. Defaults to False.

        Returns:
            tf.Tensor: Euclidean norm of inputs.
        """
        out = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=keepdims)))
        if invert_norm:
            out = tf.math.divide_no_nan(tf.constant(1, dtype=out.dtype), out)
        return out

    def call(self, inputs, **kwargs):
        r"""Forward pass for :obj:`EuclideanNorm`.

        Args:
            inputs (tf.RaggedTensor): Positions of shape `(batch, [N], ..., D, ...)`

        Returns:
            tf.RaggedTensor: Euclidean norm computed for specific axis of shape `(batch, [N], ...)`
        """
        # Possibly faster
        if isinstance(inputs, tf.RaggedTensor):
            axis = get_positive_axis(self.axis, inputs.shape.rank)
            if inputs.ragged_rank == 1 and axis > 1:
                out = self._compute_euclidean_norm(inputs.values, axis-1, self.keepdims, self.invert_norm)
                return tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)
        return self._compute_euclidean_norm(inputs, self.axis, self.keepdims, self.invert_norm)

    def get_config(self):
        """Update config."""
        config = super(EuclideanNorm, self).get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims, "invert_norm": self.invert_norm})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='ScalarProduct')
class ScalarProduct(GraphBaseLayer):
    r"""Compute geometric scalar product for edge or node coordinates.

    A distance based edge or node coordinates are defined by `(batch, [N], ..., D)` with last dimension D.
    The layer simply does for positions :

    .. math::
        <\vec{a}, \vec{b}> = \vec{a} \cdot \vec{b} = \sum_i a_i b_i

    Code example:

    .. code-block:: python

        position = tf.ragged.constant([[[0.0, -1.0, 0.0], [1.0, 1.0, 0.0]], [[2.0, 1.0, 0.0]]], ragged_rank=1)
        out = ScalarProduct()([position, position])
        print(out, out.shape)
    """

    def __init__(self, axis=-1, **kwargs):
        """Initialize layer."""
        super(ScalarProduct, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        """Build layer."""
        super(ScalarProduct, self).build(input_shape)

    @staticmethod
    def _scalar_product(inputs: list, axis: int, **kwargs):
        """Compute scalar product.

        Args:
            inputs (list): Tensor input.
            axis (int): Axis along which to sum.

        Returns:
            tf.Tensor: Scalr product of inputs.
        """
        return tf.reduce_sum(inputs[0] * inputs[1], axis=axis)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [vec1, vec2]

                - vec1 (tf.RaggedTensor): Positions of shape `(batch, [N], ..., D, ...)`
                - vec2 (tf.RaggedTensor): Positions of shape `(batch, [N], ..., D, ...)`

        Returns:
            tf.RaggedTensor: Scalar product of shape `(batch, [N], ...)`
        """
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):  # Possibly faster
            axis = get_positive_axis(self.axis, inputs[0].shape.rank)
            axis2 = get_positive_axis(self.axis, inputs[1].shape.rank)
            assert axis2 == axis
            if all([x.ragged_rank == 1 for x in inputs]) and axis > 1:
                v1, v2 = inputs[0].values, inputs[1].values
                out = self._scalar_product([v1, v2], axis=axis-1)
                return tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
        # Default
        out = self._scalar_product(inputs, axis=self.axis)
        return out

    def get_config(self):
        """Update config."""
        config = super(ScalarProduct, self).get_config()
        config.update({"axis": self.axis})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='NodeDistanceEuclidean')
class NodeDistanceEuclidean(GraphBaseLayer):
    r"""Compute euclidean distance between two node coordinate tensors.

    Let :math:`\vec{x}_1` and :math:`\vec{x}_2` be the position of two nodes, then the output is given by:

    .. math::
        || \vec{x}_1 - \vec{x}_2 ||_2.

    Calls :obj:`EuclideanNorm` on the difference of the inputs, which are position of nodes in space and for example
    the output of :obj:`NodePosition`.

    """

    def __init__(self, **kwargs):
        r"""Initialize layer instance of :obj:`NodeDistanceEuclidean`. """
        super(NodeDistanceEuclidean, self).__init__(**kwargs)
        self.layer_subtract = LazySubtract()
        self.layer_euclidean_norm = EuclideanNorm(axis=2, keepdims=True)

    def build(self, input_shape):
        """Build layer."""
        super(NodeDistanceEuclidean, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [position_start, position_stop]

                - position_start (tf.RaggedTensor): Node positions of shape `(batch, [M], 3)`
                - position_stop (tf.RaggedTensor): Node positions of shape `(batch, [M], 3)`

        Returns:
            tf.RaggedTensor: Distances as edges that match the number of indices of shape `(batch, [M], 1)`
        """
        diff = self.layer_subtract(inputs)
        return self.layer_euclidean_norm(diff)


@ks.utils.register_keras_serializable(package='kgcnn', name='EdgeDirectionNormalized')
class EdgeDirectionNormalized(GraphBaseLayer):
    r"""Compute the normalized geometric direction between two point coordinates for e.g. a geometric edge.

    Let two points have position :math:`\vec{r}_{i}` and :math:`\vec{r}_{j}` for an edge :math:`e_{ij}`, then
    the normalized distance is given by:

    .. math::
        \frac{\vec{r}_{ij}}{||r_{ij}||} = \frac{\vec{r}_{i} - \vec{r}_{j}}{||\vec{r}_{i} - \vec{r}_{j}||}.

    Note that the difference is defined here as :math:`\vec{r}_{i} - \vec{r}_{j}`.
    As the first index defines the incoming edge.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(EdgeDirectionNormalized, self).__init__(**kwargs)
        self.layer_subtract = LazySubtract()
        self.layer_euclidean_norm = EuclideanNorm(axis=2, keepdims=True, invert_norm=True)
        self.layer_multiply = LazyMultiply()

    def build(self, input_shape):
        """Build layer."""
        super(EdgeDirectionNormalized, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [position_1, position_2]

                - position_1 (tf.RaggedTensor): Stop node positions of shape `(batch, [N], 3)`
                - position_2 (tf.RaggedTensor): Start node positions of shape `(batch, [N], 3)`

        Returns:
            tf.RaggedTensor: Normalized vector distance of shape `(batch, [M], 3)`.
        """
        diff = self.layer_subtract(inputs)
        norm = self.layer_euclidean_norm(diff)
        return self.layer_multiply([diff, norm])

    def get_config(self):
        """Update config."""
        config = super(EdgeDirectionNormalized, self).get_config()
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='VectorAngle')
class VectorAngle(GraphBaseLayer):
    r"""Compute geometric angles between two vectors in euclidean space.

    The vectors :math:`\vec{v}_1` and :math:`\vec{v}_2` could be obtained from three points
    :math:`\vec{x}_i, \vec{x}_j, \vec{x}_k` spanning an angle from :math:`\vec{v}_1= \vec{x}_i - \vec{x}_j` and
    :math:`\vec{v}_2= \vec{x}_j - \vec{x}_k`.

    Those points can be defined with an index tuple `(i, j, k)` in a ragged tensor of shape `(batch, None, 3)` that
    mark vector directions of :math:`i\leftarrow j, j \leftarrow k`.

    .. note::
        However, this layer directly takes the vector :math:`\vec{v}_1` and :math:`\vec{v}_2` as input.

    The angle :math:`\theta` is computed via:

    .. math::
        \theta = \tan^{-1} \; \frac{\vec{v}_1 \cdot \vec{v}_2}{|| \vec{v}_1 \times \vec{v}_2 ||}

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(VectorAngle, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(VectorAngle, self).build(input_shape)

    @staticmethod
    def _compute_vector_angle(inputs: list):
        """Function to compute angles between two vectors v1 and v2.

        Args:
            inputs (list): List or tuple of two tensor v1, v2.

        Returns:
            tf.Tensor: Angle between inputs.
        """
        v1, v2 = inputs[0], inputs[1]
        x = tf.reduce_sum(v1 * v2, axis=-1)
        y = tf.linalg.cross(v1, v2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        out = tf.expand_dims(angle, axis=-1)
        return out

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [vector_1, vector_2]

                - vector_1 (tf.RaggedTensor): Node positions or vectors of shape `(batch, [M], 3)`
                - vector_2 (tf.RaggedTensor): Node positions or vectors of shape `(batch, [M], 3)`

        Returns:
            tf.RaggedTensor: Calculated Angle between vector 1 and 2 of shape `(batch, [M], 1)`.
        """
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):  # Possibly faster
            if all([x.ragged_rank == 1 for x in inputs]):
                angle = self._compute_vector_angle([inputs[0].values, inputs[1].values])
                return tf.RaggedTensor.from_row_splits(angle, inputs[0].row_splits, validate=self.ragged_validate)
        return self._compute_vector_angle(inputs)

    def get_config(self):
        """Update config."""
        config = super(VectorAngle, self).get_config()
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='EdgeAngle')
class EdgeAngle(GraphBaseLayer):
    r"""Compute geometric angles between two vectors that represent an edge of a graph.

    The vectors :math:`\vec{v}_1` and :math:`\vec{v}_2` span an angles as:

    .. math::
        \theta = \tan^{-1} \; \frac{\vec{v}_1 \cdot \vec{v}_2}{|| \vec{v}_1 \times \vec{v}_2 ||}

    The geometric angle is computed between edge tuples of index :math:`(i, j)`, where :math`:i, j` refer to two edges.
    The edge features are consequently a geometric vector (3D-space) for each edge.

    .. note::
        Here, the indices :math:`(i, j)` refer to edges and not to node positions!

    The layer uses :obj:`GatherEmbeddingSelection` and :obj:`VectorAngle` to compute angles.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(EdgeAngle, self).__init__(**kwargs)
        self.layer_gather_vectors = GatherNodesSelection([0, 1])
        self.layer_angle = VectorAngle()

    def build(self, input_shape):
        """Build layer."""
        super(EdgeAngle, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [vector, angle_index]

                - vector (tf.RaggedTensor): Node or Edge directions of shape `(batch, [N], 3)`
                - angle_index (tf.RaggedTensor): Angle indices of vector pairs of shape `(batch, [K], 2)`.

        Returns:
            tf.RaggedTensor: Edge angles between edges that match the indices. Shape is `(batch, [K], 1)`.
        """
        v1, v2 = self.layer_gather_vectors(inputs)
        return self.layer_angle([v1, v2])

    def get_config(self):
        """Update config."""
        config = super(EdgeAngle, self).get_config()
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='GaussBasisLayer')
class GaussBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Gaussian Basis, according to
    `Schuett et al. (2017) <https://arxiv.org/abs/1706.08566>`_.

    The distance :math:`d_{ij} = || \mathbf{r}_i - \mathbf{r}_j ||` is expanded in radial basis functions:

    .. math::
        e_k(\mathbf{r}_i - \mathbf{r}_j) = \exp{(- \gamma || d_{ij} - \mu_k ||^2 )}

    where :math:`\mu_k` represents centers located at originally :math:`0\le \mu_k \le 30  \mathring{A}`
    every :math:`0.1 \mathring{A}` with :math:`\gamma=10 \mathring{A}`

    For this layer the arguments refer directly to Gaussian of width :math:`\sigma` that connects to
    :math:`\gamma = \frac{1}{2\sigma^2}`. The Gaussian, or the :math:`\mu_k`, is placed equally
    between :obj:`offset` and :obj:`distance` and the spacing can be defined by the number of :obj:`bins` that is
    simply '(distance-offset)/bins'. The width is controlled by the layer argument :obj:`sigma`.

    """

    def __init__(self, bins: int = 20, distance: float = 4.0, sigma: float = 0.4, offset: float = 0.0,
                 **kwargs):
        r"""Initialize :obj:`GaussBasisLayer` layer.

        Args:
            bins (int): Number of bins for basis.
            distance (float): Maximum distance to for Gaussian.
            sigma (float): Width of Gaussian for bins.
            offset (float): Shift of zero position for basis.
        """
        super(GaussBasisLayer, self).__init__(**kwargs)
        # Layer variables
        self.bins = int(bins)
        self.distance = float(distance)
        self.offset = float(offset)
        self.sigma = float(sigma)
        self.gamma = 1 / sigma / sigma / 2

        # Note: For arbitrary axis the code must be adapted.

    @staticmethod
    def _compute_gauss_basis(inputs, offset, gamma, bins, distance):
        r"""Expand into gaussian basis.

        Args:
            inputs (tf.Tensor, tf.RaggedTensor): Tensor input with distance to expand into Gaussian basis.
            bins (int): Number of bins for basis.
            distance (float): Maximum distance to for Gaussian.
            gamma (float): Gamma pre-factor which is :math:`1/(2\sigma^2)` for Gaussian of width :math:`\sigma`.
            offset (float): Shift of zero position for basis.

        Returns:
            tf.Tensor: Distance tensor expanded in Gaussian.
        """
        gbs = tf.range(0, bins, 1, dtype=inputs.dtype) / float(bins) * distance
        out = inputs - offset
        out = tf.square(out - gbs) * (gamma * (-1.0))
        out = tf.exp(out)
        return out

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape `(batch, [K], 1)`

        Returns:
            tf.RaggedTensor: Expanded distance. Shape is `(batch, [K], bins)`.
        """
        # Possibly faster RaggedRank==1
        if isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                out = self._compute_gauss_basis(inputs.values, self.offset, self.gamma, self.bins, self.distance)
                return tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)
        return self._compute_gauss_basis(inputs, self.offset, self.gamma, self.bins, self.distance)

    def get_config(self):
        """Update config."""
        config = super(GaussBasisLayer, self).get_config()
        config.update({"bins": self.bins, "distance": self.distance, "offset": self.offset, "sigma": self.sigma})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='FourierBasisLayer')
class PositionEncodingBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Positional Encoding basis from `Transformer <https://arxiv.org/pdf/1706.03762.pdf>`_
    models, with :math:`\sin()` and :math:`\sin()` functions, adapted to distance-like positions.


    """

    def __init__(self,
                 dim_half: int = 8,
                 wave_length: float = 10,
                 include_frequencies: bool = False,
                 **kwargs):
        r"""Initialize :obj:`FourierBasisLayer` layer.

        Args:
            dim_half (int): Dimension of the half output embedding space. Final basis will be 2 * `dim_half`, or
                even 3 * `dim_half`, if `include_frequencies` is set to `True`. Defaults to 8.
            wave_length (float): Wavelength for positional sin and cos expansion. Defaults to 10.0.
            include_frequencies (bool): Whether to also include the frequencies. Default is False.

        """
        super(PositionEncodingBasisLayer, self).__init__(**kwargs)
        # Layer variables
        self.dim_half = dim_half
        self.wave_length = wave_length
        self.include_frequencies = include_frequencies
        # Note: For arbitrary axis the code must be adapted.

    @staticmethod
    def _compute_fourier_encoding(inputs,
                                  dim_half: int = 4, wave_length: float = 10.0, include_frequencies: bool = False):
        r"""Expand into fourier basis.

        Args:
            inputs (tf.Tensor, tf.RaggedTensor): Tensor input with position or distance to expand into encodings.
            dim_half (int): Dimension of the half output embedding space.
            include_frequencies (float): Whether to add

        Returns:
            tf.Tensor: Distance tensor expanded in Fourier basis.
        """
        k = -math.log(wave_length)
        steps = tf.range(dim_half, dtype=inputs.dtype)/dim_half
        freq = tf.exp(k * steps)
        scales = tf.cast(freq, dtype=inputs.dtype) * 2 * math.pi
        arg = inputs * scales
        # We would have to make an alternate, concatenate via additional axis and flatten it after,
        # but it is unclear, why this would make a difference for subsequent NN.
        out = tf.concat([tf.math.sin(arg), tf.math.cos(arg)], dim=-1)
        if include_frequencies:
            out = tf.concat([out, freq], dim=-1)
        return out

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (tf.RaggedTensor): Edge distance of shape `(batch, [K], 1)`

        Returns:
            tf.RaggedTensor: Expanded distance. Shape is `(batch, [K], bins)`.
        """
        return self.call_on_values_tensor_of_ragged(
            self._compute_gauss_basis, inputs,
            dim_half=self.dim_half, wave_length=self.wave_length, include_frequencies=self.include_frequencies)

    def get_config(self):
        """Update config."""
        config = super(PositionEncodingBasisLayer, self).get_config()
        config.update({"dim_half": self.dim_half,
                       "wave_length": self.wave_length, "include_frequencies": self.include_frequencies})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='BesselBasisLayer')
class BesselBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Bessel Basis with :math:`l=m=0`, according to
    `Klicpera et al. (2020) <https://arxiv.org/abs/2011.14115>`_.

    For :math:`l=m=0` the 2D spherical Fourier-Bessel simplifies to
    :math:`\Psi_{\text{RBF}}(d)=a j_0(\frac{z_{0,n}}{c}d)` with roots at :math:`z_{0,n} = n\pi`. With normalization
    on :math:`[0,c]` and :math:`j_0(d) = \sin{(d)}/d` yields
    :math:`\tilde{e}_{\text{RBF}} \in \mathbb{R}^{N_{\text{RBF}}}`:

    .. math::
        \tilde{e}_{\text{RBF}, n} (d) = \sqrt{\frac{2}{c}} \frac{\sin{\left(\frac{n\pi}{c} d\right)}}{d}

    Additionally, applies an envelope function :math:`u(d)` for continuous differentiability on the basis
    :math:`e_{\text{RBF}} = u(d)\tilde{e}_{\text{RBF}}`.
    By Default this is a polynomial of the form:

    .. math::
        u(d) = 1 − \frac{(p + 1)(p + 2)}{2} d^p + p(p + 2)d^{p+1} − \frac{p(p + 1)}{2} d^{p+2},

    where :math:`p \in \mathbb{N}_0` and typically :math:`p=6`.

    """

    def __init__(self, num_radial: int,
                 cutoff: float,
                 envelope_exponent: int = 5,
                 envelope_type: str = "poly",
                 **kwargs):
        r"""Initialize :obj:`BesselBasisLayer` layer.

        Args:
            num_radial (int): Number of radial basis functions to use.
            cutoff (float): Cutoff distance.
            envelope_exponent (int): Degree of the envelope to smoothen at cutoff. Default is 5.
            envelope_type (str): Type of envelope to use. Default is "poly".
        """
        super(BesselBasisLayer, self).__init__(**kwargs)
        # Layer variables
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope_exponent = envelope_exponent
        self.envelope_type = str(envelope_type)

        if self.envelope_type not in ["poly"]:
            raise ValueError("Unknown envelope type %s in BesselBasisLayer." % self.envelope_type)

        # Initialize frequencies at canonical positions.
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

    def expand_bessel_basis(self, inputs):
        d_scaled = inputs * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        out = d_cutoff * tf.sin(self.frequencies * d_scaled)
        return out

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape `(batch, [K], 1)`

        Returns:
            tf.RaggedTensor: Expanded distance. Shape is `(batch, [K], num_radial)`.
        """
        # Possibly faster RaggedRank==1
        if isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                out = self.expand_bessel_basis(inputs.values)
                return tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)
        return self.expand_bessel_basis(inputs)

    def get_config(self):
        """Update config."""
        config = super(BesselBasisLayer, self).get_config()
        config.update({"num_radial": self.num_radial, "cutoff": self.cutoff,
                       "envelope_exponent": self.envelope_exponent, "envelope_type": self.envelope_type})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='CosCutOffEnvelope')
class CosCutOffEnvelope(GraphBaseLayer):
    r"""Calculate cosine cutoff envelope according to
    `Behler et al. (2011) <https://aip.scitation.org/doi/10.1063/1.3553717>`_.

    For edge-like distance :math:`R_{ij}` and cutoff radius :math:`R_c` the envelope :math:`f_c` is given by:

    .. math::
        f_c(R_{ij}) = 0.5 [\cos{\frac{\pi R_{ij}}{R_c}} + 1]

    This layer only computes the cutoff envelope but does not apply it.

    """

    def __init__(self,
                 cutoff,
                 **kwargs):
        r"""Initialize layer.

        Args:
            cutoff (float): Cutoff distance :math:`R_c`.
        """
        super(CosCutOffEnvelope, self).__init__(**kwargs)
        self.cutoff = float(np.abs(cutoff)) if cutoff is not None else 1e8

    @staticmethod
    def _compute_cutoff_envelope(inputs, cutoff):
        """Implements the cutoff envelope."""
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape `(batch, [M], 1)`.

        Returns:
            tf.RaggedTensor: Cutoff envelope of shape `(batch, [M], 1)`.
        """
        return self.call_on_values_tensor_of_ragged(self._compute_cutoff_envelope, inputs, cutoff=self.cutoff)

    def get_config(self):
        """Update config."""
        config = super(CosCutOffEnvelope, self).get_config()
        config.update({"cutoff": self.cutoff})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='CosCutOff')
class CosCutOff(GraphBaseLayer):
    r"""Apply cosine cutoff according to
    `Behler et al. (2011) <https://aip.scitation.org/doi/10.1063/1.3553717>`_.

    For edge-like distance :math:`R_{ij}` and cutoff radius :math:`R_c` the envelope :math:`f_c` is given by:

    .. math::
        f_c(R_{ij}) = 0.5 [\cos{\frac{\pi R_{ij}}{R_c}} + 1]

    This layer computes the cutoff envelope and applies it to the input by simply multiplying with the envelope.

    """

    def __init__(self,
                 cutoff,
                 **kwargs):
        r"""Initialize layer.

        Args:
            cutoff (float): Cutoff distance :math:`R_c`.
        """
        super(CosCutOff, self).__init__(**kwargs)
        self.cutoff = float(np.abs(cutoff)) if cutoff is not None else 1e8

    @staticmethod
    def _compute_cutoff(inputs, cutoff):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        out = fc * inputs
        return out

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape `(batch, [M], D)`

        Returns:
            tf.RaggedTensor: Cutoff applied to input of shape `(batch, [M], D)`.
        """
        return self.call_on_values_tensor_of_ragged(self._compute_cutoff, inputs, cutoff=self.cutoff)

    def get_config(self):
        """Update config."""
        config = super(CosCutOff, self).get_config()
        config.update({"cutoff": self.cutoff})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='DisplacementVectorsASU')
class DisplacementVectorsASU(GraphBaseLayer):
    """TODO: Add docs.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        self.gather_node_positions = NodePosition()
        super(DisplacementVectorsASU, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(DisplacementVectorsASU, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [frac_coordinates, edge_indices, symmetry_ops]

                - frac_coordinates (tf.RaggedTensor): Fractional node coordinates of shape `(batch, [N], 3)`.
                - edge_indices (tf.RaggedTensor): Edge indices of shape `(batch, [M], 2)`.
                - symmetry_ops (tf.RaggedTensor): Symmetry operations of shape `(batch, [M], 4, 4)`.

        Returns:
            tf.RaggedTensor: Displacement vector for edges of shape `(batch, [M], 3)`.
        """
        inputs = self.assert_ragged_input_rank(inputs, ragged_rank=1)

        frac_coords = inputs[0]
        edge_indices = inputs[1]
        symmops = inputs[2].values

        cell_translations = inputs[3].values
        in_frac_coords, out_frac_coords = self.gather_node_positions([frac_coords, edge_indices], **kwargs)
        in_frac_coords = in_frac_coords.values
        out_frac_coords = out_frac_coords.values
        
        # Affine Transformation
        out_frac_coords_ = tf.concat(
            [out_frac_coords, tf.expand_dims(tf.ones_like(out_frac_coords[:, 0]), axis=1)], axis=1)
        affine_matrices = symmops
        out_frac_coords = tf.einsum('ij,ikj->ik', out_frac_coords_, affine_matrices)[:, :-1]
        out_frac_coords = out_frac_coords - tf.floor(out_frac_coords)  # All values should be in [0,1) interval
        
        # Cell translation
        out_frac_coords = out_frac_coords + cell_translations
        
        offset = in_frac_coords - out_frac_coords
        return tf.RaggedTensor.from_row_splits(offset, edge_indices.row_splits, validate=self.ragged_validate)


@ks.utils.register_keras_serializable(package='kgcnn', name='DisplacementVectorsUnitCell')
class DisplacementVectorsUnitCell(GraphBaseLayer):
    """TODO: Add docs.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        self.gather_node_positions = NodePosition()
        self.lazy_add = LazyAdd()
        self.lazy_sub = LazySubtract()
        super(DisplacementVectorsUnitCell, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(DisplacementVectorsUnitCell, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [frac_coordinates, edge_indices, cell_translations]

                - frac_coordinates (tf.RaggedTensor): Fractional node coordinates of shape `(batch, [N], 3)`.
                - edge_indices (tf.RaggedTensor): Edge indices of shape `(batch, [M], 2)`.
                - cell_translations (tf.RaggedTensor): Displacement across unit cell of shape `(batch, [M], 3)`.

        Returns:
            tf.RaggedTensor: Displacement vector for edges of shape `(batch, [M], 3)`.
        """
        frac_coords = inputs[0]
        edge_indices = inputs[1]
        cell_translations = inputs[2]

        # Gather sending and receiving coordinates.
        in_frac_coords, out_frac_coords = self.gather_node_positions([frac_coords, edge_indices], **kwargs)
        
        # Cell translation
        out_frac_coords = self.lazy_add([out_frac_coords, cell_translations], **kwargs)
        offset = self.lazy_sub([in_frac_coords, out_frac_coords], **kwargs)
        return offset


@ks.utils.register_keras_serializable(package='kgcnn', name='FracToRealCoordinates')
class FracToRealCoordinates(GraphBaseLayer):
    """TODO: Add docs.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        self.gather_state = GatherState()
        super(FracToRealCoordinates, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(FracToRealCoordinates, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [frac_coordinates, lattice_matrix]

                - frac_coordinates (tf.RaggedTensor): Fractional node coordinates of shape `(batch, [N], 3)`.
                - lattice_matrix (tf.Tensor): Lattice matrix of shape `(batch, 3, 3)`.

        Returns:
            tf.RaggedTensor: Real-space node coordinates of shape `(batch, [N], 3)`.
        """
        frac_coords = inputs[0]
        lattice_matrices = inputs[1]
        lattice_matrices_ = tf.repeat(lattice_matrices, frac_coords.row_lengths(), axis=0)
        real_coords = tf.einsum('ij,ikj->ik', frac_coords.values,  lattice_matrices_)
        return tf.RaggedTensor.from_row_splits(real_coords, frac_coords.row_splits, validate=self.ragged_validate)
