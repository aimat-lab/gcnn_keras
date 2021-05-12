import tensorflow as tf
import tensorflow.keras as ks
# import tensorflow.tensorflow_graphics as tfg
import numpy as np

from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type
from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition

class NodeDistance(ks.layers.Layer):
    """
    Compute geometric node distances similar to edges.

    A distance based edge is defined by edge or bond index in index list of shape (batch, None, 2) with last dimension
    of ingoing and outgoing.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
    """

    def __init__(self,
                 node_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(NodeDistance, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(NodeDistance, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs (list): [position, edge_index]

            - position: Node positions of shape (batch, [N], 3)
            - edge_index: Edge indices of shape (batch, [M], 2)

        Returns:
            distances: Gathered node distances as edges that match the number of indices of shape (batch, [M], 1)
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_index_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                       node_indexing=self.node_indexing)

        # We cast to values here
        node, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)
        edge_index, edge_part = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_index_type,
                                                   output_tensor_type="values_partition",
                                                   partition_type=self.partition_type)

        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node=self.partition_type,
                                                                           partition_type_edge=self.partition_type,
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        xi = tf.gather(node, indexlist[:, 0], axis=0)
        xj = tf.gather(node, indexlist[:, 1], axis=0)

        out = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.math.square(xi - xj), axis=-1)))

        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index[:, :, 0], batch_dims=1)

        return kgcnn_ops_dyn_cast([out, edge_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_index_type, partition_type=self.partition_type)

    def get_config(self):
        """Update config."""
        config = super(NodeDistance, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config


class NodeAngle(ks.layers.Layer):
    """
    Compute geometric node angles between edges.

    The geometric angle is computed between i<-j,j<-k for index tuple (i,j,k) in (batch, None, 3) last dimension.
    of ingoing and outgoing.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
    """

    def __init__(self,
                 node_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(NodeAngle, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(NodeAngle, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs (list): [position, edge_index]

            - position: Node positions of shape (batch, [N], 3)
            - edge_index: Edge indices of shape (batch, [M], 3)

        Returns:
            distances: Gathered node angles between edges that match the indices. Shape is (batch, [M], 1)
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_index_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                       node_indexing=self.node_indexing)

        # We cast to values here
        node, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)
        edge_index, edge_part = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_index_type,
                                                   output_tensor_type="values_partition",
                                                   partition_type=self.partition_type)

        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node=self.partition_type,
                                                                           partition_type_edge=self.partition_type,
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        xi = tf.gather(node, indexlist[:, 0], axis=0)
        xj = tf.gather(node, indexlist[:, 1], axis=0)
        xk = tf.gather(node, indexlist[:, 2], axis=0)
        v1 = xj - xi
        v2 = xk - xj
        x = tf.reduce_sum(v1 * v2, axis=-1)
        y = tf.linalg.cross(v1, v2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        angle = tf.expand_dims(angle,axis=-1)
        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index[:, :, 0], batch_dims=1)

        return kgcnn_ops_dyn_cast([angle, edge_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_index_type, partition_type=self.partition_type)

    def get_config(self):
        """Update config."""
        config = super(NodeAngle, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config



class BesselBasisLayer(ks.layers.Layer):
    """
    Expand a distance into BesselBasis, according to Klicpera et al. 2020

    Args:
        num_radial (int): Number of radial radial basis functions
        cutoff (float): Cutoff distance c
        envelope_exponent (int): Degree of the envelope to smoothen at cutoff. Default is 5.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
    """

    def __init__(self, num_radial,
                 cutoff,
                 envelope_exponent=5,
                 node_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        super(BesselBasisLayer, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected

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
        env_val = 1.0 / inputs + a*inputs**(p - 1) + b*inputs**p + c*inputs**(p + 1)
        return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))

    def call(self, inputs, **kwargs):
        found_node_type = kgcnn_ops_check_tensor_type(inputs, input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        # We cast to values here
        node, node_part = kgcnn_ops_dyn_cast(inputs, input_tensor_type=found_node_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)



        d_scaled = node * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        out = d_cutoff * tf.sin(self.frequencies * d_scaled)
        out = kgcnn_ops_dyn_cast([out, node_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_node_type, partition_type=self.partition_type)
        return out

    def get_config(self):
        """Update config."""
        config = super(BesselBasisLayer, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate,
                       "num_radial": self.num_radial,
                       "cutoff": self.cutoff,
                       "envelope_exponent": self.envelope_exponent
                       })
        return config




class SphericalBasisLayer(ks.layers.Layer):

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
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)
        self.sph_harm_formulas = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to tensorflow functions
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0], 'tensorflow')(0)
                self.sph_funcs.append(lambda tensor: tf.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0], 'tensorflow'))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j], 'tensorflow'))

    @tf.function
    def envelope(self, inputs):
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / inputs + a*inputs**(p - 1) + b*inputs**p + c*inputs**(p + 1)
        return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))

    def call(self, inputs, **kwargs):
        d, angles, id_expand_kj = inputs

        d_scaled = d * self.inv_cutoff
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = tf.stack(rbf, axis=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf
        rbf_env = tf.gather(rbf_env, id_expand_kj)

        cbf = [f(angles) for f in self.sph_funcs]
        cbf = tf.stack(cbf, axis=1)
        cbf = tf.repeat(cbf, self.num_radial, axis=1)

        return rbf_env * cbf