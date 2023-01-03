import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesSelection, GatherNodesIngoing
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition
from kgcnn.layers.pooling import PoolingLocalEdges, RelationalPoolingLocalEdges
from kgcnn.layers.modules import LazyMultiply, LazySubtract, ExpandDims

ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ACSFRadial')
class ACSFRadial(GraphBaseLayer):
    r"""Atom-centered symmetry functions (ACSF) for high-dimensional neural network potentials (HDNNPs).

    This layer implements the radial part :math:`W_{i}^{rad}` :

    .. math::

        W_{i}^{rad} = \sum_{j \neq i} \; e^{−\eta \, (r_{ij} − \mu)^{2} } \; f_{ij}

    Here, for each atom type there is a set of parameters :math:`\eta` and :math:`\mu` and cutoff.
    The cutoff function :math:`f_ij = f_c(r_{ij})` is given by:

    .. math::

        f_c(r_{ij}) = 0.5 [\cos{\frac{\pi r_{ij}}{R_c}} + 1]

    In principle these parameters can be made trainable. The above sum is conducted for each atom type.

    Example:

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.conv.acsf_conv import ACSFRadial
        layer = ACSFRadial(
            eta_rs_rc=[[[0.0, 0.0, 8.0], [1.0, 0.0, 8.0]],[[0.0, 0.0, 8.0], [1.0, 0.0, 8.0]]],
            element_mapping=[1, 6]
        )
        z = tf.ragged.constant([[1, 6]], ragged_rank=1)
        xyz = tf.ragged.constant([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], ragged_rank=1, inner_shape=(3,))
        eij = tf.ragged.constant([[[0,1], [1, 0]]], ragged_rank=1, inner_shape=(2,))
        rep_i = layer([z, xyz, eij])

    """

    _max_atomic_number = 96

    def __init__(self,
                 eta_rs_rc: list = None,
                 element_mapping: list = None,
                 add_eps: bool = False,
                 param_constraint=None, param_regularizer=None, param_initializer="zeros",
                 **kwargs):
        r"""Initialize layer.

            Args:
                eta_rs_rc (list, np.ndarray): List of shape `(N, N, m, 3)` or `(N, m, 3)` where `N` are the considered
                    atom types and m the number of representations. Tensor output will be shape `(batch, None, m*N)` .
                    In the last dimension are the values for :math:`eta`, :math:`R_s` and :math:`R_c` .
                    In case of shape `(N, m, 3)` the values are repeated for each `N` to fit `(N, N, m, 3)` .
                element_mapping (list): Atomic numbers of elements in :obj:`eta_rs_rc` , must have shape `(N, )` .
                    Should not contain duplicate elements.
                add_eps (bool): Whether to add epsilon.
                param_constraint: Parameter constraint for weights. Default is None.
                param_regularizer: Parameter regularizer for weights. Default is None.
                param_initializer: Parameter initializer for weights. Default is "zeros".
        """
        super(ACSFRadial, self).__init__(**kwargs)
        # eta_rs_rc of shape (N, N, m, 3) with m combinations of eta, rs, rc
        # or simpler (N, m, 3) where we repeat an additional N dimension assuming same parameter of source.
        self.eta_rs_rc = np.array(eta_rs_rc)
        if len(self.eta_rs_rc.shape) == 3:
            self.eta_rs_rc = np.expand_dims(self.eta_rs_rc, axis=0)
            self.eta_rs_rc = np.repeat(self.eta_rs_rc, self.eta_rs_rc.shape[1], axis=0)
        assert len(self.eta_rs_rc.shape) == 4, "Require `eta_rs_rc` parameter matrix of shape (N, N, m, 3)"
        self.element_mapping = np.array(element_mapping, dtype="int")  # of shape (N, ) with atomic number for eta_rs_rc
        self.reverse_mapping = np.empty(self._max_atomic_number, dtype="int")
        self.reverse_mapping.fill(np.iinfo(self.reverse_mapping.dtype).max)
        for i, pos in enumerate(self.element_mapping):
            self.reverse_mapping[pos] = i
        self.add_eps = add_eps

        self.lazy_mult = LazyMultiply()
        self.layer_pos = NodePosition()
        self.layer_gather = GatherNodesSelection([0, 1])
        self.layer_exp_dims = ExpandDims(axis=2)
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = RelationalPoolingLocalEdges(num_relations=self.eta_rs_rc.shape[1], pooling_method="sum")

        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = param_initializer
        self.param_regularizer = param_regularizer
        self.param_constraint = param_constraint

        self.weight_eta_rs_rc = self.add_weight(
            "eta_rs_rc",
            shape=self.eta_rs_rc.shape,
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=False
        )
        self.weight_reverse_mapping = self.add_weight(
            "reverse_mapping",
            shape=(self._max_atomic_number, ),
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype="int64", trainable=False
        )

        self.set_weights([self.eta_rs_rc, self.reverse_mapping])

    def _find_params_per_bond(self, inputs: list):
        zi, zj = inputs
        zi_map = tf.gather(self.weight_reverse_mapping, zi, axis=0)
        zj_map = tf.gather(self.weight_reverse_mapping, zj, axis=0)
        params = tf.gather(tf.gather(self.weight_eta_rs_rc, zi_map, axis=0), zj_map, axis=1, batch_dims=1)
        return params

    @staticmethod
    def _compute_fc(inputs: tf.Tensor):
        rij, params = inputs
        cutoff = tf.gather(params, 2, axis=-1)
        fc = tf.clip_by_value(tf.broadcast_to(rij, tf.shape(cutoff)), -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    @staticmethod
    def _compute_gaussian_expansion(inputs: tf.Tensor):
        rij, params = inputs
        eta, mu = tf.gather(params, 0, axis=-1), tf.gather(params, 1, axis=-1)
        arg = tf.square(rij - mu) * eta
        return tf.exp(-arg)

    @staticmethod
    def _flatten_relations(inputs):
        input_shape = tf.shape(inputs)
        flatten_shape = tf.concat(
            [input_shape[:1], tf.constant([inputs.shape[1]*inputs.shape[2]], dtype=input_shape.dtype)], axis=0)
        return tf.reshape(inputs, flatten_shape)

    def build(self, input_shape):
        super(ACSFRadial, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [z, xyz, ij]

                - z (tf.RaggedTensor): Atomic numbers of shape (batch, [N])
                - xyz (tf.RaggedTensor): Node coordinates of shape (batch, [N], 3)
                - ij (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Atomic representation of shape `(batch, None, units)` .
        """
        z, xyz, eij = inputs
        z = self.map_values(tf.cast, z, dtype=eij.dtype)
        xi, xj = self.layer_pos([xyz, eij], **kwargs)
        rij = self.layer_dist([xi, xj], **kwargs)
        zi, zj = self.layer_gather([z, eij])
        params_per_bond = self.map_values(self._find_params_per_bond, [zi, zj])
        fc = self.map_values(self._compute_fc, [rij, params_per_bond])
        gij = self.map_values(self._compute_gaussian_expansion, [rij, params_per_bond])
        rep = self.lazy_mult([gij, fc], **kwargs)
        pooled = self.pool_sum([xyz, rep, eij, zj], **kwargs)
        return self.map_values(self._flatten_relations, pooled)

    def get_config(self):
        config = super(ACSFRadial, self).get_config()
        config.update({"eta_rs_rc": self.eta_rs_rc.tolist(),
                       "element_mapping": self.element_mapping.tolist(),
                       "add_eps": self.add_eps,
                       "param_constraint": self.param_constraint,
                       "param_regularizer": self.param_regularizer,
                       "param_initializer": self.param_initializer
                       })
        return config
