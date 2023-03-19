import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesSelection, GatherNodesIngoing
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazyMultiply, LazySubtract, ExpandDims

ks = tf.keras

# Parameters for Weighted Atom-Centered Symmetry Functions (wACSF).
# We set the unoptimized C default values as default for all 118 atomic species.
# They are taken from https://aip.scitation.org/doi/suppl/10.1063/1.5019667/suppl_file/si.pdf .
# We chose the 22:10 radial:angular as default values. Other combinations ranging from 0:32 to 32:0 .
radial_eta_mu_defaults = np.array([[
    [4.5000000, 7.5000000, 8.0], [4.5000000, 7.1667000, 8.0], [4.5000000, 6.8333000, 8.0], [4.5000000, 6.5000000, 8.0],
    [4.5000000, 6.1667000, 8.0], [4.5000000, 5.8333000, 8.0], [4.5000000, 5.5000000, 8.0], [4.5000000, 5.1667000, 8.0],
    [4.5000000, 4.8333000, 8.0], [4.5000000, 4.5000000, 8.0], [4.5000000, 4.1667000, 8.0], [4.5000000, 3.8333000, 8.0],
    [4.5000000, 3.5000000, 8.0], [4.5000000, 3.1667000, 8.0], [4.5000000, 2.8333000, 8.0], [4.5000000, 2.5000000, 8.0],
    [4.5000000, 2.1667000, 8.0], [4.5000000, 1.8333000, 8.0], [4.5000000, 1.5000000, 8.0], [4.5000000, 1.1667000, 8.0],
    [4.5000000, 0.8333000, 8.0], [4.5000000, 0.5000000, 8.0]
]] * 118)
angular_eta_mu_lambda_zeta_defaults = np.array(
    [[[0.03306120, 0.0, -1.0, 1.0, 8.00], [0.03306120, 0.0, 1.0, 1.0, 8.00], [0.04986150, 0.0, -1.0, 1.0, 8.00],
      [0.04986150, 0.0, 1.0, 1.0, 8.00], [0.08367770, 0.0, -1.0, 1.0, 8.00], [0.08367770, 0.0, 1.0, 1.0, 8.00],
      [0.16857440, 0.0, -1.0, 1.0, 8.00], [0.16857440, 0.0, 1.0, 1.0, 8.00], [0.50000000, 0.0, -1.0, 1.0, 8.00],
      [0.50000000, 0.0, 1.0, 1.0, 8.00]]] * 118)
# Update optimized parameters for H, C, F, O, N .
radial_eta_mu_defaults[1] = np.array(
    [[4.3814240, 2.4432533, 8.0], [4.4149216, 6.1667000, 8.0], [4.4493159, 3.8332928, 8.0], [4.4815889, 3.1667000, 8.0],
     [4.4865336, 3.5000000, 8.0], [4.4937252, 4.8332601, 8.0], [4.4986009, 5.5010689, 8.0], [4.4997975, 2.1558150, 8.0],
     [4.4999291, 7.4956238, 8.0], [4.4999905, 6.4964721, 8.0], [4.5000000, 5.2730567, 8.0], [4.5000011, 1.1606611, 8.0],
     [4.5000039, 0.4918029, 8.0], [4.5000398, 0.8333002, 8.0], [4.5002241, 1.8334858, 8.0], [4.5036121, 2.9532907, 8.0],
     [4.5059680, 5.8330954, 8.0], [4.5062245, 6.8333000, 8.0], [4.5064492, 7.1666408, 8.0], [4.5675834, 4.5046412, 8.0],
     [4.5899630, 4.1663934, 8.0], [4.6604619, 1.4989046, 8.0]])
angular_eta_mu_lambda_zeta_defaults[1] = np.array(
    [[0.01919876, 0.0, -1.0, 1.0, 8.00], [0.03306120, 0.0, 1.0, 1.0, 8.00], [0.03447591, 0.0, -1.0, 2.0, 8.00],
     [0.04986150, 0.0, -1.0, 1.0, 8.00], [0.08235996, 0.0, -1.0, 1.0, 8.00], [0.08367770, 0.0, 1.0, 2.0, 8.00],
     [0.16857440, 0.0, -1.0, 2.0, 8.00], [0.16953019, 0.0, 1.0, 2.0, 8.00], [0.50000000, 0.0, -1.0, 1.0, 8.00],
     [0.50000000, 0.0, 1.0, 1.0, 8.00]]
)
radial_eta_mu_defaults[6] = np.array(
    [[4.4534562, 3.8338779, 8.0], [4.4804536, 5.1667000, 8.0], [4.4918531, 4.8333000, 8.0], [4.4991551, 0.8333000, 8.0],
     [4.4998245, 2.4997843, 8.0], [4.4999043, 1.5993354, 8.0], [4.4999230, 5.8324479, 8.0], [4.4999997, 6.5117668, 8.0],
     [4.5000000, 6.1860418, 8.0], [4.5000000, 3.1584848, 8.0], [4.5000001, 4.4991770, 8.0], [4.5000132, 6.8333000, 8.0],
     [4.5001099, 7.4999458, 8.0], [4.5002911, 2.1638163, 8.0], [4.5004121, 0.3038147, 8.0], [4.5014183, 5.4361520, 8.0],
     [4.5023686, 1.8302079, 8.0], [4.5027053, 3.5002020, 8.0], [4.5074706, 7.2656992, 8.0], [4.5104731, 1.1645345, 8.0],
     [4.5515436, 4.1666210, 8.0], [4.5906934, 2.8328539, 8.0]])
angular_eta_mu_lambda_zeta_defaults[6] = np.array(
    [[0.03392104, 0.0, 1.0, 2.0, 8.00], [0.04232205, 0.0, -1.0, 2.0, 8.00], [0.04663133, 0.0, -1.0, 2.0, 8.00],
     [0.05097753, 0.0, -1.0, 1.0, 8.00], [0.08367770, 0.0, -1.0, 3.0, 8.00], [0.08368552, 0.0, -1.0, 1.0, 8.00],
     [0.16857440, 0.0, 1.0, 2.0, 8.00], [0.25965214, 0.0, 1.0, 2.0, 8.00], [0.47868498, 0.0, -1.0, 1.0, 8.00],
     [0.56008922, 0.0, 1.0, 1.0, 8.00]]
)
radial_eta_mu_defaults[9] = np.array(
    [[4.3755129, 4.3891938, 8.0], [4.4739340, 6.5000000, 8.0], [4.4934064, 1.1667000, 8.0], [4.4955168, 3.1421155, 8.0],
     [4.4982145, 6.1665305, 8.0], [4.4986470, 0.8333000, 8.0], [4.4989965, 7.4999930, 8.0], [4.4999944, 4.1666929, 8.0],
     [4.4999998, 6.8333000, 8.0], [4.4999998, 2.8332546, 8.0], [4.4999999, 5.8424471, 8.0], [4.5000000, 5.4954750, 8.0],
     [4.5000000, 4.8332314, 8.0], [4.5000000, 2.1675890, 8.0], [4.5000000, 1.8332281, 8.0], [4.5000016, 3.8334246, 8.0],
     [4.5000047, 7.1667013, 8.0], [4.5002559, 0.5866000, 8.0], [4.5004160, 5.1667000, 8.0], [4.5165951, 1.5000000, 8.0],
     [4.5200686, 3.5000000, 8.0], [4.5333816, 2.5065631, 8.0]])
angular_eta_mu_lambda_zeta_defaults[9] = np.array(
    [[0.03302291, 0.0, -1.0, 1.0, 8.00], [0.03306120, 0.0, 1.0, 1.0, 8.00], [0.04986182, 0.0, -1.0, 1.0, 8.00],
     [0.05024516, 0.0, 1.0, 1.0, 8.00], [0.08367770, 0.0, -1.0, 2.0, 8.00], [0.08367770, 0.0, 1.0, 1.0, 8.00],
     [0.16857440, 0.0, 1.0, 2.0, 8.00], [0.16857440, 0.0, 1.0, 1.0, 8.00], [0.49854417, 0.0, 1.0, 1.0, 8.00],
     [0.50003188, 0.0, 1.0, 1.0, 8.00]]
)
radial_eta_mu_defaults[8] = np.array(
    [[4.3810310, 6.1634839, 8.0], [4.4613793, 4.7025864, 8.0], [4.4708866, 0.8337341, 8.0], [4.4741510, 7.1654091, 8.0],
     [4.4934208, 0.4909963, 8.0], [4.4999979, 6.7895512, 8.0], [4.4999984, 2.5548726, 8.0], [4.4999996, 5.1188661, 8.0],
     [4.4999999, 6.4941667, 8.0], [4.5000000, 7.5269793, 8.0], [4.5000000, 3.5008886, 8.0], [4.5000000, 3.1673548, 8.0],
     [4.5000000, 3.0664616, 8.0], [4.5000000, 1.1771551, 8.0], [4.5000010, 5.9628537, 8.0], [4.5000305, 1.6101757, 8.0],
     [4.5000505, 3.8352953, 8.0], [4.5000877, 5.2259600, 8.0], [4.5003116, 2.1667069, 8.0], [4.5007395, 1.8264699, 8.0],
     [4.5037122, 4.1666850, 8.0], [4.5108571, 4.7474998, 8.0]])
angular_eta_mu_lambda_zeta_defaults[8] = np.array(
    [[0.03306120, 0.0, -1.0, 1.0, 8.00], [0.04970837, 0.0, -1.0, 1.0, 8.00], [0.04986150, 0.0, 1.0, 1.0, 8.00],
     [0.05680303, 0.0, -1.0, 1.0, 8.00], [0.08366889, 0.0, 1.0, 1.0, 8.00], [0.08367770, 0.0, -1.0, 1.0, 8.00],
     [0.16857440, 0.0, 1.0, 2.0, 8.00], [0.16857440, 0.0, 1.0, 1.0, 8.00], [0.50000000, 0.0, -1.0, 1.0, 8.00],
     [0.50000000, 0.0, 1.0, 1.0, 8.00]]
)
radial_eta_mu_defaults[7] = np.array(
    [[4.4148209, 6.8472192, 8.0], [4.4481457, 1.0317668, 8.0], [4.4608631, 0.4581865, 8.0], [4.4797608, 3.9799903, 8.0],
     [4.4951823, 0.8337448, 8.0], [4.4987030, 7.4573738, 8.0], [4.4998898, 6.9779219, 8.0], [4.4999884, 2.1676806, 8.0],
     [4.4999920, 3.1667000, 8.0], [4.5000000, 6.1662980, 8.0], [4.5000000, 5.8328781, 8.0], [4.5000000, 5.4757644, 8.0],
     [4.5000000, 5.1666981, 8.0], [4.5000000, 3.3518840, 8.0], [4.5000000, 2.4982784, 8.0], [4.5001975, 4.8204812, 8.0],
     [4.5004709, 6.5050056, 8.0], [4.5016337, 4.1667000, 8.0], [4.5077984, 1.4994658, 8.0], [4.5228543, 2.8333000, 8.0],
     [4.5780852, 4.5000000, 8.0], [4.5897686, 1.8340302, 8.0]])
angular_eta_mu_lambda_zeta_defaults[7] = np.array(
    [[0.03306120, 0.0, -1.0, 1.0, 8.00], [0.03306120, 0.0, 1.0, 1.0, 8.00], [0.04907151, 0.0, -1.0, 2.0, 8.00],
     [0.07102913, 0.0, 1.0, 1.0, 8.00], [0.07116163, 0.0, 1.0, 1.0, 8.00], [0.08384128, 0.0, -1.0, 1.0, 8.00],
     [0.16857391, 0.0, -1.0, 2.0, 8.00], [0.16902220, 0.0, 1.0, 1.0, 8.00], [0.50000000, 0.0, -1.0, 1.0, 8.00],
     [0.50000000, 0.0, 1.0, 1.0, 8.00]]
)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFRad')
class wACSFRad(GraphBaseLayer):
    r"""Weighted atom-centered symmetry functions (wACSF) for high-dimensional neural network potentials (HDNNPs).
    From `Gastegger et al. (2017) <https://arxiv.org/abs/1712.05861>`_ .
    Default values can be found in `<https://aip.scitation.org/doi/suppl/10.1063/1.5019667>`_ .
    This layer implements the radial part :math:`W_{i}^{rad}` :

    .. math::

        W_{i}^{rad} = \sum_{j \neq i} \; g(Z_j) \; e^{−\eta \, (r_{ij} − \mu)^{2} } \; f_{ij}

    Here, for each atom type there is a set of parameters :math:`\eta` and :math:`\mu` .
    The cutoff function :math:`f_ij = f_c(r_{ij})` is given by:

    .. math::

        f_c(r_{ij}) = 0.5 [\cos{\frac{\pi r_{ij}}{R_c}} + 1]

    The cutoff radius is implemented as a float and can not be changed dependent on the atom type.
    Not that the parameters :math:`\eta` and :math:`\mu` can be made trainable.

    """

    def __init__(self, eta_mu: list = None, cutoff: float = 8.0, add_eps: bool = False,
                 use_external_weights: bool = False, param_constraint=None, param_regularizer=None,
                 param_initializer="zeros", **kwargs):
        super(wACSFRad, self).__init__(**kwargs)
        self.cutoff = cutoff
        self.add_eps = add_eps
        if eta_mu is None:
            eta_mu = radial_eta_mu_defaults[:, :, :2]
        self.eta_mu = np.array(eta_mu, dtype="float").tolist()
        self.use_external_weights = use_external_weights
        self.lazy_mult = LazyMultiply()
        self.layer_pos = NodePosition()
        self.layer_gather_out = GatherNodesOutgoing()
        self.layer_gather_in = GatherNodesIngoing()
        self.layer_exp_dims = ExpandDims(axis=2)
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = PoolingLocalEdges(pooling_method="sum")

        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = ks.initializers.deserialize(param_initializer)
        self.param_regularizer = ks.regularizers.deserialize(param_regularizer)
        self.param_constraint = ks.constraints.deserialize(param_constraint)
        np_eta_mu = np.array(eta_mu)
        weight_shape = np_eta_mu.shape
        self._weight_eta_mu = self.add_weight(
            "eta_mu",
            shape=weight_shape,
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=False
        )
        self.set_weights([np_eta_mu])

    @staticmethod
    def _compute_fc(inputs: tf.Tensor, cutoff: float):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    def _compute_gaussian_expansion(self, inputs: tf.Tensor):
        rij, zi = inputs
        params = tf.gather(self._weight_eta_mu, zi, axis=0)
        # eta, mu = tf.split(params, [1, 1], axis=-1)
        # eta, mu = tf.squeeze(eta, axis=-1), tf.squeeze(mu, axis=-1)
        eta, mu = tf.gather(params, 0, axis=-1), tf.gather(params, 1, axis=-1)
        arg = tf.square(rij - mu) * eta
        return tf.exp(-arg)

    def build(self, input_shape):
        super(wACSFRad, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [z, xyz, ij] or [z, xyz, ij, w]  if `use_external_weights`

                - z (tf.RaggedTensor): Atomic numbers of shape (batch, [N])
                - xyz (tf.RaggedTensor): Node coordinates of shape (batch, [N], 3)
                - ij (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
                - w (tf.RaggedTensor): Edge weight tensor of shape (batch, [M], 1)

            mask: Boolean mask for inputs. Not used. Defaults to None.

        Returns:
            tf.RaggedTensor: Atomic representation of shape `(batch, None, units)` .
        """
        if self.use_external_weights:
            z, xyz, eij, w = self.assert_ragged_input_rank(inputs, mask=mask, ragged_rank=1)
            z = self.map_values(tf.cast, z, dtype=eij.dtype)
        else:
            z, xyz, eij = self.assert_ragged_input_rank(inputs, mask=mask, ragged_rank=1)
            z = self.map_values(tf.cast, z, dtype=eij.dtype)
            zj = self.layer_gather_out([z, eij], **kwargs)
            w = self.layer_exp_dims(zj, **kwargs)
        w = self.map_values(tf.cast, w, dtype=self.dtype)
        xi, xj = self.layer_pos([xyz, eij], **kwargs)
        rij = self.layer_dist([xi, xj], **kwargs)
        fc = self.map_values(self._compute_fc, rij, cutoff=self.cutoff)
        zi = self.layer_gather_in([z, eij], **kwargs)
        gij = self.map_values(self._compute_gaussian_expansion, [rij, zi])
        rep = self.lazy_mult([gij, fc, w], **kwargs)
        return self.pool_sum([xyz, rep, eij], **kwargs)

    def get_config(self):
        config = super(wACSFRad, self).get_config()
        config.update({"eta_mu": self.eta_mu, "cutoff": self.cutoff, "use_external_weights": self.use_external_weights,
                       "add_eps": self.add_eps,
                       "param_constraint": ks.constraints.serialize(self.param_constraint),
                       "param_regularizer": ks.regularizers.serialize(self.param_regularizer),
                       "param_initializer": ks.initializers.serialize(self.param_initializer)
                       })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFAng')
class wACSFAng(GraphBaseLayer):
    r"""Weighted atom-centered symmetry functions (wACSF) for high-dimensional neural network potentials (HDNNPs).
    From `Gastegger et al. (2017) <https://arxiv.org/abs/1712.05861>`_ .
    Default values can be found in `<https://aip.scitation.org/doi/suppl/10.1063/1.5019667>`_ .
    This layer implements the angular part :math:`W_{i}^{ang}` :

    .. math::

        W_{i}^{ang} = 2^{1−\zeta} \; \sum_{j\neq i} \sum_{k\neq j,i} \; h(Z_j , Z_k) \;
        (1 + \lambda\, \cos{\theta_{ijk}})^\zeta\;
        \times \; e^{−\eta (r_{ij} −\mu )^2} \; e^{−\eta (r_{ik} −\mu )^2} \; e^{−\eta (r_{jk} −\mu )^2} \;
        \times \; f_{ij} \; f_{ik} \; f_{jk}

    Here, for each atom type there is a set of parameters :math:`\eta` , :math:`\mu` , :math:`\lambda`
    and :math:`\zeta`.
    The cutoff function :math:`f_ij = f_c(r_{ij})` is given by:

    .. math::

        f_c(r_{ij}) = 0.5 [\cos{\frac{\pi r_{ij}}{R_c}} + 1]

    The cutoff radius is implemented as a float and can not be changed dependent on the atom type.
    Not that the parameters :math:`\eta` to :math:`\lambda` can be made trainable.

    """
    def __init__(self, eta_mu_lambda_zeta: list = None, cutoff: float = 8.0,
                 add_eps: bool = False, use_external_weights: bool = False, param_initializer="zeros",
                 param_regularizer=None, param_constraint=None, **kwargs):
        super(wACSFAng, self).__init__(**kwargs)
        self.add_eps = add_eps
        self.cutoff = cutoff
        if eta_mu_lambda_zeta is None:
            eta_mu_lambda_zeta = angular_eta_mu_lambda_zeta_defaults[:, :, :4]
        self.eta_mu_lambda_zeta = np.array(eta_mu_lambda_zeta, dtype="float").tolist()
        self.use_external_weights = use_external_weights
        self.lazy_mult = LazyMultiply()
        self.layer_pos = NodePosition(selection_index=[0, 1, 2])
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = PoolingLocalEdges(pooling_method="sum")
        self.lazy_sub = LazySubtract()
        self.layer_gather_in = GatherNodesIngoing()
        self.layer_gather_1 = GatherNodesSelection(selection_index=1)
        self.layer_gather_2 = GatherNodesSelection(selection_index=2)
        self.layer_exp_dims = ExpandDims(axis=2)
        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = ks.initializers.deserialize(param_initializer)
        self.param_regularizer = ks.regularizers.deserialize(param_regularizer)
        self.param_constraint = ks.constraints.deserialize(param_constraint)
        np_eta_mu_lambda_zeta = np.array(eta_mu_lambda_zeta)
        weight_shape = np_eta_mu_lambda_zeta.shape
        self._weight_eta_mu_lambda_zeta = self.add_weight(
            "eta_mu_lambda_zeta",
            shape=weight_shape,
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=False
        )
        self.set_weights([np_eta_mu_lambda_zeta])

    @staticmethod
    def _compute_fc(inputs: tf.Tensor, cutoff: float):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    def _compute_gaussian_expansion(self, inputs: tf.Tensor):
        rij, zi = inputs
        params = tf.gather(self._weight_eta_mu_lambda_zeta, zi, axis=0)
        eta, mu = tf.gather(params, 0, axis=-1), tf.gather(params, 1, axis=-1)
        arg = tf.square(rij - mu) * eta
        return tf.exp(-arg)

    def _compute_pow_cos_angle_(self, inputs: list):
        vij, vik, rij, rik, zi = inputs
        params = tf.gather(self._weight_eta_mu_lambda_zeta, zi, axis=0)
        lamda, zeta = tf.gather(params, 2, axis=-1), tf.gather(params, 3, axis=-1)
        cos_theta = tf.reduce_sum(vij * vik, axis=-1, keepdims=True) / rij / rik
        cos_term = cos_theta * lamda + 1.0
        cos_term = tf.pow(cos_term, zeta)
        return cos_term

    def _compute_pow_scale(self, inputs: tf.Tensor):
        to_scale, zi = inputs
        params = tf.gather(self._weight_eta_mu_lambda_zeta, zi, axis=0)
        zeta = tf.gather(params, 3, axis=-1)
        scale = tf.ones_like(to_scale) * 2.0
        scaled = tf.pow(scale, 1.0 - zeta) * to_scale
        return scaled

    def build(self, input_shape):
        super(wACSFAng, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [z, xyz, ijk] or [z, xyz, ijk, w]  if `use_external_weights`

                - z (tf.RaggedTensor): Atomic numbers of shape (batch, [N])
                - xyz (tf.RaggedTensor): Node coordinates of shape (batch, [N], 3)
                - ijk (tf.RaggedTensor): Angle indices referring to nodes of shape (batch, [M], 3)
                - w (tf.RaggedTensor): Angle weight tensor of shape (batch, [M], 1)

            mask (list): Boolean mask for inputs. Not used. Defaults to None.

        Returns:
            tf.RaggedTensor: Atomic representation of shape `(batch, None, units)` .
        """
        if self.use_external_weights:
            z, xyz, ijk, w = self.assert_ragged_input_rank(inputs, mask=mask, ragged_rank=1)
            z = self.map_values(tf.cast, z, dtype=ijk.dtype)
            w = self.map_values(tf.cast, w, dtype=self.dtype)
        else:
            z, xyz, ijk = self.assert_ragged_input_rank(inputs, mask=mask, ragged_rank=1)
            z = self.map_values(tf.cast, z, dtype=ijk.dtype)
            w1 = self.layer_gather_1([z, ijk], **kwargs)[0]
            w2 = self.layer_gather_2([z, ijk], **kwargs)[0]
            w1 = self.map_values(tf.cast, w1, dtype=self.dtype)
            w2 = self.map_values(tf.cast, w2, dtype=self.dtype)
            w = self.lazy_mult([w1, w2], **kwargs)
            w = self.layer_exp_dims(w, **kwargs)
        xi, xj, xk = self.layer_pos([xyz, ijk])
        rij = self.layer_dist([xi, xj], **kwargs)
        rik = self.layer_dist([xi, xk], **kwargs)
        rjk = self.layer_dist([xj, xk], **kwargs)
        fij = self.map_values(self._compute_fc, rij, cutoff=self.cutoff)
        fik = self.map_values(self._compute_fc, rik, cutoff=self.cutoff)
        fjk = self.map_values(self._compute_fc, rjk, cutoff=self.cutoff)
        zi = self.layer_gather_in([z, ijk], **kwargs)
        gij = self.map_values(self._compute_gaussian_expansion, [rij, zi])
        gik = self.map_values(self._compute_gaussian_expansion, [rik, zi])
        gjk = self.map_values(self._compute_gaussian_expansion, [rjk, zi])
        vij = self.lazy_sub([xi, xj], **kwargs)
        vik = self.lazy_sub([xi, xk], **kwargs)
        pow_cos_theta = self.map_values(self._compute_pow_cos_angle_, [vij, vik, rij, rik, zi])
        rep = self.lazy_mult([pow_cos_theta, gij, gik, gjk, fij, fik, fjk, w], **kwargs)
        pool_ang = self.pool_sum([xyz, rep, ijk], **kwargs)
        return self.map_values(self._compute_pow_scale, [pool_ang, z])

    def get_config(self):
        config = super(wACSFAng, self).get_config()
        config.update({"eta_mu_lambda_zeta": self.eta_mu_lambda_zeta, "cutoff": self.cutoff,
                       "use_external_weights": self.use_external_weights,
                       "add_eps": self.add_eps,
                       "param_constraint": ks.constraints.serialize(self.param_constraint),
                       "param_regularizer": ks.regularizers.serialize(self.param_regularizer),
                       "param_initializer": ks.initializers.serialize(self.param_initializer)
                       })
        return config
