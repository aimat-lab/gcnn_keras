import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesSelection, GatherNodesIngoing
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazyMultiply, LazySubtract, ExpandDims

ks = tf.keras

# We set the unoptimized C default values as default for all 118 atomic species.
# They are taken from https://aip.scitation.org/doi/suppl/10.1063/1.5019667/suppl_file/si.pdf .
# We chose the 22:10 radial as default values. Other combinations ranging from 0:32 and 32:0 .
radial_eta_mu_defaults = np.array(
    [[[4.5000000, 7.5000000], [4.5000000, 7.1667000], [4.5000000, 6.8333000], [4.5000000, 6.5000000],
      [4.5000000, 6.1667000], [4.5000000, 5.8333000], [4.5000000, 5.5000000], [4.5000000, 5.1667000],
      [4.5000000, 4.8333000], [4.5000000, 4.5000000], [4.5000000, 4.1667000], [4.5000000, 3.8333000],
      [4.5000000, 3.5000000], [4.5000000, 3.1667000], [4.5000000, 2.8333000], [4.5000000, 2.5000000],
      [4.5000000, 2.1667000], [4.5000000, 1.8333000], [4.5000000, 1.5000000], [4.5000000, 1.1667000],
      [4.5000000, 0.8333000], [4.5000000, 0.5000000]]] * 118)
# Update optimized parameters for C, H, F, O, N .
radial_eta_mu_defaults[6] = np.array(
    [[4.4534562, 3.8338779], [4.4804536, 5.1667000], [4.4918531, 4.8333000], [4.4991551, 0.8333000],
     [4.4998245, 2.4997843], [4.4999043, 1.5993354], [4.4999230, 5.8324479], [4.4999997, 6.5117668],
     [4.5000000, 6.1860418], [4.5000000, 3.1584848], [4.5000001, 4.4991770], [4.5000132, 6.8333000],
     [4.5001099, 7.4999458], [4.5002911, 2.1638163], [4.5004121, 0.3038147], [4.5014183, 5.4361520],
     [4.5023686, 1.8302079], [4.5027053, 3.5002020], [4.5074706, 7.2656992], [4.5104731, 1.1645345],
     [4.5515436, 4.1666210], [4.5906934, 2.8328539]])
radial_eta_mu_defaults[1] = np.array(
    [[4.3814240, 2.4432533], [4.4149216, 6.1667000], [4.4493159, 3.8332928], [4.4815889, 3.1667000],
     [4.4865336, 3.5000000], [4.4937252, 4.8332601], [4.4986009, 5.5010689], [4.4997975, 2.1558150],
     [4.4999291, 7.4956238], [4.4999905, 6.4964721], [4.5000000, 5.2730567], [4.5000011, 1.1606611],
     [4.5000039, 0.4918029], [4.5000398, 0.8333002], [4.5002241, 1.8334858], [4.5036121, 2.9532907],
     [4.5059680, 5.8330954], [4.5062245, 6.8333000], [4.5064492, 7.1666408], [4.5675834, 4.5046412],
     [4.5899630, 4.1663934], [4.6604619, 1.4989046]])
radial_eta_mu_defaults[9] = np.array(
    [[4.3755129, 4.3891938], [4.4739340, 6.5000000], [4.4934064, 1.1667000], [4.4955168, 3.1421155],
     [4.4982145, 6.1665305], [4.4986470, 0.8333000], [4.4989965, 7.4999930], [4.4999944, 4.1666929],
     [4.4999998, 6.8333000], [4.4999998, 2.8332546], [4.4999999, 5.8424471], [4.5000000, 5.4954750],
     [4.5000000, 4.8332314], [4.5000000, 2.1675890], [4.5000000, 1.8332281], [4.5000016, 3.8334246],
     [4.5000047, 7.1667013], [4.5002559, 0.5866000], [4.5004160, 5.1667000], [4.5165951, 1.5000000],
     [4.5200686, 3.5000000], [4.5333816, 2.5065631]])
radial_eta_mu_defaults[8] = np.array(
    [[4.3810310, 6.1634839], [4.4613793, 4.7025864], [4.4708866, 0.8337341], [4.4741510, 7.1654091],
     [4.4934208, 0.4909963], [4.4999979, 6.7895512], [4.4999984, 2.5548726], [4.4999996, 5.1188661],
     [4.4999999, 6.4941667], [4.5000000, 7.5269793], [4.5000000, 3.5008886], [4.5000000, 3.1673548],
     [4.5000000, 3.0664616], [4.5000000, 1.1771551], [4.5000010, 5.9628537], [4.5000305, 1.6101757],
     [4.5000505, 3.8352953], [4.5000877, 5.2259600], [4.5003116, 2.1667069], [4.5007395, 1.8264699],
     [4.5037122, 4.1666850], [4.5108571, 4.7474998]])
radial_eta_mu_defaults[7] = np.array(
    [[4.4148209, 6.8472192], [4.4481457, 1.0317668], [4.4608631, 0.4581865], [4.4797608, 3.9799903],
     [4.4951823, 0.8337448], [4.4987030, 7.4573738], [4.4998898, 6.9779219], [4.4999884, 2.1676806],
     [4.4999920, 3.1667000], [4.5000000, 6.1662980], [4.5000000, 5.8328781], [4.5000000, 5.4757644],
     [4.5000000, 5.1666981], [4.5000000, 3.3518840], [4.5000000, 2.4982784], [4.5001975, 4.8204812],
     [4.5004709, 6.5050056], [4.5016337, 4.1667000], [4.5077984, 1.4994658], [4.5228543, 2.8333000],
     [4.5780852, 4.5000000], [4.5897686, 1.8340302]])


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFRad')
class wACSFRad(GraphBaseLayer):
    r"""Weighted atom-centered symmetry functions (wACSF) for high-dimensional neural network potentials (HDNNPs).
    From `Gastegger et al. (2017) <https://arxiv.org/abs/1712.05861>`_ .
    Default values can be found in `<https://aip.scitation.org/doi/suppl/10.1063/1.5019667>`_ .
    This layer implements the radial part :math:`W_{i}^{rad}` :

    .. math::

        W_{i}^{rad} = \sum_{j \not i} \; g(Z_j) \; e^{−\eta \, (r_{ij} − \mu)^{2} } \; f_{ij}

    Here, for each atom type there is a set of parameters :math:`\eta` and :math:`\mu` .
    The cutoff function :math:`f_ij = f_c(R_{ij})` is given by:

    .. math::

        f_c(R_{ij}) = 0.5 [\cos{\frac{\pi R_{ij}}{R_c}} + 1]

    The cutoff radius is implemented as a float and can not be changed dependent on the atom type.
    Not that the parameters :math:`\eta` and :math:`\mu` of can be made trainable.

    """

    def __init__(self, eta_mu: list = None, cutoff: float = 8.0, add_eps: bool = False,
                 use_external_weights: bool = False, param_constraint=None, param_regularizer=None,
                 param_initializer="zeros", **kwargs):
        super(wACSFRad, self).__init__(**kwargs)
        self.cutoff = cutoff
        self.add_eps = add_eps
        if eta_mu is None:
            eta_mu = radial_eta_mu_defaults
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
        self.param_initializer = param_initializer
        self.param_regularizer = param_regularizer
        self.param_constraint = param_constraint
        weight_shape = np.array(eta_mu).shape
        self._weight_eta_mu = self.add_weight(
            "eta_mu",
            shape=weight_shape,
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=False
        )
        self.set_weights([np.array(eta_mu)])

    @staticmethod
    def _compute_fc(inputs: tf.Tensor, cutoff: float):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    def _compute_gaussian_expansion(self, inputs: tf.Tensor):
        rij, zi = inputs
        params = tf.gather(self._weight_eta_mu, zi, axis=0)
        eta, mu = tf.split(params, [1, 1], axis=-1)
        arg = tf.square(inputs - mu) * eta
        return tf.exp(-arg)

    def build(self, input_shape):
        super(wACSFRad, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.use_external_weights:
            z, xyz, eij, w = inputs
        else:
            z, xyz, eij = inputs
            zj = self.layer_gather_out([z, eij], **kwargs)
            w = self.layer_exp_dims(zj, **kwargs)
        xi, xj = self.layer_pos([xyz, eij], **kwargs)
        rij = self.layer_dist([xi, xj], **kwargs)
        fc = self.call_on_values_tensor_of_ragged(self._compute_fc, rij, cutoff=self.cutoff)
        zi = self.layer_gather_in([z, eij], **kwargs)
        gij = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, [rij, zi])
        rep = self.lazy_mult([w, gij, fc], **kwargs)
        return self.pool_sum([xyz, eij, rep], **kwargs)

    def get_config(self):
        config = super(wACSFRad, self).get_config()
        config.update({"eta_mu": self.eta_mu, "cutoff": self.cutoff, "use_external_weights": self.use_external_weights,
                       "add_eps": self.add_eps,
                       "param_constraint": self.param_constraint,
                       "param_regularizer": self.param_regularizer,
                       "param_initializer": self.param_initializer
                       })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFAng')
class wACSFAng(GraphBaseLayer):

    def __init__(self, eta_mu_zeta_lambda: list = None, cutoff: float = 8.0,
                 add_eps: bool = False, use_external_weights: bool = False, **kwargs):
        super(wACSFAng, self).__init__(**kwargs)
        self.add_eps = add_eps
        self.cutoff = cutoff
        self.eta_mu_zeta_lambda = eta_mu_zeta_lambda
        self.use_external_weights = use_external_weights
        self.lazy_mult = LazyMultiply()
        self.layer_pos = NodePosition(selection_index=[0, 1, 2])
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = PoolingLocalEdges(pooling_method="sum")
        self.lazy_sub = LazySubtract()
        self.layer_gather_1 = GatherNodesSelection(selection_index=1)
        self.layer_gather_2 = GatherNodesSelection(selection_index=2)
        self.layer_exp_dims = ExpandDims(axis=2)

    @staticmethod
    def _compute_fc(inputs: tf.Tensor, cutoff: float):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    @staticmethod
    def _compute_gaussian_expansion(inputs: tf.Tensor, eta: tf.Tensor, mu: tf.Tensor):
        arg = tf.square(inputs - mu) * eta
        return tf.exp(-arg)

    @staticmethod
    def _compute_pow_cos_angle_(inputs: list, zeta: tf.Tensor, lamda: tf.Tensor):
        vij, vik, rij, rik = inputs
        cos_theta = tf.reduce_sum(vij * vik, axis=-1, keepdims=True) / rij / rik
        cos_term = cos_theta * lamda + 1.0
        cos_term = tf.pow(cos_term, tf.expand_dims(zeta, axis=0))
        return cos_term

    @staticmethod
    def _compute_pow_scale(inputs: tf.Tensor, zeta: tf.Tensor):
        scale = tf.ones_like(inputs) * 2.0
        return tf.pow(scale, 1.0 - tf.expand_dims(zeta, axis=0)) * inputs

    def build(self, input_shape):
        super(wACSFAng, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.use_external_weights:
            z, xyz, ijk, w = inputs
        else:
            z, xyz, ijk = inputs
            w1 = self.layer_gather_1([z, ijk], **kwargs)
            w2 = self.layer_gather_2([z, ijk], **kwargs)
            w = self.lazy_mult([w1, w2], **kwargs)
        xi, xj, xk = self.layer_pos([xyz, ijk])
        rij = self.layer_dist([xi, xj])
        rik = self.layer_dist([xi, xk])
        rjk = self.layer_dist([xj, xk])
        fij = self.call_on_values_tensor_of_ragged(self._compute_fc, rij, cutoff=self.cutoff)
        fik = self.call_on_values_tensor_of_ragged(self._compute_fc, rik, cutoff=self.cutoff)
        fjk = self.call_on_values_tensor_of_ragged(self._compute_fc, rjk, cutoff=self.cutoff)
        gij = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rij, eta=self.eta_mu_zeta_lambda[0], mu=self.eta_mu_zeta_lambda[1])
        gik = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rik, eta=self.eta_mu_zeta_lambda[0], mu=self.eta_mu_zeta_lambda[1])
        gjk = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rjk, eta=self.eta_mu_zeta_lambda[0], mu=self.eta_mu_zeta_lambda[1])
        vij = self.lazy_sub([xi, xj])
        vik = self.lazy_sub([xi, xk])
        pow_cos_theta = self.call_on_values_tensor_of_ragged(
            self._compute_pow_cos_angle_, [vij, vik, rij, rik], zeta=self.eta_mu_zeta_lambda[2],
            lamda=self.eta_mu_zeta_lambda[3])
        rep = self.lazy_mult([w, pow_cos_theta, gij, gik, gjk, fij, fik, fjk])
        pool_ang = self.pool_sum([xyz, ijk, rep], **kwargs)
        out = self.call_on_values_tensor_of_ragged(
            self._compute_pow_scale, pool_ang, zeta=self.self.eta_mu_zeta_lambda[2])
        return out

    def get_config(self):
        config = super(wACSFAng, self).get_config()
        config.update({"cutoff": self.cutoff})
        return config
