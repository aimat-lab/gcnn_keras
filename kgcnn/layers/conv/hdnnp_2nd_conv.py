import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesSelection, GatherNodesIngoing
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazyMultiply, LazySubtract, ExpandDims
ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFRad')
class wACSFRad(GraphBaseLayer):
    r"""Weighted atom-centered symmetry functions (wACSF) for high-dimensional neural network potentials (HDNNPs).
    From `Gastegger et al. (2017) <https://arxiv.org/abs/1712.05861>`_ .
    This is the radial part :math:`W_{i}^{rad}` :



    """

    def __init__(self, eta_mu: list, cutoff: float = 8.0, add_eps: bool = False,
                 use_external_weights: bool = False, **kwargs):
        super(wACSFRad, self).__init__(**kwargs)
        self.cutoff = cutoff
        self.add_eps = add_eps
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
        self._weight_eta_mu = self.add_weight(
            "eta_mu",
            shape=np.array(eta_mu).shape,
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=False
        )

    @staticmethod
    def _compute_fc(inputs: tf.Tensor, cutoff: float):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    def _compute_gaussian_expansion(self, inputs: tf.Tensor):
        rij, zi = inputs
        params = tf.gather(self._weight_eta_mu, zi, axis=0)
        eta, mu = tf.split(params, [1, 1], axis=1)
        arg = tf.square(inputs-mu)*eta
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
        config.update({"cutoff": self.cutoff, "eta": self.eta, "mu": self.mu, "add_eps": self.add_eps})
        return config






@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFAng')
class wACSFAng(GraphBaseLayer):

    def __init__(self, eta: list = None, mu: list = None, lamda: list = None, zeta: list = None, cutoff: float = 8.0,
                 add_eps: bool = False, use_external_weights: bool = False, **kwargs):
        super(wACSFAng, self).__init__(**kwargs)
        self.add_eps = add_eps
        self.cutoff = cutoff
        self.eta = eta
        self.mu = mu
        self.zeta = zeta
        self.lamda = lamda
        eta_mu_zeta_lambda = [[e, m, z, l] for e in eta for m in mu for z in zeta for l in lamda]
        self.eta_mu_zeta_lambda = [tf.constant(x, dtype=self.dtype) for x in np.transpose(eta_mu_zeta_lambda)]
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
        arg = tf.square(inputs-mu)*eta
        return tf.exp(-arg)

    @staticmethod
    def _compute_pow_cos_angle_(inputs: list, zeta: tf.Tensor, lamda: tf.Tensor):
        vij, vik, rij, rik = inputs
        cos_theta = tf.reduce_sum(vij*vik, axis=-1, keepdims=True)/rij/rik
        cos_term = cos_theta*lamda + 1.0
        cos_term = tf.pow(cos_term, tf.expand_dims(zeta, axis=0))
        return cos_term

    @staticmethod
    def _compute_pow_scale(inputs: tf.Tensor, zeta: tf.Tensor):
        scale = tf.ones_like(inputs)*2.0
        return tf.pow(scale, 1.0 - tf.expand_dims(zeta, axis=0))*inputs

    def build(self, input_shape):
        super(wACSFAng, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.use_external_weights:
            z, xyz, ijk, w = inputs
        else:
            z, xyz, ijk = inputs
            w1 = self.layer_gather_1([z, ijk],**kwargs)
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
