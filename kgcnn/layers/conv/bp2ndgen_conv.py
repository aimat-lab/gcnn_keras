import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition, VectorAngle
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazyMultiply, LazySubtract
ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFRad')
class wACSFRad(GraphBaseLayer):

    def __init__(self, eta: list = None, mu: list = None, cutoff: float = 5.0, add_eps: bool = False, **kwargs):
        super(wACSFRad, self).__init__(**kwargs)
        self.cutoff = cutoff
        self.add_eps = add_eps
        self.eta = eta
        self.mu = mu
        self.eta_mu = tf.constant()  # TODO

        self.lazy_mult = LazyMultiply()
        self.layer_pos = NodePosition()
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = PoolingLocalEdges(pooling_method="sum")

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

    def build(self, input_shape):
        super(wACSFRad, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        z, xyz, eij, w = inputs
        xi, xj = self.layer_pos([xyz, eij], **kwargs)
        rij = self.layer_dist([xi, xj], **kwargs)
        fc = self.call_on_values_tensor_of_ragged(self._compute_fc, rij, cutoff=self.cutoff)
        gij = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rij, eta=self.eta_mu[0], mu=self.eta_mu[1])
        rep = self.lazy_mult([w, gij, fc], **kwargs)
        return self.pool_sum([xyz, eij, rep], **kwargs)

    def get_config(self):
        config = super(wACSFRad, self).get_config()
        config.update({"cutoff": self.cutoff, "eta": self.eta, "mu": self.mu, "add_eps": self.add_eps})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFAng')
class wACSFAng(GraphBaseLayer):

    def __init__(self, zeta: list = None, mu: list = None, lamda: list = None, cutoff: float = 5,
                 add_eps: bool = False, **kwargs):
        super(wACSFAng, self).__init__(**kwargs)
        self.add_eps = add_eps
        self.cutoff = cutoff
        self.eta = zeta
        self.mu = mu
        self.lamda = lamda
        self.eta_mu_lambda = tf.constant()  # TODO

        self.lazy_mult = LazyMultiply()
        self.layer_pos = NodePosition(selection_index=[0, 1, 2])
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = PoolingLocalEdges(pooling_method="sum")
        self.lazy_sub = LazySubtract()

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

    def build(self, input_shape):
        super(wACSFAng, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        z, xyz, ijk, w = inputs
        xi, xj, xk = self.layer_pos([xyz, ijk])
        rij = self.layer_dist([xi, xj])
        rik = self.layer_dist([xi, xk])
        rjk = self.layer_dist([xj, xk])
        fij = self.call_on_values_tensor_of_ragged(self._compute_fc, rij, cutoff=self.cutoff)
        fik = self.call_on_values_tensor_of_ragged(self._compute_fc, rik, cutoff=self.cutoff)
        fjk = self.call_on_values_tensor_of_ragged(self._compute_fc, rjk, cutoff=self.cutoff)
        gij = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rij, eta=self.eta_mu_lambda[0], mu=self.eta_mu_lambda[1])
        gik = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rik, eta=self.eta_mu_lambda[0], mu=self.eta_mu_lambda[1])
        gjk = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rjk, eta=self.eta_mu_lambda[0], mu=self.eta_mu_lambda[1])
        vij = self.lazy_sub([xi, xj])
        vik = self.lazy_sub([xi, xk])
        cos_theta = None
        rep = self.lazy_mult([gij, gik, gjk, fij, fik, fjk])
        return self.pool_sum([xyz, ijk, rep], **kwargs)

    def get_config(self):
        config = super(wACSFAng, self).get_config()
        config.update({"cutoff": self.cutoff})
        return config
