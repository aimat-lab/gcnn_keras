import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazyMultiply
ks = tf.keras


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFRad')
class wACSFRad(GraphBaseLayer):

    def __init__(self, eta: list = None, mu: list = None, cutoff: float = 5, **kwargs):
        super(wACSFRad, self).__init__(**kwargs)
        self.lazy_mult = LazyMultiply()
        self.layer_pos = NodePosition()
        self.layer_dist = NodeDistanceEuclidean()
        self.pool_sum = PoolingLocalEdges(pooling_method="sum")
        self.cutoff = cutoff
        self.eta = eta
        self.mu = mu
        self.eta_mu = tf.constant()

    @staticmethod
    def _compute_fc(inputs: tf.Tensor, cutoff: float):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    @staticmethod
    def _compute_gaussian_expansion(distance: tf.Tensor, eta: tf.Tensor, mu: tf.Tensor):
        arg = tf.square(distance-mu)*eta
        return tf.exp(-arg)

    def build(self, input_shape):
        super(wACSFRad, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        xyz, eij, w = inputs
        xi, xj = self.layer_pos([xyz, eij], **kwargs)
        rij = self.layer_dist([xi, xj], **kwargs)
        fc = self.call_on_values_tensor_of_ragged(self._compute_fc, rij, cutoff=self.cutoff)
        gbasis = self.call_on_values_tensor_of_ragged(
            self._compute_gaussian_expansion, rij, eta=self.eta_mu[0], mu=self.eta_mu[1])
        rep = self.lazy_mult([w, gbasis, fc], **kwargs)
        return self.pool_sum([xyz, eij, rep], **kwargs)

    def get_config(self):
        config = super(wACSFRad, self).get_config()
        config.update({"cutoff": self.cutoff, "eta": self.eta, "mu": self.mu})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='wACSFAng')
class wACSFAng(GraphBaseLayer):

    def __init__(self, zeta: list = None, mu: list = None, lamda: list = None, cutoff: float = 5, **kwargs):
        super(wACSFAng, self).__init__(**kwargs)
        self.lazy_mult = LazyMultiply()
        self.pool_sum = PoolingLocalEdges(pooling_method="sum")
        self.cutoff = cutoff
        self.eta = zeta
        self.mu = mu
        self.lamda = lamda
        self.eta_mu_lambda = tf.constant()

    @staticmethod
    def _compute_fc(inputs: tf.Tensor, cutoff: float):
        fc = tf.clip_by_value(inputs, -cutoff, cutoff)
        fc = (tf.math.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    @staticmethod
    def _compute_gaussian_expansion(distance: tf.Tensor, eta: tf.Tensor, mu: tf.Tensor):
        arg = tf.square(distance-mu)*eta
        return tf.exp(-arg)

    def build(self, input_shape):
        super(wACSFAng, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        n, theta, ijk, rij, rik, rjk, w = inputs
        rep = None
        return self.pool_sum([n, ijk, rep], **kwargs)

    def get_config(self):
        config = super(wACSFAng, self).get_config()
        config.update({"cutoff": self.cutoff})
        return config
