from typing import Optional
import math
import tensorflow as tf
import numpy as np
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.literature.coGN._utils import enabled_ragged


class EdgeEmbedding(GraphBaseLayer):
    """Emebedding layer for edges of crystal graphs."""

    def __init__(
        self,
        bins_distance: int = 32,
        max_distance: float = 5.0,
        distance_log_base: float = 1.0,
        gaussian_width_distance: float = 1.0,
        bins_voronoi_area: Optional[int] = None,
        max_voronoi_area: Optional[float] = 32.0,
        voronoi_area_log_base: float = 1.0,
        gaussian_width_voronoi_area: float = 1.0,
        **kwargs
    ):
        """Initialized EdgeEmbedding layer.

        Args:
            bins_distance (int, optional): How many dimensions the gaussian embedding for distances has. Defaults to 32.
            max_distance (float, optional): Cutoff value for the gaussian embedding on distances.
                Gaussians will be spaced on the [0, max_distance] interval.
                Defaults to 5..
            distance_log_base (float, optional): If this value is 1.0, the gaussians will be evenly spaced on the interval.
                Otherwise a log scale with a logarithmic scaling with this basis is applied to the spacing.
                Can be used to scew information density to low/high values.
                Defaults to 1.0.
            gaussian_width_distance (float, optional): Factor for the sigma value of the gaussians. Defaults to 1..
            bins_voronoi_area (Optional[int], optional): How many dimensions the gaussian embedding for Voronoi areas has.
                Defaults to None.
            max_voronoi_area (Optional[float], optional): Cutoff value for the gaussian embedding on Voronoi areas.
                Gaussians will be spaced on the [0, max_voronoi_area] interval.
                Defaults to 32.
            voronoi_area_log_base (float, optional): If this value is 1.0, the gaussians will be evenly spaced on the interval.
                Otherwise a log scale with a logarithmic scaling with this basis is applied to the spacing.
                Can be used to scew information density to low/high values.. Defaults to 1..
            gaussian_width_voronoi_area (float, optional): Factor for the sigma value of the gaussians. Defaults to 1..
        """
        super().__init__(**kwargs)
        if distance_log_base == 1.0:
            self.distance_embedding = GaussBasisExpansion.from_bounds(
                bins_distance, 0.0, max_distance, variance=gaussian_width_distance
            )
        else:
            self.distance_embedding = GaussBasisExpansion.from_bounds_log(
                bins_distance,
                0.0,
                max_distance,
                base=distance_log_base,
                variance=gaussian_width_distance,
            )

        self.distance_log_base = distance_log_base
        self.log_max_distance = np.log(max_distance)
        if bins_voronoi_area is not None and bins_voronoi_area > 0:
            if max_voronoi_area is None:
                raise ValueError("Max voronoi area must not be None.")
            if voronoi_area_log_base == 1:
                self.voronoi_area_embedding = GaussBasisExpansion.from_bounds(
                    bins_voronoi_area,
                    0.0,
                    max_voronoi_area,
                    variance=gaussian_width_voronoi_area,
                )
            else:
                self.voronoi_area_embedding = GaussBasisExpansion.from_bounds_log(
                    bins_voronoi_area,
                    0.0,
                    max_voronoi_area,
                    base=voronoi_area_log_base,
                    variance=gaussian_width_voronoi_area,
                )

    def call(self, inputs):

        if isinstance(inputs, (list, tuple)):
            distance = inputs[0]
            voronoi_area = inputs[1]
        else:
            distance = inputs
            voronoi_area = None

        d = tf.expand_dims(distance, -1)
        distance_embedded = self.distance_embedding(d)

        if voronoi_area is not None:
            v = tf.expand_dims(voronoi_area, -1)
            voronoi_area_embedded = self.voronoi_area_embedding(v)
            edge_embedded = tf.concat(
                [distance_embedded, voronoi_area_embedded], axis=-1
            )
        else:
            edge_embedded = distance_embedded

        return edge_embedded


class SinCosExpansion(GraphBaseLayer):
    def __init__(self, dim=10, wave_length=math.pi, base=2):
        assert dim % 2 == 0, "dim has to be a multiple of 2."
        super().__init__()
        self.d = dim
        self.wave_length = float(wave_length) / math.pi
        self.frequencies = (
            tf.cast(tf.pow(base, tf.range(self.d / 2)), float) * self.wave_length
        )

    @enabled_ragged
    def call(self, inputs):
        values_x_freq = tf.expand_dims(inputs, -1) * self.frequencies
        sines = tf.sin(values_x_freq)
        cosines = tf.cos(values_x_freq)
        return tf.concat([sines, cosines], axis=-1)


class GaussBasisExpansion(GraphBaseLayer):
    """Gauss Basis expansion layer"""

    def __init__(self, mu, sigma, **kwargs):
        super().__init__(**kwargs)
        # shape: (1, len(mu))
        self.mu = tf.expand_dims(tf.constant(mu, float), 0)
        # shape: (1, len(sigma))
        self.sigma = tf.expand_dims(tf.constant(sigma, float), 0)

    @classmethod
    def from_bounds(cls, n: int, low: float, high: float, variance: float = 1.0):
        mus = np.linspace(low, high, num=n + 1)
        var = np.diff(mus)
        mus = mus[1:]
        return cls(mus, np.sqrt(var * variance))

    @classmethod
    def from_bounds_log(
        cls, n: int, low: float, high: float, base: float = 32, variance: float = 1
    ):
        mus = (np.logspace(0, 1, num=n + 1, base=base) - 1) / (base - 1) * (
            high - low
        ) + low
        var = np.diff(mus)
        mus = mus[1:]
        return cls(mus, np.sqrt(var * variance))

    @enabled_ragged
    def call(self, x, **kwargs):
        return tf.exp(-tf.pow(x - self.mu, 2) / (2 * tf.pow(self.sigma, 2)))

    def plot_gaussians(self, low: float, high: float, n=1000):
        from matplotlib import pyplot as plt

        x = tf.range(0, n) / (n - 1) * (high - low) + low
        out = self(x)
        for i in range(out.shape[1]):
            plt.plot(x, out[:, i])
        return plt

    def get_config(self):
        """Update config."""
        config = super().get_config()
        config.update(
            {"mu": self.mu[0].numpy().tolist(), "sigma": self.sigma[0].numpy().tolist()}
        )
        return config
