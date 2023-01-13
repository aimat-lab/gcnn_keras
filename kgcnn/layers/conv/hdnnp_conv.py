import tensorflow as tf
import numpy as np
import math
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.casting import ChangeTensorType

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='CENTCharge')
class CENTCharge(GraphBaseLayer):
    """Compute charge equilibration according to
    `Ko et al. (2021) <https://www.nature.com/articles/s41467-020-20427-2>`_ .

    Example:

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.conv.hdnnp_conv import CENTCharge
        layer = CENTCharge()
        z = tf.ragged.constant([[1, 6], [1, 1, 6]], ragged_rank=1)
        chi = tf.ragged.constant([[0.43, 0.37], [0.43, 0.43, 0.37]], ragged_rank=1)
        xyz = tf.ragged.constant([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.2, 0.2], [2.0, 0.0, 0.0]]],
                                 ragged_rank=1, inner_shape=(3,))
        qtot = tf.constant([1.0, 1.0])
        q = layer([z, chi, xyz, qtot])
        print(q)

    """

    # From https://en.wikipedia.org/wiki/Covalent_radius with radii in pm.
    _default_radii = 0.01 * np.array([
        0.0, 31, 28,
        128, 96, 84, 73, 71, 66, 57, 58,
        166, 141, 121, 111, 107, 105, 102, 106,
        203, 176, 170, 160, 153, 139, 139, 132, 126, 124, 132, 122, 122, 120, 119, 120, 120, 116,
        220, 195, 190, 175, 164, 154, 147, 146, 142, 139, 145, 144, 142, 139, 139, 138, 139, 140
    ])
    # Chemical hardness from https://www.pnas.org/doi/10.1073/pnas.2117416119 in eV
    _default_hardness = 0.037 / 0.529177 * np.array([
        0.0, 6.2, 8.8,
        2.2, 4.6, 3.8, 4.7, 7.1, 5.6, 6.1, 9.1,
        2.1, 4.0, 2.6, 3.3, 4.7, 3.8, 4.5, 7.7,
        2.3, 3.2, 3.2, 2.9, 3.2, 3.4, 4.0, 3.6, 3.3, 3.3, 3.8, 5.8, 3.0, 3.3, 4.5, 3.9, 4.2, 7.7,
        1.9, 3.1, 3.1, 2.9, 3.3, 3.5, 3.7, 3.7, 3.9, 4.1, 3.6, 5.4, 3.1, 3.1, 4.0, 3.6, 3.8, 6.8
    ])
    _max_atomic_number = 55

    def __init__(self, output_to_tensor: bool = False,
                 param_constraint=None, param_regularizer=None, param_initializer="glorot_uniform",
                 param_trainable: bool = False,
                 **kwargs):
        super(CENTCharge, self).__init__(**kwargs)
        self.output_to_tensor = output_to_tensor

        self.layer_cast_n = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask", boolean_mask=True)
        self.layer_cast_chi = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask")
        self.layer_cast_x = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask", boolean_mask=True)

        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = param_initializer
        self.param_regularizer = param_regularizer
        self.param_constraint = param_constraint
        self.param_trainable = param_trainable

        self.weight_j = self.add_weight(
            "hardness_j",
            shape=(self._max_atomic_number,),
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=self.param_trainable
        )
        self.weight_sigma = self.add_weight(
            "sigma",
            shape=(self._max_atomic_number,),
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=self.param_trainable
        )
        self.set_weights([self._default_hardness, self._default_radii])

    def build(self, input_shape):
        super(CENTCharge, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        n, chi, x = self.assert_ragged_input_rank(inputs[:3], ragged_rank=1)

        qtot = inputs[3]
        num_atoms = n.row_lengths()

        # Cast to padded tensor for charge equilibrium.
        # Only keep mask for atomic number.
        n_pad, n_mask = self.layer_cast_n(n, **kwargs)
        chi_pad, _ = self.layer_cast_chi(chi, **kwargs)
        x_pad, _ = self.layer_cast_x(x, **kwargs)

        # Compute mask values for diagonal and off-diagonal.
        a_mask = tf.logical_and(tf.expand_dims(n_mask, axis=1), tf.expand_dims(n_mask, axis=2))
        a_diag_mask = tf.linalg.diag(n_mask)
        a_off_mask = tf.logical_and(a_mask, tf.logical_not(a_diag_mask))

        # Compute distance and gamma matrix elements.
        diff = tf.expand_dims(x_pad, axis=1) - tf.expand_dims(x_pad, axis=2)
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1, keepdims=False))  # (batch, N, N)
        gamma_ij = tf.sqrt(tf.expand_dims(
            tf.square(tf.gather(self.weight_sigma, n_pad, axis=0)), axis=1) + tf.expand_dims(
            tf.square(tf.gather(self.weight_sigma, n_pad, axis=0)), axis=2))  # (batch, N, N)

        # Compute diagonal matrix elements.
        J_i = tf.gather(self.weight_j, n_pad, axis=0)  # (batch, N)
        sigma_i = tf.gather(self.weight_sigma, n_pad, axis=0)  # (batch, N)
        diag_part = tf.linalg.diag(J_i + 1.0 / sigma_i / tf.sqrt(math.pi))

        # Setup Matrix A.
        a = tf.where(
            a_off_mask,
            tf.math.divide_no_nan(
                tf.math.erf(tf.math.divide_no_nan(dist, gamma_ij * tf.sqrt(2.0))),
                dist
            ),
            tf.zeros_like(dist)
        ) + tf.where(
            a_diag_mask,
            diag_part,
            tf.zeros_like(dist)
        )

        # Pad A and chi for Lagrange multipliers.
        a = tf.pad(a, [[0, 0], [0, 1], [0, 1]], mode='CONSTANT', constant_values=0)
        chi_pad = tf.pad(chi_pad, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=0)

        # Add total charge and Lagrange multipliers to A.
        qtot_indices = tf.cast(tf.concat([
            tf.expand_dims(tf.range(tf.shape(chi_pad)[0], dtype=num_atoms.dtype), axis=-1),
            tf.expand_dims(num_atoms, axis=-1)
        ], axis=-1), dtype="int32")
        chi_pad = tf.tensor_scatter_nd_add(
            chi_pad,
            qtot_indices,
            qtot
        )
        idx = [tf.cast(tf.expand_dims(iter_tensor, axis=-1), dtype="int32") for iter_tensor in [
            tf.repeat(tf.range(tf.shape(num_atoms)[0]), num_atoms), tf.ragged.range(num_atoms).flat_values,
            tf.repeat(num_atoms, num_atoms)]]
        a_1 = tf.concat([tf.concat([idx[0], idx[1], idx[2]], axis=-1),
                         tf.concat([idx[0], idx[2], idx[1]], axis=-1)], axis=0)
        a = tf.tensor_scatter_nd_add(
            a,
            a_1,
            tf.ones(tf.shape(a_1)[0], dtype=a.dtype)
        )

        # Set diagonal for empty matrix
        a_empty = tf.linalg.diag(tf.cast(
            tf.repeat(tf.expand_dims(tf.range(tf.shape(a)[1], dtype=num_atoms.dtype), axis=0), tf.shape(a)[0],
                      axis=0) > tf.expand_dims(num_atoms, axis=-1), dtype=a.dtype))
        a = a + a_empty

        # return a, chi_pad
        charges = tf.linalg.solve(a, tf.expand_dims(chi_pad, axis=-1), adjoint=False, name=None)

        if self.output_to_tensor:
            return charges

        n_mask_pad = tf.pad(n_mask, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=0)
        return tf.RaggedTensor.from_row_lengths(charges[n_mask_pad], num_atoms, validate=self.ragged_validate)

    def get_config(self):
        config = super(CENTCharge, self).get_config()
        config.update({})
        return config
