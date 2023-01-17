import tensorflow as tf
import numpy as np
import math
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import ExpandDims
from kgcnn.layers.gather import GatherNodesSelection
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.pooling import PoolingGlobalEdges, PoolingNodes

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='CENTCharge')
class CENTCharge(GraphBaseLayer):
    r"""Compute charge equilibration according to
    `Ko et al. (2021) <https://www.nature.com/articles/s41467-020-20427-2>`_ .

    The charge equilibration scheme seeks to minimize the energy :math:`{E}_{{\rm{Qeq}}}` of the following expression:

    .. math::

        {E}_{{\rm{Qeq}}}={E}_{{\rm{elec}}}+\mathop{\sum }\limits_{i=1}^{{N}_{{\rm{at}}}}({\chi }_{i}{Q}_{i}+
        \frac{1}{2}{J}_{i}{Q}_{i}^{2})\quad ,

    where :math:`{E}_{{\rm{elec}}}` is given as:

    .. math::

        {E}_{{\rm{elec}}}=\mathop{\sum }\limits_{i=1}^{{N}_{{\rm{at}}}}\mathop{\sum }\limits_{j<i}^{{N}_{{\rm{at}}}}
        \frac{{\rm{erf}}\left(\frac{{r}_{ij}}{\sqrt{2}{\gamma }_{ij}}\right)}{{r}_{ij}}{Q}_{i}{Q}_{j}+
        \mathop{\sum }\limits_{i=1}^{{N}_{{\rm{at}}}}\frac{{Q}_{i}^{2}}{2{\sigma }_{i}\sqrt{\pi }}
    
    with
    
    .. math::
    
        {\gamma }_{ij}=\sqrt{{\sigma }_{i}^{2}+{\sigma }_{j}^{2}}\quad .

    Where :math:`{J}_{i}` denotes the element-specific hardness and charge densities of width :math:`\sigma_i` taken
    from the covalent radii of the respective elements.
    To solve this minimization problem the derivatives with respect to the charges are set to zero:

    .. math::

        \frac{\partial {E}_{{\rm{Qeq}}}}{\partial {Q}_{i}}=0,\forall i=1,..,{N}_{{\rm{at}}}\ \Rightarrow \
        \sum\limits_{j=1}^{{N}_{{\rm{at}}}}{A}_{ij}{Q}_{j}+{\chi }_{i}=0

    with elements of :math:`{[{\bf{A}}]}_{ij}` defined like:

    .. math::

        {[{\bf{A}}]}_{ij}=\left\{\begin{array}{ll}{J}_{i}+\frac{1}{{\sigma }_{i}\sqrt{\pi }},&\,\text{if}\,\,
        i=j\\ \frac{{\rm{erf}}\left(\frac{{r}_{ij}}{\sqrt{2}{\gamma }_{ij}}\right)}{{r}_{ij}},&\,
        \text{otherwise}\,\end{array}\right.

    Finally, the linear equation system to solve with lagrange multiplier :math:`\lambda` reads:

    .. math::

        \begin{pmatrix} \begin{matrix}  \; & \; & \; \\ \; & {\bf{A}} & \; \\ \; & \; & \; \\ \end{matrix} &
        & \vert & \begin{matrix} 1 \\ \vdots \\ 1\end{matrix} \\ \hline
        \begin{matrix} 1 & \dots & 1\end{matrix} & & \vert & 0 \end{pmatrix} =
        \begin{pmatrix} Q_1 \\ \vdots \\ Q_{{N}_{{\rm{at}}} \\ \hline \\ \lambda \end{pmatrix}
        \begin{pmatrix} \chi_1 \\ \vdots \\ \chi_{{N}_{{\rm{at}}} \\ \hline \\ Q_{\text{tot}} \end{pmatrix}

    A code example of using the layer and possible input is shown below:

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.conv.hdnnp_conv import CENTCharge
        layer = CENTCharge()
        z = tf.ragged.constant([[1, 6], [1, 1, 6]], ragged_rank=1)
        chi = tf.ragged.constant([[0.43, 0.37], [0.43, 0.43, 0.37]], ragged_rank=1)
        xyz = tf.ragged.constant([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.2], [2.0, 0.0, 0.0]]], ragged_rank=1, inner_shape=(3,))
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
        220, 195, 190, 175, 164, 154, 147, 146, 142, 139, 145, 144, 142, 139, 139, 138, 139, 140,
        244, 215, 207, 204, 203, 201, 199, 198, 198, 196, 194, 192, 192, 189, 190, 187, 175, 187, 170, 162, 151, 144,
        141, 136, 136, 132, 145, 146, 148, 140, 150, 150,
        260, 221, 215, 206, 200, 196, 190, 187, 180, 169
    ])
    # Chemical hardness from https://www.pnas.org/doi/10.1073/pnas.2117416119 in eV
    _default_hardness = 0.037 / 0.529177 * np.array([
        0.0, 6.2, 8.8,
        2.2, 4.6, 3.8, 4.7, 7.1, 5.6, 6.1, 9.1,
        2.1, 4.0, 2.6, 3.3, 4.7, 3.8, 4.5, 7.7,
        2.3, 3.2, 3.2, 2.9, 3.2, 3.4, 4.0, 3.6, 3.3, 3.3, 3.8, 5.8, 3.0, 3.3, 4.5, 3.9, 4.2, 7.7,
        1.9, 3.1, 3.1, 2.9, 3.3, 3.5, 3.7, 3.7, 3.9, 4.1, 3.6, 5.4, 3.1, 3.1, 4.0, 3.6, 3.8, 6.8,
        1.8, 2.7, 2.4, 2.3, 2.5, 2.7, 2.5, 3.0, 3.0, 3.2, 3.2, 3.3, 3.3, 3.3, 3.1, 3.5, 3.2, 3.8, 3.1, 3.6, 3.7, 3.7,
        3.8, 3.5, 3.6, 5.8, 3.1, 3.4, 3.3, 3.6, 3.6, 6.1,
        1.8, 3.0, 2.8, 2.8, 3.1, 3.0, 3.1, 3.5, 3.3, 3.3
    ])
    _max_atomic_number = 97

    def __init__(self, output_to_tensor: bool = False, use_physical_params: bool = True,
                 param_constraint=None, param_regularizer=None, param_initializer="glorot_uniform",
                 param_trainable: bool = False,
                 **kwargs):
        super(CENTCharge, self).__init__(**kwargs)
        self.output_to_tensor = output_to_tensor
        self.use_physical_params = use_physical_params

        self.layer_cast_n = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask", boolean_mask=True)
        self.layer_cast_chi = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask")
        self.layer_cast_x = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask", boolean_mask=True)

        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = ks.initializers.deserialize(param_initializer)
        self.param_regularizer = ks.regularizers.deserialize(param_regularizer)
        self.param_constraint = ks.constraints.deserialize(param_constraint)
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
        if self.use_physical_params:
            self.set_weights([self._default_hardness, self._default_radii])

    def build(self, input_shape):
        super(CENTCharge, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass. Casts to padded tensor for :obj:`tf.linalg.solve()` .

        Args:
            inputs (list): [n, chi, xyz, qtot]

                - n (tf.RaggedTensor): Atomic numbers of shape (batch, [N])
                - chi (tf.RaggedTensor): Learned electronegativities. Shape (batch, [N], 1)
                - xyz (tf.RaggedTensor): Node coordinates of shape (batch, [N], 3)
                - qtot (tf.Tensor): Total charge per molecule of shape (batch, 1)

            mask (list): Boolean mask for inputs. Not used. Defaults to None.

        Returns:
            tf.RaggedTensor: Charges of shape (batch, None, 1)
        """
        n, chi, x = self.assert_ragged_input_rank(inputs[:3], mask=mask, ragged_rank=1)
        qtot = inputs[3]
        if qtot.shape.rank > 1:
            qtot = tf.squeeze(qtot, axis=-1)

        num_atoms = n.row_lengths()

        # Cast to padded tensor for charge equilibrium.
        # Only keep mask for atomic number.
        n_pad, n_mask = self.layer_cast_n(n, **kwargs)
        chi_pad, _ = self.layer_cast_chi(chi, **kwargs)
        x_pad, _ = self.layer_cast_x(x, **kwargs)
        if chi_pad.shape.rank > 2:
            chi_pad = tf.squeeze(chi_pad, axis=-1)

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
        )
        a = tf.where(
            a_diag_mask,
            diag_part,
            a
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
        idx = tf.concat([tf.concat([idx[0], idx[1], idx[2]], axis=-1),
                         tf.concat([idx[0], idx[2], idx[1]], axis=-1)], axis=0)
        a = tf.tensor_scatter_nd_add(
            a,
            idx,
            tf.ones(tf.shape(idx)[0], dtype=a.dtype)
        )

        # Set diagonal for empty matrix
        idx_0 = tf.ragged.range(num_atoms + 1, tf.repeat(tf.shape(a)[2], tf.shape(num_atoms)[0]))
        idx_0 = tf.concat([tf.cast(tf.expand_dims(iter_tensor, axis=-1), dtype="int32") for iter_tensor in [
            idx_0.value_rowids(), idx_0.flat_values, idx_0.flat_values
        ]], axis=-1)
        a = tf.tensor_scatter_nd_add(
            a,
            idx_0,
            tf.ones(tf.shape(idx_0)[0], dtype=a.dtype)
        )

        # Check system to solve
        # return a, chi_pad

        # Compute charges with solve().
        charges = tf.linalg.solve(a, tf.expand_dims(chi_pad, axis=-1), adjoint=False, name=None)

        n_mask_pad = tf.pad(n_mask, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=0)
        if self.output_to_tensor:
            # Padded tensor will have total charges in it.
            return charges, n_mask_pad

        # Make ragged tensor from padded charges.
        return tf.RaggedTensor.from_row_lengths(charges[n_mask_pad], num_atoms, validate=self.ragged_validate)

    def get_config(self):
        config = super(CENTCharge, self).get_config()
        config.update({
            "output_to_tensor": self.output_to_tensor,
            "use_physical_params": self.use_physical_params,
            "param_constraint": ks.constraints.serialize(self.param_constraint),
            "param_regularizer": ks.regularizers.serialize(self.param_regularizer),
            "param_initializer": ks.initializers.serialize(self.param_initializer),
            "param_trainable": self.param_trainable
        })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='ElectrostaticEnergyCharge')
class ElectrostaticEnergyCharge(GraphBaseLayer):
    r"""Compute electric energy according to
    `Ko et al. (2021) <https://www.nature.com/articles/s41467-020-20427-2>`_ .

    where :math:`{E}_{{\rm{elec}}}` is given as:

    .. math::

        {E}_{{\rm{elec}}}=\frac{1}{2}\,
        \mathop{\sum }\limits_{i=1}^{{N}_{{\rm{at}}}}\mathop{\sum }\limits_{j\neq i}^{{N}_{{\rm{at}}}}
        \frac{{\rm{erf}}\left(\frac{{r}_{ij}}{\sqrt{2}{\gamma }_{ij}}\right)}{{r}_{ij}}{Q}_{i}{Q}_{j}+
        \mathop{\sum }\limits_{i=1}^{{N}_{{\rm{at}}}}\frac{{Q}_{i}^{2}}{2{\sigma }_{i}\sqrt{\pi }}
    
    with
    
    .. math::
    
        {\gamma }_{ij}=\sqrt{{\sigma }_{i}^{2}+{\sigma }_{j}^{2}}\quad .


    Where :math:`{J}_{i}` denotes the element-specific hardness and charge densities of width :math:`\sigma_i` taken
    from the covalent radii of the respective elements.

    However, here the indices of atoms for the pair contribution in the energy, are expected as graph-like indices.
    The factor :math:`\frac{1}{2}` can be controlled by multiplicity argument.
    Example of using this layer:

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.conv.hdnnp_conv import ElectrostaticEnergyCharge
        layer = ElectrostaticEnergyCharge()
        z = tf.ragged.constant([[1, 6], [1, 1, 6]], ragged_rank=1)
        q = tf.ragged.constant([[0.43, 0.37], [0.43, 0.43, 0.37]], ragged_rank=1)
        xyz = tf.ragged.constant([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.2], [2.0, 0.0, 0.0]]], ragged_rank=1, inner_shape=(3,))
        ij = tf.ragged.constant([[[0, 1], [1, 0]],
            [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]], ragged_rank=1, inner_shape=(2,))
        eng = layer([z, q, xyz, ij])
        print(eng)

    """

    # From https://en.wikipedia.org/wiki/Covalent_radius with radii in pm.
    _default_radii = 0.01 * np.array([
        0.0, 31, 28,
        128, 96, 84, 73, 71, 66, 57, 58,
        166, 141, 121, 111, 107, 105, 102, 106,
        203, 176, 170, 160, 153, 139, 139, 132, 126, 124, 132, 122, 122, 120, 119, 120, 120, 116,
        220, 195, 190, 175, 164, 154, 147, 146, 142, 139, 145, 144, 142, 139, 139, 138, 139, 140,
        244, 215, 207, 204, 203, 201, 199, 198, 198, 196, 194, 192, 192, 189, 190, 187, 175, 187, 170, 162, 151, 144,
        141, 136, 136, 132, 145, 146, 148, 140, 150, 150,
        260, 221, 215, 206, 200, 196, 190, 187, 180, 169
    ])

    _max_atomic_number = 97

    def __init__(self, add_eps: bool = False, use_physical_params: bool = True, multiplicity: float = 2.0,
                 param_constraint=None, param_regularizer=None, param_initializer="glorot_uniform",
                 param_trainable: bool = False, **kwargs):
        super(ElectrostaticEnergyCharge, self).__init__(**kwargs)
        self.add_eps = add_eps
        self.multiplicity = multiplicity
        self.use_physical_params = use_physical_params
        self.layer_pos = NodePosition(selection_index=[0, 1])
        self.layer_gather = GatherNodesSelection(selection_index=[0, 1])
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.layer_exp_dims = ExpandDims(axis=2)
        self.layer_pool_edges = PoolingGlobalEdges(pooling_method="sum")
        self.layer_pool_nodes = PoolingNodes(pooling_method="sum")

        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = ks.initializers.deserialize(param_initializer)
        self.param_regularizer = ks.regularizers.deserialize(param_regularizer)
        self.param_constraint = ks.constraints.deserialize(param_constraint)
        self.param_trainable = param_trainable

        self.weight_sigma = self.add_weight(
            "sigma",
            shape=(self._max_atomic_number,),
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=self.param_trainable
        )
        if self.use_physical_params:
            self.set_weights([self._default_radii])

    def build(self, input_shape):
        super(ElectrostaticEnergyCharge, self).build(input_shape)

    def _find_sigma_from_atom_number(self, inputs):
        return tf.gather(self.weight_sigma, inputs, axis=0)

    def _compute_gamma(self, inputs):
        sigma_i, sigma_j = inputs
        gamma_squared = tf.square(sigma_i) + tf.square(sigma_j)
        if self.add_eps:
            gamma_squared = gamma_squared + ks.backend.epsilon()
        return tf.sqrt(gamma_squared)

    @staticmethod
    def _compute_pair_energy(inputs):
        qi, qj, rij, gamma_ij = inputs
        frac = tf.math.divide_no_nan(tf.math.erf(tf.math.divide_no_nan(rij, gamma_ij * tf.sqrt(2.0))), rij)
        return qi*qj*frac

    @staticmethod
    def _compute_self_energy(inputs):
        q, sigma = inputs
        return tf.math.divide_no_nan(tf.square(q), sigma)/2.0/tf.sqrt(math.pi)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [n, q, xyz, ij]

                - n (tf.RaggedTensor): Atomic numbers of shape (batch, [N])
                - q (tf.RaggedTensor): Learned atomic charges. Shape (batch, [N], 1)
                - xyz (tf.RaggedTensor): Node coordinates of shape (batch, [N], 3)
                - ij (tf.RaggedTensor): Edge indices of shape (batch, [N], 2)

            mask (list): Boolean mask for inputs. Not used. Defaults to None.

        Returns:
            tf.Tensor: Energy of shape (batch, 1)

        """
        n, q, xyz, ij = self.assert_ragged_input_rank(inputs, mask=mask, ragged_rank=1)
        if n.shape.rank <= 2:
            n = self.layer_exp_dims(n, **kwargs)
        if q.shape.rank <= 2:
            q = self.layer_exp_dims(q, **kwargs)
        xi, xj = self.layer_pos([xyz, ij], **kwargs)
        rij = self.layer_dist([xi, xj], **kwargs)
        ni, nj = self.layer_gather([n, ij], **kwargs)
        qi, qj = self.layer_gather([q, ij], **kwargs)
        sigma_i = self.map_values(self._find_sigma_from_atom_number, ni)
        sigma_j = self.map_values(self._find_sigma_from_atom_number, nj)
        sigma = self.map_values(self._find_sigma_from_atom_number, n)
        gamma_ij = self.map_values(self._compute_gamma, [sigma_i, sigma_j])
        pair_energy = self.map_values(self._compute_pair_energy, [qi, qj, rij, gamma_ij])
        self_energy = self.map_values(self._compute_self_energy, [q, sigma])
        sum_pair = self.layer_pool_edges(pair_energy)
        sum_self = self.layer_pool_nodes(self_energy)
        # Both are normal tensors now with shape (batch, 1).
        if self.multiplicity:
            sum_pair = sum_pair/self.multiplicity
        return sum_pair + sum_self

    def get_config(self):
        config = super(ElectrostaticEnergyCharge, self).get_config()
        config.update({
            "add_eps": self.add_eps,
            "multiplicity": self.multiplicity,
            "use_physical_params": self.use_physical_params,
            "param_constraint": ks.constraints.serialize(self.param_constraint),
            "param_regularizer": ks.regularizers.serialize(self.param_regularizer),
            "param_initializer": ks.initializers.serialize(self.param_initializer),
            "param_trainable": self.param_trainable
        })
        return config



