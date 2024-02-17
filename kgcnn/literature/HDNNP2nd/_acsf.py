import keras as ks
import numpy as np
import math
from kgcnn.layers.gather import GatherNodes
from keras import ops
# from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.geom import NodeDistanceEuclidean, NodePosition
from kgcnn.layers.aggr import RelationalAggregateLocalEdges
# from kgcnn.layers.pooling import AggregateLocalEdges
from keras.layers import Multiply, Subtract, Layer
# from kgcnn.layers.modules import ExpandDims


class ACSFG2(Layer):
    r"""Atom-centered symmetry functions (ACSF) for high-dimensional neural network potentials (HDNNPs).

    `Jörg Behler, The Journal of Chemical Physics 134, 074106 (2011)
    <https://aip.scitation.org/doi/full/10.1063/1.3553717>`__

    This layer implements the radial part :math:`G_{i}^{2}` :

    .. math::

        G_{i}^{2} = \sum_{j \neq i} \; e^{−\eta \, (r_{ij} − \mu)^{2} } \; f_{ij}

    Here, for each atom type there is a set of parameters :math:`\eta` and :math:`\mu` and cutoff.
    The cutoff function :math:`f_ij = f_c(r_{ij})` is given by:

    .. math::

        f_c(r_{ij}) = 0.5 [\cos{\frac{\pi r_{ij}}{R_c}} + 1]

    In principle these parameters can be made trainable. The above sum is conducted for each atom type.

    Example:

    .. code-block:: python

        from kgcnn.literature.HDNNP2nd._acsf import ACSFG2
        from keras import ops
        layer = ACSFG2(
            eta_rs_rc=[[[0.0, 0.0, 8.0], [1.0, 0.0, 8.0]],[[0.0, 0.0, 8.0], [1.0, 0.0, 8.0]]],
            element_mapping=[1, 6]
        )
        z = ops.convert_to_tensor([1, 6])
        xyz = ops.convert_to_tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        eij = ops.convert_to_tensor([[0,1], [1, 0]], dtype="int64")
        rep_i = layer([z, xyz, eij])

    """

    _max_atomic_number = 96

    def __init__(self,
                 eta_rs_rc: list,
                 element_mapping: list,
                 add_eps: bool = False,
                 param_constraint=None,
                 param_regularizer=None,
                 param_initializer="zeros",
                 param_trainable: bool = False,
                 **kwargs):
        r"""Initialize layer.

        .. note::

            You can use simpler :obj:`make_param_table` method to generate `eta_zeta_lambda_rc` argument.

        Args:
            eta_rs_rc (list, np.ndarray): List of shape `(N, N, m, 3)` or `(N, m, 3)` where `N` are the considered
                atom types and m the number of representations. Tensor output will be shape `(batch, None, N*m)` .
                In the last dimension are the values for :math:`eta`, :math:`R_s` and :math:`R_c` .
            element_mapping (list): Atomic numbers of elements in :obj:`eta_rs_rc` , must have shape `(N, )` .
                Should not contain duplicate elements.
            add_eps (bool): Whether to add epsilon. Default is False.
            param_constraint: Parameter constraint for weights. Default is None.
            param_regularizer: Parameter regularizer for weights. Default is None.
            param_initializer: Parameter initializer for weights. Default is "zeros".
            param_trainable (bool): Parameter make trainable. Default is False.
        """
        super(ACSFG2, self).__init__(**kwargs)
        # eta_rs_rc of shape (N, N, m, 3) with m combinations of eta, rs, rc
        # or simpler (N, m, 3) where we repeat an additional N dimension assuming same parameter of source.
        self.eta_rs_rc = np.array(eta_rs_rc)
        assert len(self.eta_rs_rc.shape) in [3, 4], "Require `eta_rs_rc` of shape `(N, N, m, 3)` or `(N, m, 3)`"
        self.use_target_set = (len(self.eta_rs_rc.shape) == 4)
        self.num_relations = self.eta_rs_rc.shape[1] if self.use_target_set else self.eta_rs_rc.shape[0]
        self.element_mapping = np.array(element_mapping, dtype="int")  # of shape (N, ) with atomic number for eta_rs_rc
        self.reverse_mapping = np.empty(self._max_atomic_number, dtype="int")
        self.reverse_mapping.fill(np.iinfo(self.reverse_mapping.dtype).max)
        for i, pos in enumerate(self.element_mapping):
            self.reverse_mapping[pos] = i
        self.add_eps = add_eps

        self.lazy_mult = Multiply()
        self.layer_pos = NodePosition()
        self.layer_gather = GatherNodes(split_indices=[0, 1], concat_axis=None)
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = RelationalAggregateLocalEdges(num_relations=self.num_relations, pooling_method="sum")

        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = ks.initializers.deserialize(param_initializer)
        self.param_regularizer = ks.regularizers.deserialize(param_regularizer)
        self.param_constraint = ks.constraints.deserialize(param_constraint)
        self.param_trainable = param_trainable

        self.weight_eta_rs_rc = self.add_weight(
            name="eta_rs_rc",
            shape=self.eta_rs_rc.shape,
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=self.param_trainable
        )
        self.weight_reverse_mapping = self.add_weight(
            name="reverse_mapping",
            shape=(self._max_atomic_number,),
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype="int64", trainable=False
        )

        self.set_weights([self.eta_rs_rc, self.reverse_mapping])

    @staticmethod
    def make_param_table(eta: list, rs: list, rc: float, elements: list, **kwargs):
        r"""Simplified method to generate a parameter table and input for this layer based on a list of values for
        :math:`R_c` and :math:`\eta` etc.

        Args:
            eta (list): List of etas.
            rs (list): List of rs.
            rc (float): Single Cutoff value.
            elements (list): List of elements.

        Returns:
            dict: Kwargs input for this layer.
        """
        eta_rs_rc = [(et, Rs, rc) for Rs in rs for et in eta]
        elements = np.sort(elements)
        params = np.broadcast_to(eta_rs_rc, (len(elements), len(eta_rs_rc), 3))
        return {"eta_rs_rc": params, "element_mapping": elements, **kwargs}

    def _find_atomic_number_maps(self, inputs):
        return ops.take(self.weight_reverse_mapping, inputs, axis=0)

    @staticmethod
    def _take_2_(w, x1, x2):
        # take at 0, 1 axis
        w_shape = list(ops.shape(w))
        w_merge = ops.reshape(w, [w_shape[0]*w_shape[1]] + w_shape[2:])
        return ops.take(w_merge, x1 * ops.cast(w_shape[1], dtype=x1.dtype) + x2, axis=0)

    def _find_params_per_bond(self, inputs: list):
        zi_map, zj_map = inputs
        if self.use_target_set:
            params = self._take_2_(self.weight_eta_rs_rc, zi_map, zj_map)
        else:
            # Atomic specific for j but not i.
            params = ops.take(self.weight_eta_rs_rc, zj_map, axis=0)
        return params

    @staticmethod
    def _compute_fc(inputs: list):
        rij, params = inputs
        cutoff = ops.take(params, 2, axis=-1)
        fc = ops.clip(ops.broadcast_to(rij, ops.shape(cutoff)), -cutoff, cutoff)
        fc = (ops.cos(fc * math.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    @staticmethod
    def _compute_gaussian_expansion(inputs: list):
        rij, params = inputs
        eta, mu = ops.take(params, 0, axis=-1), ops.take(params, 1, axis=-1)
        arg = ops.square(rij - mu) * eta
        return ops.exp(-arg)

    @staticmethod
    def _flatten_relations(inputs):
        input_shape = list(ops.shape(inputs))
        flatten_shape = input_shape[:1] + [inputs.shape[1] * inputs.shape[2]]
        return ops.reshape(inputs, flatten_shape)

    def build(self, input_shape):
        super(ACSFG2, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [z, xyz, ij]

                - z (Tensor): Atomic numbers of shape ([N], )
                - xyz (Tensor): Node coordinates of shape ([N], 3)
                - ij (Tensor): Edge indices referring to nodes of shape (2, [M])

            mask: Boolean mask for inputs. Not used. Defaults to None.

        Returns:
            Tensor: Atomic representation of shape `(None, units)` .
        """
        z, xyz, eij = inputs
        z = ops.cast(z, dtype=eij.dtype)
        xi, xj = self.layer_pos([xyz, eij], **kwargs)
        rij = self.layer_dist([xi, xj], **kwargs)
        zi, zj = self.layer_gather([z, eij])
        zi_map = self._find_atomic_number_maps(zi)
        zj_map = self._find_atomic_number_maps(zj)
        params_per_bond = self._find_params_per_bond([zi_map, zj_map])
        fc = self._compute_fc([rij, params_per_bond])
        gij = self._compute_gaussian_expansion([rij, params_per_bond])
        rep = self.lazy_mult([gij, fc], **kwargs)
        pooled = self.pool_sum([xyz, rep, eij, zj_map], **kwargs)
        return self._flatten_relations(pooled)

    def get_config(self):
        config = super(ACSFG2, self).get_config()
        config.update({
            "eta_rs_rc": self.eta_rs_rc.tolist(),
            "element_mapping": self.element_mapping.tolist(),
            "add_eps": self.add_eps,
            "param_constraint": ks.constraints.serialize(self.param_constraint),
            "param_regularizer": ks.regularizers.serialize(self.param_regularizer),
            "param_initializer": ks.initializers.serialize(self.param_initializer),
            "param_trainable": self.param_trainable
        })
        return config


class ACSFG4(Layer):
    r"""Atom-centered symmetry functions (ACSF) for high-dimensional neural network potentials (HDNNPs).

    `Jörg Behler, The Journal of Chemical Physics 134, 074106 (2011)
    <https://aip.scitation.org/doi/full/10.1063/1.3553717>`_

    This layer implements the angular part :math:`G_{i}^{4}` :

    .. math::

        G_{i}^{4} =  \; \sum_{j\neq i} \sum_{k\neq j,i} \; 2^{1−\zeta}
        (1 + \lambda\, \cos{\theta_{ijk}})^\zeta\;
        \times \; e^{−\eta r_{ij}^{2}} \; e^{−\eta r_{ik}^{2}} \; e^{−\eta r_{jk}^{2}} \;
        \times \; f_{ij} \; f_{ik} \; f_{jk}

    Here, for each atom type there is a set of parameters :math:`\eta` , :math:`\mu` , :math:`\lambda`
    and :math:`\zeta`.
    The cutoff function :math:`f_ij = f_c(r_{ij})` is given by:

    Example:

    .. code-block:: python

        from keras import ops
        from kgcnn.literature.HDNNP2nd._acsf import ACSFG4
        layer = ACSFG4(
            eta_zeta_lambda_rc=[[[0.0, 1.0, -1.0, 8.0]],[[0.0, 1.0, -1.0, 8.0]], [[0.0, 1.0, -1.0, 8.0]]],
            element_mapping=[1, 6],
            keep_pair_order=False
        )
        z = ops.convert_to_tensor([1, 6, 6])
        xyz = ops.convert_to_tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        ijk = ops.transpose(ops.convert_to_tensor([[0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype="int64"))
        rep_i = layer([z, xyz, ijk])

    """

    _max_atomic_number = 96

    def __init__(self, eta_zeta_lambda_rc: list,
                 element_mapping: list,
                 element_pair_mapping: list = None,
                 add_eps: bool = False,
                 keep_pair_order: bool = False,
                 multiplicity: float = None,
                 param_initializer="zeros",
                 param_regularizer=None,
                 param_constraint=None,
                 param_trainable: bool = False,
                 **kwargs):
        r"""Initialize layer.

        .. note::

            You can use simpler :obj:`make_param_table` method to generate `eta_zeta_lambda_rc` argument.

        Args:
            eta_zeta_lambda_rc: A list of parameters of shape `(N, M, m, 4)` or simply `(M, m, 4)` where `m`
                represents the number of parameter sets, `N` the number of different atom types (set with
                :obj:`element_mapping` ) but which is optional, then all elements share parameters. And `M`
                being the number of angle combinations that can occur. By default, if order is ignored, this
                will be :math:`M=N(N+1)/2` combinations.
            element_mapping (list): Atomic numbers of elements in :obj:`eta_zeta_lambda_rc` ,
                must have shape `(N, )` . Should not contain duplicate elements.
            element_pair_mapping: Atomic pairs for :obj:`eta_zeta_lambda_rc` , where each entry contains atomic
                numbers. Must have shape `(M, 2)` . Default this is generated from N*(N+1)/2 combinations or
                N*N combinations if :obj:`keep_pair_order` is `False`. Can be set manually but must match shape.
            keep_pair_order (bool): Whether to have parameters for order atom pairs that make an angle.
                Default is False.
            multiplicity (float): Angle term is divided by multiplicity, if not None. Default is None.
            add_eps (bool): Whether to add epsilon. Default is False.
            param_constraint: Parameter constraint for weights. Default is None.
            param_regularizer: Parameter regularizer for weights. Default is None.
            param_initializer: Parameter initializer for weights. Default is "zeros".
            param_trainable (bool): Parameter make trainable. Default is False.
        """
        super(ACSFG4, self).__init__(**kwargs)
        self.add_eps = add_eps
        self.multiplicity = multiplicity
        self.keep_pair_order = keep_pair_order
        self.eta_zeta_lambda_rc = np.array(eta_zeta_lambda_rc, dtype="float")
        assert len(self.eta_zeta_lambda_rc.shape) in [3, 4], "Require `eta_zeta_lambda_rc` rank 3 or 4."
        self.use_target_set = (len(self.eta_zeta_lambda_rc.shape) == 4)
        if self.use_target_set:
            self.num_relations = self.eta_zeta_lambda_rc.shape[1]
        else:
            self.num_relations = self.eta_zeta_lambda_rc.shape[0]
        self.element_mapping = np.array(element_mapping, dtype="int")  # of shape (N, ) with atomic number.
        if element_pair_mapping is None:
            element_pair_index = np.expand_dims(self.element_mapping, axis=-1)
            self.element_pair_mapping = np.concatenate([
                    np.repeat(np.expand_dims(element_pair_index, axis=0), len(self.element_mapping), axis=0),
                    np.repeat(np.expand_dims(element_pair_index, axis=1), len(self.element_mapping), axis=1)
                ], axis=-1
            ).reshape((-1, 2))
            if not self.keep_pair_order:
                self.element_pair_mapping = np.sort(self.element_pair_mapping, axis=-1)
                self.element_pair_mapping = self.element_pair_mapping[
                    np.sort(np.unique(self.element_pair_mapping, axis=0, return_index=True)[1])]
        else:
            self.element_pair_mapping = np.array(element_pair_mapping, dtype="int")
        assert len(self.element_pair_mapping.shape) == 2 and self.element_pair_mapping.shape[1] == 2
        assert self.element_pair_mapping.shape[0] == self.num_relations

        self.reverse_mapping = np.empty(self._max_atomic_number, dtype="int")
        self.reverse_mapping.fill(np.iinfo(self.reverse_mapping.dtype).max)
        for i, pos in enumerate(self.element_mapping):
            self.reverse_mapping[pos] = i

        self.reverse_pair_mapping = np.empty((self._max_atomic_number, self._max_atomic_number), dtype="int")
        self.reverse_pair_mapping.fill(np.iinfo(self.reverse_pair_mapping.dtype).max)
        for i, pos in enumerate(self.element_pair_mapping):
            self.reverse_pair_mapping[pos[0], pos[1]] = i
            if not self.keep_pair_order:
                self.reverse_pair_mapping[pos[1], pos[0]] = i

        # Sub-layer.
        self.lazy_mult = Multiply()
        self.layer_pos = NodePosition(selection_index=[0, 1, 2])
        self.layer_dist = NodeDistanceEuclidean(add_eps=add_eps)
        self.pool_sum = RelationalAggregateLocalEdges(num_relations=self.num_relations, pooling_method="sum")
        self.lazy_sub = Subtract()
        self.layer_gather = GatherNodes(split_indices=[0, 1, 2], concat_axis=None)

        # We can do this in init since weights do not depend on input shape.
        self.param_initializer = ks.initializers.deserialize(param_initializer)
        self.param_regularizer = ks.regularizers.deserialize(param_regularizer)
        self.param_constraint = ks.constraints.deserialize(param_constraint)
        self.param_trainable = param_trainable

        self.weight_eta_zeta_lambda_rc = self.add_weight(
            name="eta_zeta_lambda_rc",
            shape=self.eta_zeta_lambda_rc.shape,
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype=self.dtype, trainable=self.param_trainable
        )
        self.weight_reverse_mapping = self.add_weight(
            name="reverse_mapping",
            shape=(self._max_atomic_number,),
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype="int64", trainable=False
        )
        self.weight_reverse_pair_mapping = self.add_weight(
            name="reverse_pair_mapping",
            shape=(self._max_atomic_number, self._max_atomic_number),
            initializer=self.param_initializer,
            regularizer=self.param_regularizer,
            constraint=self.param_constraint,
            dtype="int64", trainable=False
        )
        self.set_weights([self.eta_zeta_lambda_rc, self.reverse_mapping, self.reverse_pair_mapping])

    @staticmethod
    def make_param_table(eta: list, zeta: list, lamda: list, rc: float, elements: list, **kwargs):
        r"""Simplified method to generate a parameter table and input for this layer based on a list of values for
        :math:`R_c` and :math:`\eta` etc.

        Args:
            eta (list): List of etas.
            zeta (list): List of zeta.
            lamda (list): List of lamda.
            rc (float): Single Cutoff value.
            elements (list): List of elements.

        Returns:
            dict: Kwargs input for this layer.
        """
        eta_zeta_lambda_rc = [[eta, z, la, rc] for eta in eta for z in zeta for la in lamda]
        elements = np.sort(elements)
        params = np.broadcast_to(
            eta_zeta_lambda_rc, (int(len(elements) * (len(elements) + 1) / 2), len(eta_zeta_lambda_rc), 4))
        return {"eta_zeta_lambda_rc": params, "element_mapping": elements, "element_pair_mapping": None, **kwargs}

    def _find_atomic_number_maps(self, inputs):
        return ops.take(self.weight_reverse_mapping, inputs, axis=0)

    @staticmethod
    def _take_2_(w, x1, x2):
        # take at 0, 1 axis
        w_shape = list(ops.shape(w))
        w_merge = ops.reshape(w, [w_shape[0] * w_shape[1]] + w_shape[2:])
        return ops.take(w_merge, x1 * ops.cast(w_shape[1], dtype=x1.dtype) + x2, axis=0)

    def _find_atomic_number_pair_maps(self, inputs):
        zj, zk = inputs
        w_rev = self._take_2_(self.weight_reverse_pair_mapping, zj, zk)
        return w_rev

    def _find_params_per_bond(self, inputs: list):
        zi_map, zjk_map = inputs
        if self.use_target_set:
            params = self._take_2_(self.weight_eta_zeta_lambda_rc, zi_map, zjk_map)
        else:
            # Atomic specific for j,k but not i.
            params = ops.take(self.weight_eta_zeta_lambda_rc, zjk_map, axis=0)
        return params

    @staticmethod
    def _compute_fc(inputs: list):
        rij, params = inputs
        cutoff = ops.take(params, 3, axis=-1)
        fc = ops.clip(ops.broadcast_to(rij, ops.shape(cutoff)), -cutoff, cutoff)
        fc = (ops.cos(fc * np.pi / cutoff) + 1.0) * 0.5
        # fc = tf.where(tf.abs(inputs) < self.cutoff, fc, tf.zeros_like(fc))
        return fc

    @staticmethod
    def _compute_gaussian_expansion(inputs: list):
        rij, params = inputs
        eta = ops.take(params, 0, axis=-1)
        # mu = tf.gather(params, 1, axis=-1)
        arg = ops.square(rij) * eta
        return ops.exp(-arg)

    def _compute_pow_cos_angle_(self, inputs: list):
        vij, vik, rij, rik, params = inputs
        lamda, zeta = ops.take(params, 2, axis=-1), ops.take(params, 1, axis=-1)
        cos_theta = ops.sum(vij * vik, axis=-1, keepdims=True) / rij / rik
        cos_term = cos_theta * lamda + 1.0
        cos_term = ops.power(cos_term, zeta)
        scale = ops.ones_like(cos_term) * 2.0
        scaled_cos_term = ops.power(scale, 1.0 - zeta) * cos_term
        if self.multiplicity is not None:
            scaled_cos_term = scaled_cos_term/self.multiplicity
        return scaled_cos_term

    @staticmethod
    def _flatten_relations(inputs):
        input_shape = list(ops.shape(inputs))
        flatten_shape = input_shape[:1] + [inputs.shape[1] * inputs.shape[2]]
        return ops.reshape(inputs, flatten_shape)

    def build(self, input_shape):
        super(ACSFG4, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [z, xyz, ijk]

                - z (Tensor): Atomic numbers of shape ([N])
                - xyz (Tensor): Node coordinates of shape ([N], 3)
                - ijk (Tensor): Angle indices referring to nodes of shape (3, [M])

            mask: Boolean mask for inputs. Not used. Defaults to None.

        Returns:
            Tensor: Atomic representation of shape `(None, units)` .
        """
        z, xyz, ijk = inputs
        z = ops.cast(z, dtype=ijk.dtype)
        zi, zj, zk = self.layer_gather([z, ijk], **kwargs)
        xi, xj, xk = self.layer_pos([xyz, ijk], **kwargs)
        zi_map = self._find_atomic_number_maps(zi)
        zjk_map = self._find_atomic_number_pair_maps([zj, zk])
        params_per_bond = self._find_params_per_bond([zi_map, zjk_map])
        rij = self.layer_dist([xi, xj], **kwargs)
        rik = self.layer_dist([xi, xk], **kwargs)
        rjk = self.layer_dist([xj, xk], **kwargs)
        fij = self._compute_fc([rij, params_per_bond])
        fik = self._compute_fc([rik, params_per_bond])
        fjk = self._compute_fc([rjk, params_per_bond])
        gij = self._compute_gaussian_expansion([rij, params_per_bond])
        gik = self._compute_gaussian_expansion([rik, params_per_bond])
        gjk = self._compute_gaussian_expansion([rjk, params_per_bond])
        vij = self.lazy_sub([xi, xj], **kwargs)
        vik = self.lazy_sub([xi, xk], **kwargs)
        pow_cos_theta = self._compute_pow_cos_angle_([vij, vik, rij, rik, params_per_bond])
        rep = self.lazy_mult([pow_cos_theta, gij, gik, gjk, fij, fik, fjk], **kwargs)
        pool_ang = self.pool_sum([xyz, rep, ijk, zjk_map], **kwargs)
        return self._flatten_relations(pool_ang)

    def get_config(self):
        config = super(ACSFG4, self).get_config()
        config.update({
            "eta_zeta_lambda_rc": self.eta_zeta_lambda_rc,
            "add_eps": self.add_eps,
            "element_mapping": self.element_mapping,
            "keep_pair_order": self.keep_pair_order,
            "multiplicity": self.multiplicity,
            "element_pair_mapping": self.element_pair_mapping,
            "param_trainable": self.param_trainable,
            "param_constraint": ks.constraints.serialize(self.param_constraint),
            "param_regularizer": ks.regularizers.serialize(self.param_regularizer),
            "param_initializer": ks.initializers.serialize(self.param_initializer)
        })
        return config


class ACSFConstNormalization(Layer):
    """Simple layer to add a constant feature normalization to conform with reference code."""

    def __init__(self, std=1.0, mean=0.0, **kwargs):
        super(ACSFConstNormalization, self).__init__(**kwargs)
        self._np_std = np.array(std)
        self._np_mean = np.array(mean)
        # Could do some shape checks of std and mean here.
        self._t_std = ops.convert_to_tensor(self._np_std.tolist())
        self._t_mean = ops.convert_to_tensor(self._np_mean.tolist())

    def call(self, inputs, mask=None, **kwargs):
        r"""Forward pass.

        Args:
            inputs: Tensor of ACSF representation of shape `([None], units)` .
            mask: Boolean mask for inputs. Not used. Defaults to None.

        Returns:
            Tensor: Normalized atomic representation of shape `([None], units)` .
        """
        return (inputs-ops.cast(self._t_mean, inputs.dtype))/ops.cast(self._t_std, inputs.dtype)

    def get_config(self):
        config = super(ACSFConstNormalization, self).get_config()
        config.update({
            "mean": self._np_mean.tolist(),
            "std": self._np_std.tolist()
        })
        return config
