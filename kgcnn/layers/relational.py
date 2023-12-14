import keras as ks
from keras.layers import Activation, Layer
from keras import ops


class RelationalDense(Layer):
    r"""Relational :obj:`Dense` for node or edge attributes, embeddings or features.

    A :obj:`RelationalDense` layer computes a densely-connected NN layer, i.e. a linear transformation of the input
    :math:`\mathbf{x}` with the kernel weights matrix :math:`\mathbf{W}_r` and bias :math:`\mathbf{b}`
    plus (possibly non-linear) activation function :math:`\sigma` for each type of relation :math:`r` that underlies
    the feature or embedding. Examples are different edge or node types such as chemical bonds and atomic species.

    .. math::

        \mathbf{x}'_r = \sigma (\mathbf{x}_r \mathbf{W}_r + \mathbf{b})

    This has been proposed by `Schlichtkrull et al. (2017) <https://arxiv.org/abs/1703.06103>`__ for graph networks.
    Additionally, there are a set of regularization schemes to improve performance and reduce the number of learnable
    parameters proposed by `Schlichtkrull et al. (2017) <https://arxiv.org/abs/1703.06103>`__ .
    Here, the following is implemented: basis-, block-diagonal-decomposition.
    With the basis decom-position, each :math:`\mathbf{W}_r` is defined as follows:

     .. math::

        \mathbf{W}_r = \sum_{b=1}^{B} a_{rb}\; \mathbf{V}_b

    i.e. as a linear combination of basis transformations :math:`V_b \in \mathbb{R}^{d' \times d}` with coefficients
    :math:`a_{rb}` such that only the coefficients depend on :math:`r`.
    In the block-diagonal decomposition, let each :math:`W_r` be defined through the direct sum over a set of
    low-dimensional matrices:

    .. math::

        \mathbf{W}_r = \otimes_{b=1}^{B} \mathbf{Q}_{br}

    Thereby, :math:`W_r` are block-diagonal matrices: :math:`diag(Q_{1r} , \dots , Q_{Br})` with
    :math:`Q_{br} \in \mathbb{R}^{(d'/B)\times(d/B)}`.
    Usage:

    .. code-block:: python

        from keras import ops
        from kgcnn.layers.relational import RelationalDense
        f = ops.convert_to_tensor([[0., 1.], [2., 2.]])
        r = ops.convert_to_tensor([1, 2])
        layer = RelationalDense(6, num_relations=5, num_bases=3, num_blocks=2)
        out = layer([f, r])
    """

    def __init__(self,
                 units: int,
                 num_relations: int,
                 num_bases: int = None,
                 num_blocks: int = None,
                 activation=None,
                 use_bias: bool = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        r"""Initialize layer similar to :obj:`ks.layers.Dense`.

         Args:
            units: Positive integer, dimensionality of the output space.
            num_relations: Number of relations expected to construct weights.
            num_bases: Number of kernel basis functions to construct relations. Default is None.
            num_blocks: Number of block-matrices to get for parameter reduction. Default is None.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
        """
        super(RelationalDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.bias_initializer = ks.initializers.get(bias_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.bias_regularizer = ks.regularizers.get(bias_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)
        self.bias_constraint = ks.constraints.get(bias_constraint)
        self._layer_activation = Activation(activation=activation, activity_regularizer=activity_regularizer)

    def build(self, input_shape):
        """Build layer, i.e. check input and construct weights for this layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            None.
        """
        assert len(input_shape) == 2, "`RelationalDense` layer requires feature plus relation information."

        feature_shape = input_shape[0]
        relation_shape = input_shape[1]
        last_dim = feature_shape[-1]
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {feature_shape}"
            )
        assert len(feature_shape) - 1 == len(relation_shape), "Relations must be without feature dimension."

        # Find kernel shape.
        if self.num_bases is not None:
            num_multi_kernels = self.num_bases
        else:
            num_multi_kernels = self.num_relations
        if self.num_blocks is not None:
            assert (last_dim % self.num_blocks == 0
                    and self.units % self.num_blocks == 0), "`num_blocks` must divide in- and output dimension."
            in_kernel_dim = int(last_dim / self.num_blocks)
            out_kernel_dim = int(self.units / self.num_blocks)
        else:
            in_kernel_dim = last_dim
            out_kernel_dim = self.units
        kernel_shape = [num_multi_kernels, in_kernel_dim, out_kernel_dim]
        if self.num_blocks is not None:
            kernel_shape = [num_multi_kernels, self.num_blocks, in_kernel_dim, out_kernel_dim]
        # Make kernel
        self.kernel = self.add_weight(
            name="kernel",
            shape=tuple(kernel_shape),
            initializer=self._multi_kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype, trainable=True,
        )
        if self.num_bases is not None:
            self.lin_bases = self.add_weight(
                name="lin_bases",
                shape=tuple([self.num_relations, self.num_bases]),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype, trainable=True,
            )
        else:
            self.lin_bases = None

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units, ),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype, trainable=True,
            )
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        assert len(input_shape) == 2, "`RelationalDense` requires '[features, relations]'."
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def _multi_kernel_initializer(self, shape, dtype=None, **kwargs):
        """Initialize multiple relational kernels.

        Args:
            shape: Shape of multi-kernel tensor.
            dtype: Optional dtype of the tensor.
            kwargs: Additional keyword arguments.

        Returns:
            Tensor: Tensor for initialization.
        """
        # For blocks the ks.initialize seems to have proper behaviour.
        # Each block is treated as a convolution field.
        if len(shape) < 3:
            return self.kernel_initializer(shape, dtype=dtype, **kwargs)
        # Initialize each kernel separately.
        separate_kernels = [
            ops.expand_dims(self.kernel_initializer(shape[1:], dtype=dtype, **kwargs), axis=0) for _ in range(shape[0])]
        relational_kernel = ops.concatenate(separate_kernels, axis=0)
        return relational_kernel

    def call(self, inputs, **kwargs):
        r"""Forward pass. Here, the relation is assumed to be encoded at axis=1.

        Args:
            inputs: [features, relations]

                - features (Tensor): Feature tensor of shape `([N], F)` of type 'float'.
                - relations (Tensor): Relation tensor of shape `([N], )` of type 'int'.

        Returns:
            Tensor: Processed feature tensor. Shape is `([N], units)` of type 'float'.
        """
        features, relations = inputs

        if self.num_bases is not None:
            kernel = ops.tensordot(self.lin_bases, self.kernel, [1, 0])
        else:
            kernel = self.kernel

        if self.num_blocks is not None:
            kernel_list = ops.split(kernel, ops.shape(kernel)[1], axis=1)
            kernel_list = [ops.squeeze(x, axis=1) for x in kernel_list]
            kernel_per_feature_list = [ops.take(x, relations, axis=0) for x in kernel_list]
            features_list = ops.split(features, ops.shape(features)[-1], axis=-1)
            new_feature_list = [self.batch_dot(x, y) for x, y in zip(features_list, kernel_per_feature_list)]
            new_features = ops.concatenate(new_feature_list, axis=-1)
        else:
            kernel_per_feature = ops.take(self.kernel, relations, axis=0)
            new_features = self.batch_dot(features, kernel_per_feature)

        if self.use_bias:
            new_features = new_features + self.bias
        new_features = self._layer_activation(new_features, **kwargs)
        return new_features

    @staticmethod
    def batch_dot(x, k):
        # x as features (..., N)
        # k as kernels (..., N, M)
        return ops.sum(ops.expand_dims(x, axis=-1) * k, axis=-2, keepdims=False)

    def get_config(self):
        """Update layer config."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "num_relations": self.num_relations,
            "num_bases": self.num_bases,
            "num_blocks": self.num_blocks,
            "kernel_initializer": ks.initializers.serialize(self.kernel_initializer),
            "bias_initializer": ks.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": ks.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": ks.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": ks.constraints.serialize(self.kernel_constraint),
            "bias_constraint": ks.constraints.serialize(self.bias_constraint),
        })
        config_act = self._layer_activation.get_config()
        for x in ["activation", "activity_regularizer"]:
            if x in config_act.keys():
                config.update({x: config_act[x]})
        return config
