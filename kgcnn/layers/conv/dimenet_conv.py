import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import DenseEmbedding, LazyMultiply, LazyAdd
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.mlp import GraphMLP
from kgcnn.ops.polynom import spherical_bessel_jn_zeros, spherical_bessel_jn_normalization_prefactor, \
    tf_spherical_bessel_jn, tf_spherical_harmonics_yl


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ResidualLayer')
class ResidualLayer(GraphBaseLayer):
    """Residual Layer as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`_ .

    Args:
        units: Dimension of the kernel.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str): Activation function. Default is "kgcnn>swish".
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, units,
                 use_bias=True,
                 activation='kgcnn>swish',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize layer."""
        super(ResidualLayer, self).__init__(**kwargs)
        dense_args = {"units": units, "activation": activation, "use_bias": use_bias,
                      "kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                      "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                      "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                      "bias_initializer": bias_initializer}

        self.dense_1 = DenseEmbedding(**dense_args)
        self.dense_2 = DenseEmbedding(**dense_args)
        self.add_end = LazyAdd()

    def build(self, input_shape):
        """Build layer."""
        super(ResidualLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Node or edge embedding of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Node or edge embedding of shape (batch, [N], F)
        """
        x = self.dense_1(inputs, **kwargs)
        x = self.dense_2(x, **kwargs)
        x = self.add_end([inputs, x], **kwargs)
        return x

    def get_config(self):
        config = super(ResidualLayer, self).get_config()
        conf_dense = self.dense_1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias", "units"]:
            config.update({x: conf_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DimNetInteractionPPBlock')
class DimNetInteractionPPBlock(GraphBaseLayer):
    """DimNetPP Interaction Block as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`_ .

    Args:
        emb_size: Embedding size used for the messages
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size: Embedding size used inside the basis transformation
        num_before_skip: Number of residual layers in interaction block before skip connection
        num_after_skip: Number of residual layers in interaction block before skip connection
        use_bias (bool, optional): Use bias. Defaults to True.
        pooling_method (str): Pooling method information for layer. Default is 'sum'.
        activation (str): Activation function. Default is "kgcnn>swish".
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'kgcnn>glorot_orthogonal'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, emb_size,
                 int_emb_size,
                 basis_emb_size,
                 num_before_skip,
                 num_after_skip,
                 use_bias=True,
                 pooling_method="sum",
                 activation='kgcnn>swish',  # default is 'kgcnn>swish'
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer="kgcnn>glorot_orthogonal",  # default is 'kgcnn>glorot_orthogonal'
                 bias_initializer='zeros',
                 **kwargs):
        super(DimNetInteractionPPBlock, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.pooling_method = pooling_method
        self.emb_size = emb_size
        self.int_emb_size = int_emb_size
        self.basis_emb_size = basis_emb_size
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = DenseEmbedding(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_rbf2 = DenseEmbedding(emb_size, use_bias=False, **kernel_args)
        self.dense_sbf1 = DenseEmbedding(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_sbf2 = DenseEmbedding(int_emb_size, use_bias=False, **kernel_args)

        # Dense transformations of input messages
        self.dense_ji = DenseEmbedding(emb_size, activation=activation, use_bias=True, **kernel_args)
        self.dense_kj = DenseEmbedding(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Embedding projections for interaction triplets
        self.down_projection = DenseEmbedding(int_emb_size, activation=activation, use_bias=False, **kernel_args)
        self.up_projection = DenseEmbedding(emb_size, activation=activation, use_bias=False, **kernel_args)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))
        self.final_before_skip = DenseEmbedding(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))

        self.lay_add1 = LazyAdd()
        self.lay_add2 = LazyAdd()
        self.lay_mult1 = LazyMultiply()
        self.lay_mult2 = LazyMultiply()

        self.lay_gather = GatherNodesOutgoing()  # Are edges here
        self.lay_pool = PoolingLocalEdges(pooling_method=pooling_method)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [edges, rbf, sbf, angle_index]

                - edges (tf.RaggedTensor): Edge embeddings of shape (batch, [M], F)
                - rbf (tf.RaggedTensor): Radial basis features of shape (batch, [M], F)
                - sbf (tf.RaggedTensor): Spherical basis features of shape (batch, [K], F)
                - angle_index (tf.RaggedTensor): Angle indices referring to two edges of shape (batch, [K], 2)

        Returns:
            tf.RaggedTensor: Updated edge embeddings.
        """
        x, rbf, sbf, id_expand = inputs

        # Initial transformation
        x_ji = self.dense_ji(x, **kwargs)
        x_kj = self.dense_kj(x, **kwargs)

        # Transform via Bessel basis
        rbf = self.dense_rbf1(rbf, **kwargs)
        rbf = self.dense_rbf2(rbf, **kwargs)
        x_kj = self.lay_mult1([x_kj, rbf], **kwargs)

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.down_projection(x_kj, **kwargs)
        x_kj = self.lay_gather([x_kj, id_expand], **kwargs)

        # Transform via 2D spherical basis
        sbf = self.dense_sbf1(sbf, **kwargs)
        sbf = self.dense_sbf2(sbf, **kwargs)
        x_kj = self.lay_mult1([x_kj, sbf], **kwargs)

        # Aggregate interactions and up-project embeddings
        x_kj = self.lay_pool([rbf, x_kj, id_expand], **kwargs)
        x_kj = self.up_projection(x_kj, **kwargs)

        # Transformations before skip connection
        x2 = self.lay_add1([x_ji, x_kj], **kwargs)
        for layer in self.layers_before_skip:
            x2 = layer(x2, **kwargs)
        x2 = self.final_before_skip(x2, **kwargs)

        # Skip connection
        x = self.lay_add2([x, x2],**kwargs)

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x, **kwargs)

        return x

    def get_config(self):
        config = super(DimNetInteractionPPBlock, self).get_config()
        config.update({"use_bias": self.use_bias, "pooling_method": self.pooling_method, "emb_size": self.emb_size,
                       "int_emb_size": self.int_emb_size, "basis_emb_size": self.basis_emb_size,
                       "num_before_skip": self.num_before_skip, "num_after_skip": self.num_after_skip})
        conf_dense = self.dense_ji.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DimNetOutputBlock')
class DimNetOutputBlock(GraphBaseLayer):
    """DimNetPP Output Block as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`_ .

    Args:
        emb_size (list): List of node embedding dimension.
        out_emb_size (list): List of edge embedding dimension.
        num_dense (list): Number of dense layer for MLP.
        num_targets (int): Number of output target dimension. Defaults to 12.
        use_bias (bool, optional): Use bias. Defaults to True.
        kernel_initializer: Initializer for kernels. Default is 'glorot_orthogonal' with fallback 'orthogonal'.
        output_kernel_initializer: Initializer for last kernel. Default is 'zeros'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        activation (str): Activation function. Default is 'kgcnn>swish'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        pooling_method (str): Pooling method information for layer. Default is 'mean'.
    """

    def __init__(self, emb_size,
                 out_emb_size,
                 num_dense,
                 num_targets=12,
                 use_bias=True,
                 output_kernel_initializer="zeros", kernel_initializer='kgcnn>glorot_orthogonal',
                 bias_initializer='zeros',
                 activation='kgcnn>swish',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 pooling_method="sum",
                 **kwargs):
        """Initialize layer."""
        super(DimNetOutputBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size
        self.num_dense = num_dense
        self.num_targets = num_targets
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_initializer": bias_initializer,
                       "bias_regularizer": bias_regularizer, "bias_constraint": bias_constraint, }

        self.dense_rbf = DenseEmbedding(emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.up_projection = DenseEmbedding(out_emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.dense_mlp = GraphMLP([out_emb_size] * num_dense, activation=activation,
                                  kernel_initializer=kernel_initializer, use_bias=use_bias, **kernel_args)
        self.dimnet_mult = LazyMultiply()
        self.pool = PoolingLocalEdges(pooling_method=self.pooling_method)
        self.dense_final = DenseEmbedding(num_targets, use_bias=False, kernel_initializer=output_kernel_initializer,
                                          **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetOutputBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index, state]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - rbf (tf.RaggedTensor): Edge distance basis of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F_T).
        """
        # Calculate edge Update
        n_atoms, x, rbf, idnb_i = inputs
        g = self.dense_rbf(rbf, **kwargs)
        x = self.dimnet_mult([g, x], **kwargs)
        x = self.pool([n_atoms, x, idnb_i], **kwargs)
        x = self.up_projection(x, **kwargs)
        x = self.dense_mlp(x, **kwargs)
        x = self.dense_final(x, **kwargs)
        return x

    def get_config(self):
        config = super(DimNetOutputBlock, self).get_config()
        conf_mlp = self.dense_mlp.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_mlp[x][0]})
        conf_dense_output = self.dense_final.get_config()
        config.update({"output_kernel_initializer": conf_dense_output["kernel_initializer"]})
        config.update({"pooling_method": self.pooling_method, "use_bias": self.use_bias})
        config.update({"emb_size": self.emb_size, "out_emb_size": self.out_emb_size, "num_dense": self.num_dense,
                       "num_targets": self.num_targets})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EmbeddingDimeBlock')
class EmbeddingDimeBlock(tf.keras.layers.Layer):
    """Custom Embedding Block of `DimNetPP <https://arxiv.org/abs/2011.14115>`_ . Naming of inputs here should match
    keras Embedding layer.

    Args:
        input_dim (int): Integer. Size of the vocabulary, i.e. maximum integer index + 1.
        output_dim (int): Integer. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the embeddings matrix (see keras.initializers).
        embeddings_regularizer: Regularizer function applied to the embeddings matrix (see keras.regularizers).
        embeddings_constraint: Constraint function applied to the embeddings matrix (see keras.constraints).

    """
    def __init__(self,
                 input_dim,  # Vocabulary
                 output_dim,  # Embedding size
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        super(EmbeddingDimeBlock, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

        # Original implementation used initializer:
        # embeddings_initializer = {'class_name': 'RandomUniform', 'config': {'minval': -1.7320508075688772,
        # 'maxval': 1.7320508075688772, 'seed': None}}
        self.embeddings = self.add_weight(name="embeddings", shape=(self.input_dim + 1, self.output_dim),
                                          dtype=self.dtype, initializer=self.embeddings_initializer,
                                          regularizer=self.embeddings_regularizer,
                                          constraint=self.embeddings_constraint,
                                          trainable=True)

    def call(self, inputs, **kwargs):
        """Embedding of inputs. Forward pass."""
        out = tf.gather(self.embeddings, tf.cast(inputs, dtype=tf.int32))
        return out

    def get_config(self):
        config = super(EmbeddingDimeBlock, self).get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim,
                       "embeddings_initializer": tf.keras.initializers.serialize(self.embeddings_initializer),
                       "embeddings_regularizer": tf.keras.regularizers.serialize(self.embeddings_regularizer),
                       "embeddings_constraint": tf.keras.constraints.serialize(self.embeddings_constraint)
                       })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SphericalBasisLayer')
class SphericalBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Bessel Basis with :math:`l=m=0`, according to
    `Klicpera et al. 2020 <https://arxiv.org/abs/2011.14115>`_ .

    Args:
        num_spherical (int): Number of spherical basis functions
        num_radial (int): Number of radial basis functions
        cutoff (float): Cutoff distance c
        envelope_exponent (int): Degree of the envelope to smoothen at cutoff. Default is 5.
    """

    def __init__(self, num_spherical,
                 num_radial,
                 cutoff,
                 envelope_exponent=5,
                 **kwargs):
        super(SphericalBasisLayer, self).__init__(**kwargs)

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope_exponent = envelope_exponent

        # retrieve formulas
        self.bessel_n_zeros = spherical_bessel_jn_zeros(num_spherical, num_radial)
        self.bessel_norm = spherical_bessel_jn_normalization_prefactor(num_spherical, num_radial)

        self.layer_gather_out = GatherNodesOutgoing(**self._kgcnn_info)

    @tf.function
    def envelope(self, inputs):
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1 / inputs + a * inputs ** (p - 1) + b * inputs ** p + c * inputs ** (p + 1)
        return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [distance, angles, angle_index]

                - distance (tf.RaggedTensor): Edge distance of shape (batch, [M], 1)
                - angles (tf.RaggedTensor): Angle list of shape (batch, [K], 1)
                - angle_index (tf.RaggedTensor): Angle indices referring to edges of shape (batch, [K], 2)

        Returns:
            tf.RaggedTensor: Expanded angle/distance basis. Shape is (batch, [K], #Radial * #Spherical)
        """
        self.assert_ragged_input_rank(inputs)
        edge, edge_part = inputs[0].values, inputs[0].row_splits
        angles, angle_part = inputs[1].values, inputs[1].row_splits

        d = edge
        d_scaled = d[:, 0] * self.inv_cutoff
        rbf = []
        for n in range(self.num_spherical):
            for k in range(self.num_radial):
                rbf += [self.bessel_norm[n, k] * tf_spherical_bessel_jn(d_scaled * self.bessel_n_zeros[n][k], n)]
        rbf = tf.stack(rbf, axis=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf
        ragged_rbf_env = tf.RaggedTensor.from_row_splits(rbf_env, edge_part, validate=self.ragged_validate)
        rbf_env = self.layer_gather_out([ragged_rbf_env, inputs[2]], **kwargs).values
        # rbf_env = tf.gather(rbf_env, id_expand_kj[:, 1])

        cbf = [tf_spherical_harmonics_yl(angles[:, 0], n) for n in range(self.num_spherical)]
        cbf = tf.stack(cbf, axis=1)
        cbf = tf.repeat(cbf, self.num_radial, axis=1)
        out = rbf_env * cbf

        out = tf.RaggedTensor.from_row_splits(out, angle_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update config."""
        config = super(SphericalBasisLayer, self).get_config()
        config.update({"num_radial": self.num_radial, "cutoff": self.cutoff,
                       "envelope_exponent": self.envelope_exponent, "num_spherical": self.num_spherical})
        return config