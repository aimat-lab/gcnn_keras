import keras as ks
from keras import ops
from keras.layers import Dense, Multiply, Add, Layer
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.update import ResidualLayer
from kgcnn.initializers.initializers import GlorotOrthogonal, HeOrthogonal


class DimNetInteractionPPBlock(Layer):
    """DimNetPP Interaction Block as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`__ .

    Args:
        emb_size: Embedding size used for the messages
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size: Embedding size used inside the basis transformation
        num_before_skip: Number of residual layers in interaction block before skip connection
        num_after_skip: Number of residual layers in interaction block before skip connection
        use_bias (bool, optional): Use bias. Defaults to True.
        pooling_method (str): Pooling method information for layer. Default is 'sum'.
        activation (str): Activation function. Default is "swish".
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
                 activation='swish',
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
        self.dense_rbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_rbf2 = Dense(emb_size, use_bias=False, **kernel_args)
        self.dense_sbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_sbf2 = Dense(int_emb_size, use_bias=False, **kernel_args)

        # Dense transformations of input messages
        self.dense_ji = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)
        self.dense_kj = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Embedding projections for interaction triplets
        self.down_projection = Dense(int_emb_size, activation=activation, use_bias=False, **kernel_args)
        self.up_projection = Dense(emb_size, activation=activation, use_bias=False, **kernel_args)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))
        self.final_before_skip = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))

        self.lay_add1 = Add()
        self.lay_add2 = Add()
        self.lay_mult1 = Multiply()
        self.lay_mult2 = Multiply()

        self.lay_gather = GatherNodesOutgoing()  # Are edges here
        self.lay_pool = AggregateLocalEdges(pooling_method=pooling_method)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetInteractionPPBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [edges, rbf, sbf, angle_index]

                - edges (Tensor): Edge embeddings of shape ([M], F)
                - rbf (Tensor): Radial basis features of shape ([M], F)
                - sbf (Tensor): Spherical basis features of shape ([K], F)
                - angle_index (Tensor): Angle indices referring to two edges of shape (2, [K])

        Returns:
            Tensor: Updated edge embeddings.
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
        x_kj = self.lay_mult2([x_kj, sbf], **kwargs)

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
            if x in conf_dense:
                config.update({x: conf_dense[x]})
        return config


class DimNetOutputBlock(Layer):
    """DimNetPP Output Block as defined by `DimNetPP <https://arxiv.org/abs/2011.14115>`__ .

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
                 activation='swish',
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

        self.dense_rbf = Dense(emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.up_projection = Dense(out_emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.dense_mlp = GraphMLP([out_emb_size] * num_dense, activation=activation,
                                  kernel_initializer=kernel_initializer, use_bias=use_bias, **kernel_args)
        self.dimnet_mult = Multiply()
        self.pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.dense_final = Dense(num_targets, use_bias=False, kernel_initializer=output_kernel_initializer,
                                 **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetOutputBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index, state]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - rbf (Tensor): Edge distance basis of shape ([M], F)
                - tensor_index (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Updated node embeddings of shape ([N], F_T).
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
            if x in conf_mlp:
                config.update({x: conf_mlp[x][0]})
        conf_dense_output = self.dense_final.get_config()
        config.update({"output_kernel_initializer": conf_dense_output["kernel_initializer"]})
        config.update({"pooling_method": self.pooling_method, "use_bias": self.use_bias})
        config.update({"emb_size": self.emb_size, "out_emb_size": self.out_emb_size, "num_dense": self.num_dense,
                       "num_targets": self.num_targets})
        return config


class EmbeddingDimeBlock(Layer):
    """Custom Embedding Block of `DimNetPP <https://arxiv.org/abs/2011.14115>`__ .

    Naming of inputs here should match keras Embedding layer.

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
        self.embeddings_initializer = ks.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = ks.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = ks.constraints.get(embeddings_constraint)

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
        out = ops.take(self.embeddings, inputs, axis=0)
        return out

    def get_config(self):
        config = super(EmbeddingDimeBlock, self).get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim,
                       "embeddings_initializer": ks.initializers.serialize(self.embeddings_initializer),
                       "embeddings_regularizer": ks.regularizers.serialize(self.embeddings_regularizer),
                       "embeddings_constraint": ks.constraints.serialize(self.embeddings_constraint)
                       })
        return config
