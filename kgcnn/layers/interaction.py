import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.keras import Dense, Activation, Add, Multiply, Concatenate
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingWeightedLocalEdges, PoolingGlobalEdges, \
    PoolingNodes
import kgcnn.ops.activ
import kgcnn.ops.initializer
from kgcnn.layers.conv import SchNetCFconv


@tf.keras.utils.register_keras_serializable(package='kgcnn',name='SchNetInteraction')
class SchNetInteraction(GraphBaseLayer):
    """
    Schnet interaction block, which uses the continuous filter convolution from SchNetCFconv.

    Args:
        units (int): Dimension of node embedding. Default is 128.
        cfconv_pool (str): Pooling method information for SchNetCFconv layer. Default is'segment_sum'.
        use_bias (bool): Use bias in last layers. Default is True.
        activation (str): Activation function. Default is 'kgcnn>shifted_softplus'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units=128,
                 cfconv_pool='sum',
                 use_bias=True,
                 activation='kgcnn>shifted_softplus',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)

        self.cfconv_pool = cfconv_pool
        self.use_bias = use_bias
        self.units = units

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        conv_args = {"units": self.units, "use_bias": use_bias, "activation": activation, "cfconv_pool": cfconv_pool}

        # Layers
        self.lay_cfconv = SchNetCFconv(**conv_args, **kernel_args, **self._kgcnn_info)
        self.lay_dense1 = Dense(units=self.units, activation='linear', use_bias=False,
                                **self._kgcnn_info, **kernel_args)
        self.lay_dense2 = Dense(units=self.units, activation=activation, use_bias=self.use_bias,
                                **self._kgcnn_info, **kernel_args)
        self.lay_dense3 = Dense(units=self.units, activation='linear', use_bias=self.use_bias,
                                **self._kgcnn_info, **kernel_args)
        self.lay_add = Add(**self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate node update.

        Args:
            inputs: [nodes, edges, edge_index]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edges: Edge or message embeddings of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)

        Returns:
            node_update: Updated node embeddings.
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(node)
        x = self.lay_cfconv([x, edge, indexlist])
        x = self.lay_dense2(x)
        x = self.lay_dense3(x)
        out = self.lay_add([node, x])
        return out

    def get_config(self):
        config = super(SchNetInteraction, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units, "use_bias": self.use_bias})
        conf_dense = self.lay_dense2.get_config()
        for x in ["activation", "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                  "kernel_constraint", "bias_constraint", "kernel_initializer", "bias_initializer"]:
            config.update({x: conf_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn',name='ResidualLayer')
class ResidualLayer(GraphBaseLayer):
    """Residual Layer as defined by DimNet.

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
        **kwargs:
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

        self.dense_1 = Dense(**dense_args, **self._kgcnn_info)
        self.dense_2 = Dense(**dense_args, **self._kgcnn_info)
        self.add_end = Add(**self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(ResidualLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass

        Args:
            inputs (tf.ragged): Node or edge embedding of shape (batch, [N], F)

        Returns:
            embeddings: Node or edge embedding of shape (batch, [N], F)
        """
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.add_end([inputs, x])
        return x

    def get_config(self):
        config = super(ResidualLayer, self).get_config()
        conf_dense = self.dense_1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias", "units"]:
            config.update({x: conf_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn',name='DimNetInteractionPPBlock')
class DimNetInteractionPPBlock(GraphBaseLayer):
    """DimNetInteractionPPBlock as defined by DimNet.

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
                 activation='kgcnn>swish', # default is 'swish'
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer="kgcnn>glorot_orthogonal", # default is 'glorot_orthogonal'
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
        self.dense_rbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args, **self._kgcnn_info)
        self.dense_rbf2 = Dense(emb_size, use_bias=False, **kernel_args, **self._kgcnn_info)
        self.dense_sbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args, **self._kgcnn_info)
        self.dense_sbf2 = Dense(int_emb_size, use_bias=False, **kernel_args,  **self._kgcnn_info)

        # Dense transformations of input messages
        self.dense_ji = Dense(emb_size, activation=activation, use_bias=True, **kernel_args, **self._kgcnn_info)
        self.dense_kj = Dense(emb_size, activation=activation, use_bias=True, **kernel_args, **self._kgcnn_info)

        # Embedding projections for interaction triplets
        self.down_projection = Dense(int_emb_size, activation=activation, use_bias=False,
                                     **kernel_args, **self._kgcnn_info)
        self.up_projection = Dense(emb_size, activation=activation, use_bias=False, **kernel_args, **self._kgcnn_info)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args, **self._kgcnn_info))
        self.final_before_skip = Dense(emb_size, activation=activation, use_bias=True,
                                       **kernel_args, **self._kgcnn_info)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args, **self._kgcnn_info))

        self.lay_add1 = Add(**self._kgcnn_info)
        self.lay_add2 = Add(**self._kgcnn_info)
        self.lay_mult1 = Multiply(**self._kgcnn_info)
        self.lay_mult2 = Multiply(**self._kgcnn_info)

        self.lay_gather = GatherNodesOutgoing(**self._kgcnn_info)  # Are edges here
        self.lay_pool = PoolingLocalEdges(pooling_method=pooling_method, **self._kgcnn_info)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [edges, rbf, sbf, angle_index]

            - edges (tf.ragged): Edge embeddings of shape (batch, [M], F)
            - rbf (tf.ragged): Radial basis features of shape (batch, [M], F)
            - sbf (tf.ragged): Spherical basis features of shape (batch, [K], F)
            - angle_index (tf.ragged): Angle indices between two edges of shape (batch, [K], 2)

        Returns:
            tf.ragged: Updated edge embeddings.
        """
        x, rbf, sbf, id_expand = inputs

        # Initial transformation
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)

        # Transform via Bessel basis
        rbf = self.dense_rbf1(rbf)
        rbf = self.dense_rbf2(rbf)
        x_kj = self.lay_mult1([x_kj, rbf])

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.down_projection(x_kj)
        x_kj = self.lay_gather([x_kj, id_expand])

        # Transform via 2D spherical basis
        sbf = self.dense_sbf1(sbf)
        sbf = self.dense_sbf2(sbf)
        x_kj = self.lay_mult1([x_kj, sbf])

        # Aggregate interactions and up-project embeddings
        x_kj = self.lay_pool([rbf, x_kj, id_expand])
        x_kj = self.up_projection(x_kj)

        # Transformations before skip connection
        x2 = self.lay_add1([x_ji, x_kj])
        for layer in self.layers_before_skip:
            x2 = layer(x2)
        x2 = self.final_before_skip(x2)

        # Skip connection
        x = self.lay_add2([x, x2])

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x)

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
