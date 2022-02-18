import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazyAdd


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ReadoutAlignAttendPooling')
class ReadoutAlignAttendPooling(GraphBaseLayer):
    r"""Computes readout or fingerprint generation according to `HamNet <https://arxiv.org/abs/2105.03688>`_ .



    Args:
        units (int): Units for the linear trafo of node features before attention.
        activation (str): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}},
        use_bias (bool): Use bias. Default is True.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units,
                 activation='kgcnn>leaky_relu',
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_dropout=False,
                 rate=None, noise_shape=None, seed=None,
                 **kwargs):
        """Initialize layer."""
        super(ReadoutAlignAttendPooling, self).__init__(**kwargs)
        self.units = int(units)
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}



    def build(self, input_shape):
        """Build layer."""
        super(ReadoutAlignAttendPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled node attentions of shape (batch, F)
        """
        self.assert_ragged_input_rank(inputs)


        return

    def get_config(self):
        """Update layer config."""
        config = super(ReadoutAlignAttendPooling, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units, "has_self_loops": self.has_self_loops,
                       "use_final_activation": self.use_final_activation})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        return config
