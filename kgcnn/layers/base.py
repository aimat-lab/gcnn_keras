import tensorflow as tf

import kgcnn.ops.activ


class GraphBaseLayer(tf.keras.layers.Layer):
    r"""Base layer for graph layers used in :obj:`kgcnn` that holds some additional information about the graph, which
    could improve performance, if set differently.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
        input_tensor_type (str): Tensor input type. Default is "RaggedTensor".
        output_tensor_type (str): Tensor output type. Default is "RaggedTensor".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        is_directed (bool): If the graph can be considered directed. Default is True.
            This parameter is not used atm but can be useful for expensive edge computation.
    """

    def __init__(self,
                 node_indexing="sample",
                 partition_type="row_length",
                 input_tensor_type="RaggedTensor",
                 output_tensor_type=None,
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 is_directed=True,
                 **kwargs):
        """Initialize layer."""
        super(GraphBaseLayer, self).__init__(**kwargs)
        self.is_directed = is_directed
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.output_tensor_type = input_tensor_type if output_tensor_type is None else output_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._supports_ragged_inputs = True
        self._kgcnn_info = {"node_indexing": self.node_indexing, "partition_type": self.partition_type,
                            "input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate,
                            "is_sorted": self.is_sorted, "has_unconnected": self.has_unconnected,
                            "output_tensor_type": self.output_tensor_type, "is_directed": self.is_directed}

        if self.node_indexing != "sample":
            raise ValueError("Indexing for disjoint representation is not supported as of version 1.0")

    def get_config(self):
        config = super(GraphBaseLayer, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "output_tensor_type": self.output_tensor_type,
                       "is_directed": self.is_directed
                       })
        return config

    def _build_input_shape_check(self, input_shape, input_position):
        """Simply intercept of input type check."""
        if isinstance(input_shape, tf.TensorShape):
            if input_shape[-1] is None:
                print("WARNING:kgcnn: Layer {0} has undefined inner dimension {1} for input {2}".format(
                    self.name, input_shape, input_position))

    def build(self, input_shape):
        """Build base layer."""
        super(GraphBaseLayer, self).build(input_shape)
        if isinstance(input_shape, (list, tuple)):
            for i, ips in enumerate(input_shape):
                self._build_input_shape_check(ips, i)
        else:
            self._build_input_shape_check(input_shape, 0)


class KerasWrapperBase(GraphBaseLayer):
    r"""Base layer for wrapping tf.keras.layers to support ragged tensors and to optionally call original layer
    only on the values of :obj:`RaggedTensor`. If inputs is a list, then a lazy operation (e.g. add, concat)
    is performed if :obj:`ragged_validate` is set to :obj:`False` on the values too.
    """

    def __init__(self, **kwargs):
        r"""Initialize instance of :obj:`KerasWrapperBase`"""
        super(KerasWrapperBase, self).__init__(**kwargs)
        self._kgcnn_wrapper_layer = None
        self._kgcnn_wrapper_args = None

    def build(self, input_shape):
        """Build layer."""
        super(KerasWrapperBase, self).build(input_shape)

    def get_config(self):
        """Make config from wrapped keras layer,"""
        config = super(KerasWrapperBase, self).get_config()
        # Only necessary if this instance has a keras layer internally as attribute.
        if hasattr(self, "_kgcnn_wrapper_layer") and hasattr(self, "_kgcnn_wrapper_args"):
            if self._kgcnn_wrapper_layer is not None:
                layer_conf = self._kgcnn_wrapper_layer.get_config()
                for x in self._kgcnn_wrapper_args:
                    if x in layer_conf:
                        config.update({x: layer_conf[x]})
        return config

    def call(self, inputs, **kwargs):
        """Call on values of the ragged input or fall back to keras layer.

        Args:
            inputs (tf.RaggedTensor, list): Ragged tensor of list of tensors to call wrapped layer on.

        Returns:
            tf.RaggedTensor: Output of keras layer.
        """
        return self._kgcnn_wrapper_layer.call(inputs, **kwargs)