import tensorflow as tf

import kgcnn.ops.activ


class GraphBaseLayer(tf.keras.layers.Layer):
    """Base layer for graph layers used in kgcnn that holds some additional information about the graph, which can
    improve performance, if set differently.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
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
        if output_tensor_type is None:
            self.output_tensor_type = input_tensor_type
        else:
            self.output_tensor_type = output_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._supports_ragged_inputs = True

        self._tensor_input_type_known = ["ragged", "tensor", "RaggedTensor",
                                         "Tensor", "Sparse", "SparseTensor"]
        self._kgcnn_static_tensor_input_check()

        # All info as dict
        self._kgcnn_info = {"node_indexing": self.node_indexing, "partition_type": self.partition_type,
                            "input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate,
                            "is_sorted": self.is_sorted, "has_unconnected": self.has_unconnected,
                            "output_tensor_type": self.output_tensor_type, "is_directed": self.is_directed}

        if self.node_indexing != "sample":
            raise ValueError("Indexing for disjoint representation is not supported as of version 1.0")

    def _kgcnn_static_tensor_input_check(self):

        if self.input_tensor_type not in self._tensor_input_type_known:
            raise NotImplementedError("Tensor input type ", self.input_tensor_type,
                                      "is not implemented for", self.name, ", expected one of the following:",
                                      self._tensor_input_type_known)

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
        # Only necessary if has some keras wrapper layer internally
        if hasattr(self, "_kgcnn_wrapper_layer") and hasattr(self, "_kgcnn_wrapper_args"):
            if self._kgcnn_wrapper_layer is not None:
                layer_conf = self._kgcnn_wrapper_layer.get_config()
                for x in self._kgcnn_wrapper_args:
                    if x in layer_conf:
                        config.update({x: layer_conf[x]})
        return config

    def build(self, input_shape):
        super(GraphBaseLayer, self).build(input_shape)
        if isinstance(input_shape, list):
            for i, ips in enumerate(input_shape):
                if isinstance(ips, tf.TensorShape):
                    if ips[-1] is None:
                        print("WARNING:kgcnn: Layer", self.name, "has undefined inner dimension", ips, "for input", i)
        if isinstance(input_shape, tf.TensorShape):
            if input_shape[-1] is None:
                print("WARNING:kgcnn: Layer", self.name, "has undefined inner dimension for input", input_shape)

