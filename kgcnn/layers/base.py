import tensorflow as tf

from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type

class GraphBaseLayer(tf.keras.layers.Layer):
    """
    Base layer for graph layers used in kgcnn that holds some additional information about the graph, which can
    improve performance, if set differently. Also input type check to support different tensor in- and output.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        output_tensor_type (str): Output type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        is_directed (bool): If the graph can be considered directed.
    """

    def __init__(self,
                 node_indexing="sample",
                 partition_type="row_length",
                 input_tensor_type="ragged",
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

        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint",
                                               "tensor", "RaggedTensor", "Tensor"]

        self._tensor_input_type_found = []
        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

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


    def _kgcnn_map_input(self, inputs, num_input):

        return input