import tensorflow as tf

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_partition_type
from kgcnn.ops.ragged import DummyRankOneRaggedTensor
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
        is_directed (bool): If the graph can be considered directed. Default is True.
            This parameter is not used atm but can be useful for expensive edge computation.
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

        self._all_kgcnn_info = {"node_indexing": self.node_indexing, "partition_type": self.partition_type,
                                "input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate,
                                "is_sorted": self.is_sorted, "has_unconnected": self.has_unconnected,
                                "output_tensor_type": self.output_tensor_type, "is_directed": self.is_directed}

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

    def _kgcnn_map_input_ragged(self, inputs, num_input):
        """Flexible input tensor check.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition) as in a
        disjoint graph representation where partition holds the graph ids.
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs (list): List of inputs. Must always be a list.
            num_input (int): Number of tensor-like objects in inputs.

        Returns:
            list: Mapped to a ragged-like input of inputs.
        """
        out_inputs = []
        # tensor_keys = ["Tensor", "tensor"]
        ragged_keys = ["ragged", "RaggedTensor"]
        value_partition_keys = ["disjoint", "values_partition"]

        for x in inputs:
            if isinstance(x, tf.RaggedTensor):
                if self.input_tensor_type not in ragged_keys:
                    print("Warning:", self.name, "received RaggedTensor but tensor type specified as:",
                          self.input_tensor_type)
                if self.node_indexing not in ["sample"]:
                    print("Warning: For ragged tensor input, default indexing scheme is considered 'sample'.",
                          "Layer", self.name, "will assume node_indexing", self.node_indexing)
                out_inputs.append(x)
                self._tensor_input_type_found.append("ragged")

            elif isinstance(x, list):
                if self.input_tensor_type not in value_partition_keys:
                    print("Warning:", self.name, "received received [values, partition] but tensor type specified as:",
                          self.input_tensor_type)
                if self.node_indexing not in ["batch"]:
                    print("Warning: For [values, partition] input, default node_indexing is considered 'batch'.",
                          "Layer", self.name, "will assume node_indexing", self.node_indexing)
                self._tensor_input_type_found.append("values_partition")
                if len(x) != 2:
                    print("Warning:", self.name, "input does not match rank=1 partition scheme for batch dimension.")
                # Here partition type must be known
                dummy_tensor = DummyRankOneRaggedTensor()
                dummy_tensor.from_values_partition(x[0], x[1], self.partition_type)
                out_inputs.append(dummy_tensor)

            else:
                # Default value
                raise TypeError("Error:", self.name, "input type for ragged-like input is not supported for", x)

        return out_inputs

    def _kgcnn_map_output_ragged(self, inputs, input_partition_type, output_tensor_type=None):
        x = inputs
        ragged_keys = ["ragged", "RaggedTensor"]
        value_partition_keys = ["disjoint", "values_partition"]

        if output_tensor_type is None:
            output_tensor_type = self.output_tensor_type
        elif isinstance(output_tensor_type, int):
            output_tensor_type = self._tensor_input_type_found[output_tensor_type]

        if isinstance(x, tf.RaggedTensor):
            if output_tensor_type in ragged_keys:
                return x
            elif output_tensor_type in value_partition_keys:
                return kgcnn_ops_dyn_cast(x, input_tensor_type="ragged",
                                          output_tensor_type=output_tensor_type, partition_type=self.partition_type)
            else:
                raise TypeError("Error:", self.name, "output type for ragged-like input is not supported for", x)

        elif isinstance(x, list):
            if len(x) != 2:
                print("Warning:", self.name, "output does not match rank=1 partition scheme for batch dimension.")
            if output_tensor_type in ragged_keys:
                return kgcnn_ops_dyn_cast(x, input_tensor_type="values_partition",
                                          output_tensor_type=output_tensor_type, partition_type=input_partition_type)
            elif output_tensor_type in value_partition_keys:
                tens_part = kgcnn_ops_change_partition_type(x[1], input_partition_type, self.partition_type)
                return [x[0], tens_part]
            else:
                raise TypeError("Error:", self.name, "output type for ragged-like input is not supported for", x)
        else:
            raise TypeError("Error:", self.name, "input type for ragged-like input is not supported for", x)







class KerasLayerWrapperBase(tf.keras.layers.Layer):
    """Base layer for keras wrapper in kgcnn that allows for ragged input type.

    Args:
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        output_tensor_type (str): Output type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 output_tensor_type=None,
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(KerasLayerWrapperBase, self).__init__(**kwargs)

        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        if output_tensor_type is None:
            self.output_tensor_type = input_tensor_type
        else:
            self.output_tensor_type = output_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True

        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint",
                                               "tensor", "RaggedTensor", "Tensor"]

        self._kgcnn_wrapper_call_type = 0
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = None

    def call(self, inputs, **kwargs):
        if self._kgcnn_wrapper_call_type == 0:
            # get a single tensor
            if isinstance(inputs, tf.RaggedTensor):
                if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                    print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
                # Can work already with RaggedTensor
                # out = self._kgcnn_wrapper_layer(inputs, **kwargs)
                # We need a check of ragged rank here and default to standard layer
                value_tensor = inputs.values
                out_tensor = self._kgcnn_wrapper_layer(value_tensor, **kwargs)
                return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
            elif isinstance(inputs, list):
                if self.input_tensor_type not in ["disjoint", "values_partition"]:
                    print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
                out = self._kgcnn_wrapper_layer(inputs[0], **kwargs)
                return [out] + inputs[1:]
            elif isinstance(inputs, tf.Tensor):
                if self.input_tensor_type not in ["Tensor", "tensor"]:
                    print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
                return self._kgcnn_wrapper_layer(inputs, **kwargs)
            else:
                raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)
        elif self._kgcnn_wrapper_call_type == 1:
            # Get a list of tensors
            if isinstance(inputs[0], tf.RaggedTensor):
                if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                    print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
                # Works already with RaggedTensor but much much slower
                # out = self._layer_keras(inputs, **kwargs)
                # We need a check of ragged rank here and default to standard layer
                out = self._kgcnn_wrapper_layer([x.values for x in inputs], **kwargs)
                out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
                return out
            elif isinstance(inputs[0], list):
                if self.input_tensor_type not in ["disjoint", "values_partition"]:
                    print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
                out_part = inputs[0][1:]
                out = self._kgcnn_wrapper_layer([x[0] for x in inputs], **kwargs)
                return [out] + out_part
            elif isinstance(inputs[0], tf.Tensor):
                if self.input_tensor_type not in ["Tensor", "tensor"]:
                    print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
                return self._kgcnn_wrapper_layer(inputs, **kwargs)
            else:
                raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        config = super(KerasLayerWrapperBase, self).get_config()
        config.update({"input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate,
                       "output_tensor_type": self.output_tensor_type})
        if self._kgcnn_wrapper_layer is not None:
            layer_conf = self._kgcnn_wrapper_layer.get_config()
            for x in self._kgcnn_wrapper_args:
                config.update({x: layer_conf[x]})
        return config

