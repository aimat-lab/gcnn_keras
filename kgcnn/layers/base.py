import tensorflow as tf

import kgcnn.ops.activ


class GraphBaseLayer(tf.keras.layers.Layer):
    r"""Base layer for graph layers used in :obj:`kgcnn` that holds some additional information about the graph, which
    could improve performance for some layers, if set differently.

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

        self._add_layer_config_to_self = {}

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
        # Also add to config info of layer to self
        for key, value in self._add_layer_config_to_self.items():
            if hasattr(self, key):
                if getattr(self, key) is not None:
                    layer_conf = getattr(self, key).get_config()
                    for x in value:
                        if x in layer_conf:
                            config.update({x: layer_conf[x]})
        return config

    def _build_input_shape_check(self, input_shape, input_position):
        """Simple intercept of input type check."""
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

    def _assert_ragged_input(self, inputs, ragged_rank: int = 1):
        """Assert input to be ragged with a given ragged_rank.

        Args:
            inputs: Tensor or list of tensors to assert to be ragged and have ragged rank.
            ragged_rank (int): Assert ragged tensor to have ragged_rank = 1.
        """
        if isinstance(inputs, (list, tuple)):
            assert all(
                [isinstance(x, tf.RaggedTensor) for x in inputs]), "%s requires `RaggedTensor` input." % self.name
            if ragged_rank is not None:
                assert all(
                    [x.ragged_rank == ragged_rank for x in inputs]), "%s must have input with ragged_rank=%s." % (
                        self.name, ragged_rank)
        else:
            assert isinstance(inputs, tf.RaggedTensor), "%s requires `RaggedTensor` input." % self.name
            if ragged_rank is not None:
                assert inputs.ragged_rank == ragged_rank, "%s must have input with ragged_rank=%s." % (
                    self.name, ragged_rank)

    def call_on_ragged_values(self, fun, inputs, *args, **kwargs):
        r"""This is a helper function that attempts to call :obj:`fun` on the value tensor of :obj:`inputs`,
        if :obj:`inputs` is a ragged tensors with ragged rank of one, or a list of ragged tensors. The fallback
        is to call fun directly on inputs.
        For list input assumes lazy operation if :obj:`ragged_validate` is set to :obj:`False`, which means it is not
        checked if splits are equal. For the output is always assumed that the ragged partition does not change in
        :obj:`fun`.

        Args:
            fun (callable): Callable function that accepts inputs and kwargs.
            inputs (tf.RaggedTensor, list): Tensor input or list of tensors.
            args: Additional args for fun.
            kwargs: Additional kwargs for fun.

        Returns:
            tf.RaggedTensor: Output of fun only on the values tensor of the ragged input.
        """
        if isinstance(inputs, list):
            if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
                if all([x.ragged_rank == 1 for x in inputs]) and not self.ragged_validate:
                    out = fun([x.values for x in inputs], *args, **kwargs)
                    if isinstance(out, list):
                        return [tf.RaggedTensor.from_row_splits(x, inputs[i].row_splits, validate=self.ragged_validate)
                                for i, x in enumerate(out)]
                    else:
                        return tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
        elif isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                return tf.RaggedTensor.from_row_splits(fun(inputs.values, *args, **kwargs),
                                                       inputs.row_splits, validate=self.ragged_validate)
        else:
            print("Layer %s fail call on ragged values, calling directly on Tensor " % self.name)
        return fun(inputs, *args, **kwargs)
