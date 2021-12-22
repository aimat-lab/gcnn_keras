import tensorflow as tf
import kgcnn.ops.activ

# from keras.engine.base_layer import Layer
from tensorflow.keras.layers import Layer


class GraphBaseLayer(Layer):
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
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(GraphBaseLayer, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._supports_ragged_inputs = True
        self._kgcnn_info = {"node_indexing": self.node_indexing,
                            "ragged_validate": self.ragged_validate,
                            "is_sorted": self.is_sorted, "has_unconnected": self.has_unconnected}

        if self.node_indexing != "sample":
            raise ValueError("Indexing for disjoint representation is not supported as of version 1.0")

        self._add_layer_config_to_self = {}

    def get_config(self):
        config = super(GraphBaseLayer, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "ragged_validate": self.ragged_validate,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       })
        # Also add the config of a sub-layer to self.
        # Should only be done if sub-layer does not change config on built.
        for key, value in self._add_layer_config_to_self.items():
            if hasattr(self, key):
                if getattr(self, key) is not None:
                    layer_conf = getattr(self, key).get_config()
                    for x in value:
                        if x in layer_conf:
                            config.update({x: layer_conf[x]})
        return config

    def build(self, input_shape):
        """Build base layer."""
        super(GraphBaseLayer, self).build(input_shape)

    def assert_ragged_input_rank(self, inputs, ragged_rank: int = 1):
        """Assert input to be ragged with a given ragged_rank.

        Args:
            inputs: Tensor or list of tensors to assert to be ragged and have given ragged rank.
            ragged_rank (int): Assert ragged tensor to have ragged_rank. Default is 1.
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

    def call_on_values_tensor_of_ragged(self, fun, inputs, **kwargs):
        r"""This is a helper function that attempts to call :obj:`fun` on the value tensor of :obj:`inputs`.
        For ragged rank of one, the values is a tf.Tensor.
        Function :obj:`inputs` must be a ragged tensors with ragged rank of one, or a list of ragged tensors.
        The fallback is to call :obj:`fun` directly on inputs.
        For list input assumes lazy operation if :obj:`ragged_validate` is set to :obj:`False`, which means it is not
        checked if splits are equal.
        For the output of :obj:`fun` it is always assumed that the ragged partition does not change in :obj:`fun`.
        If `axis` is found in kwargs, the axis argument of adapted for the values tensor if possible, otherwise
        the tensor is passed as fallback to :obj:`fun` directly.

        Args:
            fun (callable): Callable function that accepts inputs and kwargs.
            inputs (tf.RaggedTensor, list): Tensor input or list of tensors.
            kwargs: Additional kwargs for fun.

        Returns:
            tf.RaggedTensor: Output of fun only on the values tensor of the ragged input.
        """
        if "axis" in kwargs:
            axis = kwargs["axis"]
            axis_values = None
            kwargs_values = None
            if isinstance(axis, int):
                if axis > 1:
                    axis_values = axis - 1
            elif isinstance(axis, (list, tuple)):
                if all([x > 1 for x in axis]):
                    axis_values = [x - 1 for x in axis]
            if axis_values is not None:
                kwargs_values = {key: value for key, value in kwargs.items()}
                kwargs_values.pop("axis")
                kwargs_values.update({"axis": axis_values})
        else:
            kwargs_values = {key: value for key, value in kwargs.items()}

        if isinstance(inputs, list) and kwargs_values is not None:
            if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
                if all([x.ragged_rank == 1 for x in inputs]) and not self.ragged_validate:
                    out = fun([x.values for x in inputs], **kwargs_values)
                    if isinstance(out, list):
                        return [tf.RaggedTensor.from_row_splits(x, inputs[i].row_splits, validate=self.ragged_validate)
                                for i, x in enumerate(out)]
                    else:
                        return tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
        elif isinstance(inputs, tf.RaggedTensor) and kwargs_values is not None:
            if inputs.ragged_rank == 1:
                out = fun(inputs.values, **kwargs_values)
                if isinstance(out, list):
                    return [tf.RaggedTensor.from_row_splits(x, inputs.row_splits, validate=self.ragged_validate) for x
                            in out]
                else:
                    return tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)

        if isinstance(inputs, tf.RaggedTensor):
            print("WARNING: Layer %s fail call on value Tensor of ragged Tensor." % self.name)
        if isinstance(inputs, list):
            if any([isinstance(x, tf.RaggedTensor) for x in inputs]):
                print("WARNING: Layer %s fail call on value Tensor for ragged Tensor in list." % self.name)
        return fun(inputs, **kwargs)
