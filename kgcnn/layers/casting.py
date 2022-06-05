import tensorflow as tf
from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.ragged import partition_from_ragged_tensor_by_name
from kgcnn.layers.base import GraphBaseLayer
# import tensorflow.keras as ks
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='ChangeTensorType')
class ChangeTensorType(GraphBaseLayer):
    r"""Layer to change the ragged tensor representation into tensor type information.

    The tensor representation can be :obj:`tf.RaggedTensor`, :obj:`tf.Tensor` or a tuple of (values, partition).
    For example, the :obj:`RaggedTensor` has shape `(batch, None, F)`.
    The dense tensor in case of equal sized graphs or zero padded graphs will have shape `(batch, N, F)`.
    For disjoint representation (values, partition), the embeddings are given by
    a flattened value :obj:`tf.Tensor` of shape `(batch*N, F)` and a partition tensor of either 'row_length',
    'row_splits' or 'value_rowids'. This matches the :obj:`tf.RaggedTensor` partition information for a ragged rank of
    one, which effectively keeps the batch assignment.

    """

    def __init__(self,
                 partition_type: str = "row_length",
                 input_tensor_type: str = "RaggedTensor",
                 output_tensor_type: str = "RaggedTensor",
                 **kwargs):
        r"""Initialize layer.

        Args:
            partition_type (str): Partition tensor type. Default is "row_length".
            input_tensor_type (str): Input type of the tensors for :obj:`call`. Default is "RaggedTensor".
            output_tensor_type (str): Output type of the tensors for :obj:`call`. Default is "RaggedTensor".
        """
        super(ChangeTensorType, self).__init__(**kwargs)
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.output_tensor_type = output_tensor_type

        if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
            raise ValueError("Input must be RaggedTensor for layer %s" % self.name)

    def build(self, input_shape):
        """Build layer."""
        super(ChangeTensorType, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Ragged tensor.

        Returns:
            tensor: Changed tensor type.
        """
        self.assert_ragged_input_rank(inputs)

        if self.output_tensor_type in ["Tensor", "tensor", "padded", "masked"]:
            return inputs.to_tensor()
        elif self.output_tensor_type in ["ragged", "RaggedTensor"]:
            return inputs  # Nothing to do here.
        elif self.output_tensor_type in ["disjoint", "row_partition", "nested", "values_partition", "values"]:
            return partition_from_ragged_tensor_by_name(inputs, self.partition_type)
        else:
            raise NotImplementedError("Unsupported output_tensor_type %s" % self.output_tensor_type)

    def get_config(self):
        """Update layer config."""
        config = super(ChangeTensorType, self).get_config()
        config.update({"partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "output_tensor_type": self.output_tensor_type
                       })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='ChangeIndexing')
class ChangeIndexing(GraphBaseLayer):
    r"""Shift the indices for index-tensors to reference within the batch or each sample separately or vice-versa.

    This can be interpreted as a per-graph assignment of batched graph representations versus a global disjoint graph
    representation with unconnected sub-graphs. A ragged index tensor would have indices referencing each sample.
    If the batch-dimension was dissolved for target and index tensor, indices must be corrected.

    For example, a flatten operation changes an index tensor as `[[0, 1, 2], [0, 1], [0, 1]]` to `[0, 1, 2, 0, 1, 0, 1]`
    with requires a subsequent index-shift of `[0, 1, 2, 1, 1, 0, 1]` to `[0, 1, 2, 3+0, 3+1, 5+0, 5+1]` equals
    `[0, 1, 2, 3, 4, 5, 6]`, so that correct indexing is preserved if batch dimension is dissolved.

    This layer shifts a ragged index tensor according to a batch partition taken from a ragged target tensor with
    both having a ragged rank of one.

    """

    def __init__(self,
                 to_indexing: str = 'batch',
                 from_indexing: str = 'sample',
                 **kwargs):
        r"""Initialize layer.

        Args:
            to_indexing (str): The index refer to the overall 'batch' or to single 'sample'.
                The disjoint representation assigns nodes within the 'batch'.
                It changes 'sample' to 'batch' or 'batch' to 'sample'. Default is 'batch'.
            from_indexing (str): Index convention that has been set for the input. Default is 'sample'.
        """
        super(ChangeIndexing, self).__init__(**kwargs)
        self.to_indexing = to_indexing
        self.from_indexing = from_indexing

        if hasattr(self, "node_indexing"):
            if self.from_indexing != self.node_indexing:
                raise ValueError("Graph layer's indexing does not agree with `from_indexing` '%s' vs '%s'" % (
                        self.node_indexing, self.to_indexing))

    def build(self, input_shape):
        """Build layer."""
        super(ChangeIndexing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [nodes, index_tensor]

                - nodes (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`
                - index_tensor (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [N], 2)`.
            
        Returns:
            tf.RaggedTensor: Edge indices with modified reference of shape `(batch, [N], 2)`.
        """
        self.assert_ragged_input_rank(inputs)
        nodes, edge_index = inputs
        index_list = partition_row_indexing(edge_index.values,
                                            nodes.row_splits,
                                            edge_index.value_rowids(),
                                            partition_type_target="row_splits",
                                            partition_type_index="value_rowids",
                                            from_indexing=self.from_indexing,
                                            to_indexing=self.to_indexing)

        out = tf.RaggedTensor.from_row_splits(index_list, edge_index.row_splits, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(ChangeIndexing, self).get_config()
        config.update({"to_indexing": self.to_indexing,
                       "from_indexing": self.from_indexing})
        return config
