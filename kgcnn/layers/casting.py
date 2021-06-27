import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.ragged import partition_from_ragged_tensor_by_name
from kgcnn.layers.base import GraphBaseLayer


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ChangeTensorType')
class ChangeTensorType(GraphBaseLayer):
    """Layer to change graph representation tensor type.

    The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
    The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
    For disjoint representation (values, partition), the node embeddings are given by
    a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
    "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information.

    Args:
        partition_type (str): Partition tensor type. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "RaggedTensor".
        output_tensor_type (str): Input type of the tensors for call(). Default is "RaggedTensor".
    """
    def __init__(self,
                 partition_type="row_length",
                 input_tensor_type="RaggedTensor",
                 output_tensor_type="RaggedTensor",
                 **kwargs):
        """Initialize layer."""
        super(ChangeTensorType, self).__init__(**kwargs)
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.output_tensor_type = output_tensor_type

        if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
            raise ValueError("Input must be RaggedTensor for layer", self.name)

    def build(self, input_shape):
        """Build layer."""
        super(ChangeTensorType, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Graph tensor.

        Returns:
            tensor: Changed tensor type.
        """

        if self.output_tensor_type in ["Tensor", "tensor", "padded", "masked"]:
            return inputs.to_tensor()

        elif self.output_tensor_type in ["disjoint", "row_partition", "nested", "values_partition"]:
            return partition_from_ragged_tensor_by_name(inputs, self.partition_type)

        else:
            raise NotImplementedError("Unsupported output_tensor_type", self.output_tensor_type)

    def get_config(self):
        """Update layer config."""
        config = super(ChangeTensorType, self).get_config()
        config.update({"partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "output_tensor_type": self.output_tensor_type
                       })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ChangeIndexing')
class ChangeIndexing(GraphBaseLayer):
    """Shift the index for index-tensors to assign nodes in a disjoint graph from single batched graph
    representation or vice-versa.
    
    Example: 
        Flatten operation changes index tensor as [[0,1,2],[0,1],[0,1]] -> [0,1,2,0,1,0,1] with
        requires a subsequent index-shift of [0,1,2,1,1,0,1] -> [0,1,2,3+0,3+1,5+0,5+1].
        This is equivalent to a single graph with disconnected sub-graphs.
        Therefore tf.gather will find the correct nodes for a 1D tensor.
    
    Args:
        to_indexing (str): The index refer to the overall 'batch' or to single 'sample'.
            The disjoint representation assigns nodes within the 'batch'.
            It changes "sample" to "batch" or "batch" to "sample." Default is 'batch'.
        from_indexing (str): Index convention that has been set for the input. Default is 'sample'.
        partition_type (str): Partition tensor type. Default is "row_length". Only used for values_partition input.
        input_tensor_type (str): Input type of the tensors for call(). Default is "values_partition".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self,
                 to_indexing='batch',
                 from_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="values_partition",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(ChangeIndexing, self).__init__(**kwargs)
        self.to_indexing = to_indexing
        self.from_indexing = from_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate

        if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
            raise ValueError("Input must be RaggedTensor for layer", self.name)

        self._supports_ragged_inputs = True

        if self.from_indexing != self.node_indexing:
            print("WARNING: Graph layer's node_indexing does not agree with from_indexing", self.node_indexing,
                  "vs.", self.to_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(ChangeIndexing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, tensor_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [N], 2).
            
        Returns:
            tf.RaggedTensor: Corrected edge indices of shape (batch, [N], 2).
        """

        nod, edge_index = inputs
        indexlist = partition_row_indexing(edge_index.values,
                                           nod.row_splits,
                                           edge_index.value_rowids(),
                                           partition_type_target="row_splits",
                                           partition_type_index="value_rowids",
                                           from_indexing=self.from_indexing,
                                           to_indexing=self.to_indexing
                                           )

        out = tf.RaggedTensor.from_row_splits(indexlist, edge_index.row_splits, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(ChangeIndexing, self).get_config()
        config.update({"to_indexing": self.to_indexing,
                       "from_indexing": self.from_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config
