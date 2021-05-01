import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition


class ChangeTensorType(ks.layers.Layer):
    """
    Layer to change graph representation.

    Args:
        partition_type (str): Partition tensor type. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        output_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
    """
    def __init__(self,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 output_tensor_type="ragged",
                 **kwargs):
        """Initialize layer."""
        super(ChangeTensorType, self).__init__(**kwargs)
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.output_tensor_type = output_tensor_type

    def build(self, input_shape):
        """Build layer."""
        super(ChangeTensorType, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: Graph tensor-information.

        Returns:
            outputs: Changed tensor-information.
        """
        return kgcnn_ops_dyn_cast(inputs, input_tensor_type=self.input_tensor_type,
                                  output_tensor_type=self.output_tensor_type,
                                  partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(ChangeTensorType, self).get_config()
        config.update({"partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "output_tensor_type": self.output_tensor_type
                       })
        return config


class ChangeIndexing(ks.layers.Layer):
    """
    Shift the index for index-tensors to assign nodes in a disjoint graph from single batched graph
    representation or vice-versa.
    
    Example: 
        Flatten operation changes index tensor as [[0,1,2],[0,1],[0,1]] -> [0,1,2,0,1,0,1] with
        requires a subsequent index-shift of [0,1,2,1,1,0,1] -> [0,1,2,3+0,3+1,5+0,5+1].
        This is equivalent to a single graph with disconnected sub-graphs.
        Therefore tf.gather will find the correct nodes for a 1D tensor.
    
    Args:
        to_indexing (str): The index refer to the overall 'batch' or to single 'sample'.
                           The disjoint representation assigns nodes within the 'batch'.
                           It changes "sample" to "batch" or "batch" to "sample."
                           Default is 'batch'.
        from_indexing (str): Index convention that has been set for the input.
                             Default is 'sample'.
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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]

        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(ChangeIndexing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs (list): [nodes, edge_index]

            - nodes: Node features of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2).
            
        Returns:
            edge_index: Corrected edge indices.
        """
        if self.input_tensor_type == "values_partition":
            [_, part_node], [edge_index, part_edge] = inputs

            indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, part_node, part_edge,
                                                                               partition_type_node=self.partition_type,
                                                                               partition_type_edge=self.partition_type,
                                                                               from_indexing=self.from_indexing,
                                                                               to_indexing=self.to_indexing
                                                                               )

            return [indexlist, part_edge]

        elif self.input_tensor_type == "ragged":
            nod, edge_index = inputs
            indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index.values,
                                                                               nod.row_splits,
                                                                               edge_index.value_rowids(),
                                                                               partition_type_node="row_splits",
                                                                               partition_type_edge="value_rowids",
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
