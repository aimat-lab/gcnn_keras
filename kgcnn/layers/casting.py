import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.casting import kgcnn_ops_cast_ragged_to_value_partition, kgcnn_ops_cast_masked_to_value_partition, \
    kgcnn_ops_cast_tensor_to_value_partition, kgcnn_ops_cast_value_partition_to_tensor, \
    kgcnn_ops_cast_value_partition_to_masked, kgcnn_ops_cast_value_partition_to_ragged
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition


class CastRaggedToTensor(tf.keras.layers.Layer):
    """
    Layer to cast a ragged tensor to a dense tensor.

    Args:
        **kwargs
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastRaggedToTensor, self).__init__(**kwargs)
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToTensor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            tf.ragged: Feature ragged tensor of shape e.g. (batch,None,F)

        Returns:
            tf.tensor: Input.to_tensor() with zero padding.
        """
        out = inputs.to_tensor()
        return out


class CastRaggedToValuesPartition(ks.layers.Layer):
    """
    Cast a ragged tensor with one ragged dimension, like node feature list to a single value plus partition tensor.
    This matches the disjoint graph representation.
    
    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        input_tensor_type (str): Input type of the tensor for call(). Default is "ragged".
        **kwargs
    """

    def __init__(self, partition_type="row_length", input_tensor_type="ragged", **kwargs):
        """Initialize layer."""
        super(CastRaggedToValuesPartition, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToValuesPartition, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass.

        Args:
            inputs (tf.ragged): Ragged tensor of shape (batch,None,F),
                                  where None is the number of nodes or edges in each graph and
                                  F denotes the feature dimension.
            **kwargs
    
        Returns:
            list: [values, value_partition]
            
            - values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids,
              row_splits etc. Yields the assignment of nodes/edges per graph. Default is row_length.
        """

        out = kgcnn_ops_cast_ragged_to_value_partition(inputs, self.partition_type)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastRaggedToValuesPartition, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config


class CastMaskedToValuesPartition(ks.layers.Layer):
    """
    Cast a zero-padded tensor plus mask input to a single list plus row_partition tensor.
    This matches the disjoint graph representation.
    
    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "masked".
        **kwargs
    """

    def __init__(self, partition_type="row_length", input_tensor_type="masked", **kwargs):
        """Initialize layer."""
        super(CastMaskedToValuesPartition, self).__init__(**kwargs)
        self.partition_type = partition_type
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(CastMaskedToValuesPartition, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [padded_values, mask]

            - padded_values (tf.tensor): Zero padded feature tensor of shape (batch,N,F).
              where F denotes the feature dimension and N the maximum
              number of edges/nodes in graph.
            - mask (tf.tensor): Boolean mask of shape (batch,N),
              where N is the maximum number of nodes or edges.
        
        Returns:
            list: [values, value_partition] 
            
            - values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
              The output shape is given (batch[Mask],F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids,
              row_splits etc.
              Yields the assignment of nodes/edges per graph in batch. Default is row_length.
        """
        out = kgcnn_ops_cast_masked_to_value_partition(inputs, self.partition_type)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastMaskedToValuesPartition, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config


class CastTensorToValuesPartition(ks.layers.Layer):
    """
    Layer to squeeze the batch dimension to match the disjoint representation. Simply flattens out the batch-dimension.
    Important: For graphs of the same size in batch!

    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "tensor".
        **kwargs    
    """

    def __init__(self, partition_type="row_length", input_tensor_type="tensor", **kwargs):
        """Make layer."""
        super(CastTensorToValuesPartition, self).__init__(**kwargs)
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(CastTensorToValuesPartition, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.
        
        Inputs tf.tensor values.
            
        Args: 
            inputs (tf.tensor): Feature tensor with explicit batch dimension of shape (batch,N,F)
        
        Returns:
            list: [values, value_partition] 
            
            - values (tf.tensor): Feature tensor of flatten batch dimension with shape (batch*None,F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids,
              row_splits etc.
              Yields the assignment of nodes/edges per graph in batch. Default is row_length.
        """
        out = kgcnn_ops_cast_tensor_to_value_partition(inputs, self.partition_type)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastTensorToValuesPartition, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config


class CastValuesPartitionToTensor(ks.layers.Layer):
    """
    Add batch-dimension according to row_partition. Reverse the disjoint representation.
    Important: For graphs of the same size in batch!!!

    Args:
        partition_type (str): Partition tensor type for output. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "values_partition".
        **kwargs
    """

    def __init__(self, partition_type="row_length", input_tensor_type="values_partition", **kwargs):
        """Initialize layer."""
        super(CastValuesPartitionToTensor, self).__init__(**kwargs)
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(CastValuesPartitionToTensor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [values, value_partition]

            - values (tf.tensor): Flatten feature tensor of shape (batch*N,F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes/edges per graph in batch. Default is row_length.
                                      
        Returns:
            features (tf.tensor): Feature tensor of shape (batch,N,F).
            The first and second dimensions is reshaped according to a reference tensor.
            F denotes the feature dimension. Requires graphs of identical size in batch.
        """
        out = kgcnn_ops_cast_value_partition_to_tensor(inputs, self.partition_type)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastValuesPartitionToTensor, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config


class CastValuesPartitionToMasked(ks.layers.Layer):
    """
    Layer to add zero padding for a fixed size tensor having an explicit batch-dimension.
    
    The layer maps disjoint representation to padded tensor plus mask.
    
    Args:
        partition_type (str): Partition tensor type. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "values_partition".
        **kwargs
    """

    def __init__(self, partition_type="row_length", input_tensor_type="values_partition", **kwargs):
        """Initialize layer."""
        super(CastValuesPartitionToMasked, self).__init__(**kwargs)
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(CastValuesPartitionToMasked, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [values, value_partition]

            - values (tf.tensor): Feature tensor with flatten batch dimension of shape (batch*None,F).
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes/edges per graph in batch. Default is row_length.
            
        Returns:
            list: [values, mask]
            
            - values (tf.tensor): Padded feature tensor with shape (batch,N,F)
            - mask (tf.tensor): Boolean mask of shape (batch,N)
        """
        out = kgcnn_ops_cast_value_partition_to_masked(inputs, self.partition_type)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastValuesPartitionToMasked, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config


class CastValuesPartitionToRagged(ks.layers.Layer):
    """
    Layer to make ragged tensor from a flatten value tensor plus row partition tensor.
    
    Args:
        partition_type (str): Partition tensor type. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "values_partition".
        **kwargs
    """

    def __init__(self, partition_type="row_length", input_tensor_type="values_partition", **kwargs):
        """Initialize layer."""
        super(CastValuesPartitionToRagged, self).__init__(**kwargs)
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(CastValuesPartitionToRagged, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [values, value_partition]

            - values (tf.tensor): Feature tensor of nodes/edges of shape (batch*None,F)
              where F stands for the feature dimension and None represents
              the flexible size of the graphs.
            - value_partition (tf.tensor): Row partition tensor. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes/edges per graph in batch. Default is row_length.
            
        Returns:
            features (tf.ragged): A ragged feature tensor of shape (batch,None,F).
        """
        out = kgcnn_ops_cast_value_partition_to_ragged(inputs, self.partition_type)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastValuesPartitionToRagged, self).get_config()
        config.update({"partition_type": self.partition_type})
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
        **kwargs
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

        Args:
            inputs (list): [nodes, edge_index]

            - nodes: Node features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - edge_index: Edge indices features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,2)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, 2)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,2)
              or a singe tensor for equally sized graphs (batch,N,2).
            
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
