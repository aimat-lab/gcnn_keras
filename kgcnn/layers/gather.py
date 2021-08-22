import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.axis import get_positive_axis


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GatherNodes')
class GatherNodes(GraphBaseLayer):
    """Gather nodes by indices, e.g. that define an edge.

    An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    """

    def __init__(self,
                 axis=1,
                 concat_axis=2,
                 **kwargs):
        """Initialize layer."""
        super(GatherNodes, self).__init__(**kwargs)
        self.concat_axis = concat_axis
        self.axis = axis

    def build(self, input_shape):
        super(GatherNodes, self).build(input_shape)
        if len(input_shape) != 2:
            print("WARNING: Number of inputs for layer", self.name, "is expected to be 2.")
        if [i for i in range(len(input_shape[0]))][self.axis] == 0:
            print("WARNING: Shape error for", self.name, ", gather from batch-dimension is not intended.")

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, tensor_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Gathered node embeddings that match the number of edges of shape (batch, [M], 2*F)
        """
        # The primary case for aggregation of nodes from node feature list. Case from doc-string.
        # Faster implementation via values and indices shifted by row-partition. Equal to disjoint implementation.
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
            if all([x.ragged_rank == 1 for x in inputs]) and self.axis == 1 and self.concat_axis in [None, 2]:
                node, node_part = inputs[0].values, inputs[0].row_splits
                edge_index, edge_part = inputs[1].values, inputs[1].row_lengths()
                disjoint_list = partition_row_indexing(edge_index, node_part, edge_part,
                                                       partition_type_target="row_splits",
                                                       partition_type_index="row_length", to_indexing='batch',
                                                       from_indexing=self.node_indexing)
                out = tf.gather(node, disjoint_list, axis=0)
                if self.concat_axis == 2:
                    out = tf.keras.backend.concatenate([out[:, i] for i in range(edge_index.shape[-1])], axis=1)
                out = tf.RaggedTensor.from_row_lengths(out, edge_part, validate=self.ragged_validate)
                return out

        # For arbitrary gather from ragged tensor use tf.gather with batch_dims=1.
        # Works in tf.__version__>=2.4
        out = tf.gather(inputs[0], inputs[1], batch_dims=1, axis=self.axis)
        if self.concat_axis is not None:
            out = tf.concat([tf.gather(out, i, axis=self.concat_axis) for i in range(out.shape[self.concat_axis])],
                            axis=self.concat_axis)
        return out

    def get_config(self):
        config = super(GatherNodes, self).get_config()
        config.update({"concat_axis": self.concat_axis, "axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GatherNodesSelection')
class GatherNodesSelection(GraphBaseLayer):
    """Gather nodes with a defined index. E.g. for ingoing or outgoing nodes or angles.

    For ingoing nodes, layer uses only index[0]. An edge is defined by index tuple (i,j) with i<-j connection.
    """

    def __init__(self, selection_index, axis=1, axis_indices=2, **kwargs):
        """Initialize layer."""
        super(GatherNodesSelection, self).__init__(**kwargs)
        self.axis = axis
        self.axis_indices = axis_indices

        if not isinstance(selection_index, (list, tuple, int)):
            raise ValueError("Indices for selection must be list or tuple for layer `GatherNodesSelection`.")

        if isinstance(selection_index, int):
            self.selection_index = [selection_index]
        else:
            self.selection_index = list(selection_index)

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesSelection, self).build(input_shape)
        if len(input_shape) != 2:
            print("WARNING: Number of inputs for layer", self.name, "is expected to be 2.")
        if [i for i in range(len(input_shape[0]))][self.axis] == 0:
            print("WARNING: Shape error for", self.name, ", gather from batch-dimension is not intended.")

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, tensor_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            list: Gathered node embeddings that match the number of edges of shape (batch, [M], F) for selection_index.
        """
        # The primary case for aggregation of nodes from node feature list. Case from doc-string.
        # Faster implementation via values and indices shifted by row-partition. Equal to disjoint implementation.
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
            if all([x.ragged_rank == 1 for x in inputs]) and self.axis == 1 and self.axis_indices == 2:
                # We cast to values here
                node, node_part = inputs[0].values, inputs[0].row_splits
                edge_index, edge_part = inputs[1].values, inputs[1].row_lengths()
                indexlist = partition_row_indexing(edge_index, node_part, edge_part,
                                                   partition_type_target="row_splits",
                                                   partition_type_index="row_length",
                                                   to_indexing='batch',
                                                   from_indexing=self.node_indexing)
                out = [tf.gather(node, tf.gather(indexlist, i, axis=1), axis=0) for i in self.selection_index]
                out = [tf.RaggedTensor.from_row_lengths(x, edge_part, validate=self.ragged_validate) for x in out]
                return out

        # For arbitrary gather from ragged tensor use tf.gather with batch_dims=1.
        out = [tf.gather(inputs[0], tf.gather(inputs[1], i, axis=self.axis_indices), batch_dims=1, axis=self.axis) for i
               in self.selection_index]  # Works in tf.__version__>=2.4
        return out

    def get_config(self):
        config = super(GatherNodesSelection, self).get_config()
        config.update({"axis": self.axis, "axis_indices": self.axis_indices, "selection_index": self.selection_index})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GatherNodesOutgoing')
class GatherNodesOutgoing(GatherNodesSelection):
    """Gather nodes by indices.

    For outgoing nodes, layer uses only index[1]. An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    """

    def __init__(self, selection_index=1, **kwargs):
        super(GatherNodesOutgoing, self).__init__(selection_index=selection_index, **kwargs)

    def call(self, inputs, **kwargs):
        return super(GatherNodesOutgoing, self).call(inputs, **kwargs)[0]


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GatherNodesIngoing')
class GatherNodesIngoing(GatherNodesSelection):
    """Gather nodes by edge edge_indices.
    
    For ingoing nodes, layer uses only index[0]. An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    """

    def __init__(self, selection_index=0, **kwargs):
        super(GatherNodesIngoing, self).__init__(selection_index=selection_index, **kwargs)

    def call(self, inputs, **kwargs):
        return super(GatherNodesIngoing, self).call(inputs, **kwargs)[0]


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GatherState')
class GatherState(GraphBaseLayer):
    """Layer to repeat environment or global state for node or edge lists.
    
    To repeat the correct environment for each sample, a tensor with the target length/partition is required.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [state, target]

                - state (tf.Tensor): Graph specific embedding tensor. This is tensor of shape (batch, F)
                - target (tf.RaggedTensor): Target to collect state for, of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Graph embedding with repeated single state for each graph of shape (batch, [N], F).
        """
        dyn_inputs = inputs[1]

        # We cast to values here
        env = inputs[0]
        target_len = dyn_inputs.row_lengths()

        out = tf.repeat(env, target_len, axis=0)
        out = tf.RaggedTensor.from_row_lengths(out, target_len, validate=self.ragged_validate)
        return out

    def get_config(self):
        config = super(GatherState, self).get_config()
        return config
