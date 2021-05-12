import tensorflow as tf


def kgcnn_ops_check_tensor_type(inputs, input_tensor_type, node_indexing=None):
    tensor_keys = ["Tensor", "tensor"]
    ragged_keys = ["ragged", "RaggedTensor"]
    value_partition_keys = ["values_partition", "disjoint"]

    if isinstance(inputs, tf.RaggedTensor):
        if input_tensor_type not in ragged_keys:
            print("Warning: Received RaggedTensor but tensor type specified as:", input_tensor_type)
        if node_indexing not in ["sample"]:
            print("Warning: For ragged tensor input, default node_indexing is considered 'sample'.",
                  "This layer will assume node_indexing", node_indexing)
        return ragged_keys[0]

    if isinstance(inputs, list):
        if input_tensor_type not in value_partition_keys:
            print("Warning: Received [values, partition] but tensor type specified as:", input_tensor_type)
        if node_indexing not in ["batch"]:
            print("Warning: For [values, partition] tensor input, default node_indexing is considered 'batch'.",
                  "This layer will assume node_indexing", node_indexing)
        return value_partition_keys[0]

    if isinstance(inputs, tf.Tensor):
        if input_tensor_type not in tensor_keys:
            print("Warning: Received [values, partition] but tensor type specified as:", input_tensor_type)
        if node_indexing not in ["sample"]:
            print("Warning: For tensor input, default node_indexing is considered 'sample'.",
                  "This layer will assume node_indexing", node_indexing)
        return tensor_keys[0]

    else:
        # Default value
        return input_tensor_type


def kgcnn_ops_static_test_tensor_input_type(input_tensor_type, tensor_input_type_implemented, node_indexing=None):
    """Test input tensor type for all layers in init().

    Args:
        input_tensor_type (str): Information on tensor input.
        tensor_input_type_implemented: Information on acceptable input.
        node_indexing: Information on index reference.

    Returns:
        None.
    """

    tensor_keys = ["Tensor", "tensor"]
    ragged_keys = ["ragged", "RaggedTensor"]
    value_partition_keys = ["disjoint", "values_partition"]

    if input_tensor_type not in tensor_input_type_implemented:
        raise NotImplementedError("Error: Tensor input type ", input_tensor_type,
                                  "is not implemented for this layer, choose one of the following:",
                                  tensor_input_type_implemented)
    if node_indexing is not None:
        if input_tensor_type in ragged_keys and node_indexing not in ["sample"]:
            print("Warning: For ragged tensor input, default node_indexing is considered 'sample'.",
                  "This layer will use node_indexing", node_indexing)

        if input_tensor_type in value_partition_keys and node_indexing not in ["batch", "disjoint"]:
            print("Warning: For [values, partition] tensor input, default n"
                  "ode_indexing is considered 'batch'.",
                  "This layer will use node_indexing", node_indexing)

        if input_tensor_type in tensor_keys and node_indexing not in ["sample"]:
            print("Warning: For tensor input, default node_indexing is considered 'sample'.",
                  "This layer will use node_indexing", node_indexing)
