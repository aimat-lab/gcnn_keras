import tensorflow as tf



def kgcnn_ops_static_test_tensor_input_type(input_tensor_type, tensor_input_type_implemented, node_indexing=None):
    """Test input tensor type for all layers in init().

    Args:
        input_tensor_type (str): Information on tensor input.
        tensor_input_type_implemented: Information on acceptable input.
        node_indexing: Information on index reference.

    Returns:
        None.
    """
    if input_tensor_type not in tensor_input_type_implemented:
        raise NotImplementedError("Error: Tensor input type ", input_tensor_type,
                                  "is not implemented for this layer, choose one of the following:",
                                  tensor_input_type_implemented)
    if node_indexing is not None:
        if input_tensor_type in ["ragged", "RaggedTensor"] and node_indexing not in ["sample"]:
            print("Warning: For ragged tensor input, default node_indexing is considered 'sample'. ",
                  "This layer will use node_indexing", node_indexing)

        if input_tensor_type in ["values_partition", "disjoint"] and node_indexing not in ["batch", "disjoint"]:
            print("Warning: For [values, partition] tensor input, default n"
                  "ode_indexing is considered 'batch'.",
                  "This layer will use node_indexing", node_indexing)

        if input_tensor_type in ["tensor", "Tensor"] and node_indexing not in ["sample"]:
            print("Warning: For tensor input, default node_indexing is considered 'sample'.",
                  "This layer will use node_indexing", node_indexing)