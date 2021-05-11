import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as ksb

from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition, kgcnn_ops_change_partition_type


@tf.function
def kgcnn_ops_cast_ragged_to_value_partition(inputs, partition_type="row_length"):
    tens = inputs
    flat_tens = tens.values

    if partition_type == "row_length":
        outpart = tens.row_lengths()
    elif partition_type == "row_splits":
        outpart = tens.row_splits
    elif partition_type == "value_rowids":
        outpart = tens.value_rowids()
    else:
        raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")

    return [flat_tens, outpart]

@tf.function
def kgcnn_ops_cast_value_partition_to_ragged(inputs, partition_type="row_length", ragged_validate=False):
    nod, n_part = inputs

    if partition_type == "row_length":
        out = tf.RaggedTensor.from_row_lengths(nod, n_part,validate=ragged_validate)
    elif partition_type == "row_splits":
        out = tf.RaggedTensor.from_row_splits(nod, n_part,validate=ragged_validate)
    elif partition_type == "value_rowids":
        out = tf.RaggedTensor.from_value_rowids(nod, n_part,validate=ragged_validate)
    else:
        raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")

    return out


@tf.function
def kgcnn_ops_cast_masked_to_value_partition(inputs, partition_type="row_length"):
    tens, mask = inputs
    # Ensure mask is of type bool
    mask = ksb.cast(mask, dtype="bool")
    fmask = ksb.cast(mask, dtype="int64")
    row_lengths = ksb.sum(fmask, axis=1)
    # shape of nodematrix
    shape_tens = ksb.shape(tens)
    shape_tens_int = ksb.int_shape(tens)
    # Flatten batch dimension
    batchred_tens = ksb.reshape(tens, (shape_tens[0] * shape_tens[1], shape_tens_int[2]))
    batchred_mask = ksb.reshape(mask, (shape_tens[0] * shape_tens[1],))
    # Apply boolean mask
    flat_tens = tf.boolean_mask(batchred_tens, batchred_mask)

    # Output
    outpart = kgcnn_ops_change_partition_type(row_lengths, "row_length", partition_type)

    return [flat_tens, outpart]

@tf.function
def kgcnn_ops_cast_tensor_to_value_partition(inputs, partition_type ="row_length"):
    feat = inputs
    sh_feat = ksb.shape(feat)
    sh_feat_int = ksb.int_shape(feat)
    out = ksb.reshape(feat, (sh_feat[0] * sh_feat[1], sh_feat_int[-1]))
    out_len = tf.repeat(sh_feat[1], sh_feat[0])

    # Output
    outpart = kgcnn_ops_change_partition_type(out_len, "row_length", partition_type)

    return [out, outpart]


@tf.function
def kgcnn_ops_cast_value_partition_to_tensor(inputs, partition_type="row_length"):
    out = kgcnn_ops_cast_value_partition_to_ragged(inputs, partition_type=partition_type)
    # out = kgcnn_ops_cast_value_partition_to_tensor(inputs, self.partition_type)
    return out.to_tensor()


@tf.function
def kgcnn_ops_cast_value_partition_to_masked(inputs, partition_type="row_length", ragged_validate=False):
    nod, npartin = inputs

    # Just make ragged tensor.
    if partition_type == "row_length":
        n_len = npartin
        out = tf.RaggedTensor.from_row_lengths(nod, n_len,validate=ragged_validate)
    elif partition_type == "row_splits":
        out = tf.RaggedTensor.from_row_splits(nod, npartin,validate=ragged_validate)
        n_len = out.row_lengths()
    elif partition_type == "value_rowids":
        out = tf.RaggedTensor.from_value_rowids(nod, npartin,validate=ragged_validate)
        n_len = out.row_lengths()
    else:
        raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")

    # Make padded
    out = out.to_tensor()
    # Make mask
    max_len = tf.shape(out)[1]
    n_padd = max_len - n_len
    mask = ks.backend.flatten(tf.concat([tf.expand_dims(tf.ones_like(n_len, dtype=tf.bool), axis=-1),
                                         tf.expand_dims(tf.zeros_like(n_len, dtype=tf.bool), axis=-1)], axis=-1))
    reps = ks.backend.flatten(tf.concat([tf.expand_dims(n_len, axis=-1), tf.expand_dims(n_padd, axis=-1)], axis=-1))
    mask = tf.repeat(mask, reps)
    mask = tf.reshape(mask, tf.shape(out)[:2])
    return [out, mask]





@tf.function
def kgcnn_ops_dyn_cast(inputs, input_tensor_type=None, output_tensor_type=None, partition_type="row_length"):

    tensor_keys = ["Tensor", "tensor"]
    ragged_keys = ["ragged", "RaggedTensor"]
    value_partition_keys = ["disjoint", "values_partition"]

    if input_tensor_type in ragged_keys:
        if output_tensor_type in ragged_keys:
            return inputs
        elif output_tensor_type in value_partition_keys:
            return kgcnn_ops_cast_ragged_to_value_partition(inputs, partition_type=partition_type)
        elif output_tensor_type in tensor_keys:
            return inputs.to_tensor()
        else:
            raise NotImplementedError("Error: Unsupported tensor output type of ", output_tensor_type)

    elif input_tensor_type in value_partition_keys:
        if output_tensor_type in value_partition_keys:
            return inputs
        elif output_tensor_type in ragged_keys:
            return kgcnn_ops_cast_value_partition_to_ragged(inputs, partition_type=partition_type)
        elif output_tensor_type in tensor_keys:
            return kgcnn_ops_cast_value_partition_to_tensor(inputs, partition_type=partition_type)
        else:
            raise NotImplementedError("Error: Unsupported tensor output type of ", output_tensor_type)

    elif input_tensor_type in tensor_keys:
        if output_tensor_type in tensor_keys:
            return inputs
        elif output_tensor_type in value_partition_keys:
            return kgcnn_ops_cast_tensor_to_value_partition(inputs, partition_type=partition_type)
        else:
            return NotImplementedError("Error: Unsupported tensor output type of ", output_tensor_type)

    else:
        raise NotImplementedError("Error: Unsupported tensor input type of ", input_tensor_type)

        #  old casting
        # if self.input_tensor_type == self.output_tensor_type:
        #     return inputs
        #
        # if self.input_tensor_type=="values_partition" and self.output_tensor_type=="ragged":
        #     out = kgcnn_ops_cast_value_partition_to_ragged(inputs, self.partition_type)
        #     return out
        # if self.input_tensor_type=="ragged" and self.output_tensor_type=="values_partition":
        #     out = kgcnn_ops_cast_ragged_to_value_partition(inputs, self.partition_type)
        #     return out
        #
        # if self.input_tensor_type == "masked" and self.output_tensor_type == "ragged":
        #     raise NotImplementedError("Error: Conversion has not been implemented yet.")
        # if self.input_tensor_type == "ragged" and self.output_tensor_type == "masked":
        #     raise NotImplementedError("Error: Conversion has not been implemented yet.")
        #
        # if self.input_tensor_type == "values_partition" and self.output_tensor_type == "masked":
        #     out = kgcnn_ops_cast_value_partition_to_masked(inputs, self.partition_type)
        #     return out
        # if self.input_tensor_type == "masked" and self.output_tensor_type == "values_partition":
        #     out = kgcnn_ops_cast_masked_to_value_partition(inputs, self.partition_type)
        #     return out
        #
        # if self.input_tensor_type == "values_partition" and self.output_tensor_type == "tensor":
        #     out = kgcnn_ops_cast_value_partition_to_ragged(inputs, self.partition_type)
        #     # out = kgcnn_ops_cast_value_partition_to_tensor(inputs, self.partition_type)
        #     return out.to_tensor()
        # if self.input_tensor_type == "tensor" and self.output_tensor_type == "values_partition":
        #     out = kgcnn_ops_cast_tensor_to_value_partition(inputs, self.partition_type)
        #     return out
        #
        # if self.input_tensor_type == "ragged" and self.output_tensor_type == "tensor":
        #     out = inputs.to_tensor()
        #     return out
        # if self.input_tensor_type == "tensor" and self.output_tensor_type == "ragged":
        #     raise NotImplementedError("Error: Conversion has not been implemented yet.")