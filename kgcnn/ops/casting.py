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
    infeat, inpartition = inputs
    outsh = ksb.int_shape(infeat)

    if partition_type == "row_length":
        ref = inpartition
        insh = ksb.shape(ref)
        out = ksb.reshape(infeat, (insh[0], -1, outsh[-1]))
    elif partition_type == "row_splits":
        ref = inpartition[:-1]
        insh = ksb.shape(ref)
        out = ksb.reshape(infeat, (insh[0], -1, outsh[-1]))
    elif partition_type == "value_rowids":
        ref = tf.math.segment_sum(tf.ones_like(inpartition), inpartition)
        insh = ksb.shape(ref)
        out = ksb.reshape(infeat, (insh[0], -1, outsh[-1]))
    else:
        raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")

    return out

@tf.function
def kgcnn_ops_cast_value_partition_to_masked(inputs, partition_type="row_length"):
    nod, npartin = inputs

    # Just make ragged tensor.
    if partition_type == "row_length":
        n_len = npartin
        out = tf.RaggedTensor.from_row_lengths(nod, n_len)
    elif partition_type == "row_splits":
        out = tf.RaggedTensor.from_row_splits(nod, npartin)
        n_len = out.row_lengths()
    elif partition_type == "value_rowids":
        out = tf.RaggedTensor.from_value_rowids(nod, npartin)
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
def kgcnn_ops_cast_value_partition_to_ragged(inputs, partition_type="row_length"):
    nod, n_part = inputs

    if partition_type == "row_length":
        out = tf.RaggedTensor.from_row_lengths(nod, n_part)
    elif partition_type == "row_splits":
        out = tf.RaggedTensor.from_row_splits(nod, n_part)
    elif partition_type == "value_rowids":
        out = tf.RaggedTensor.from_value_rowids(nod, n_part)
    else:
        raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")

    return out
