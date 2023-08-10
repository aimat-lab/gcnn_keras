import tensorflow as tf


def pad_segments(segment, max_id):
    """

    Args:
        segment:
        max_id:

    Returns:
        tf.Tensor: Padded Tensor.
    """
    missing_num = tf.expand_dims(max_id, axis=0) - tf.shape(segment)[:1]
    # out = tf.pad(out,
    # tf.concat([tf.constant([0], dtype=missing_num), missing_num], axis=0)),
    # )
    out = tf.concat([
        segment,
        tf.zeros(tf.concat([missing_num, tf.shape(segment)[1:]], axis=0), dtype=segment.dtype)
    ], axis=0)
    # out = tf.scatter_nd(ks.backend.expand_dims(tf.range(tf.shape(pool)[0]), axis=-1), pool,
    #                     tf.concat([tf.expand_dims(max_id, axis=0), tf.shape(pool)[1:]], axis=0))
    return out


@tf.function
def segment_softmax(data, segment_ids, normalize: bool = True):
    """Segment softmax similar to segment_max but with a softmax function.

    Args:
        data (tf.Tensor): Data tensor that has sorted segments.
        segment_ids (tf.Tensor): IDs of the segments.
        normalize (bool): Normalize data for softmax. Default is True.

    Returns:
        tf.Tensor: reduced segment data with a softmax function.
    """
    if normalize:
        data_segment_max = tf.math.segment_max(data, segment_ids)
        data_max = tf.gather(data_segment_max, segment_ids)
        data = data - data_max

    data_exp = tf.math.exp(data)
    data_exp_segment_sum = tf.math.segment_sum(data_exp, segment_ids)
    data_exp_sum = tf.gather(data_exp_segment_sum, segment_ids)
    return data_exp / data_exp_sum


@tf.function
def segment_ops_by_name(segment_name: str, data, segment_ids, max_id=None):
    """Segment operation chosen by string identifier.

    Args:
        segment_name (str): Name of the segment operation.
        data (tf.Tensor): Data tensor that has sorted segments.
        segment_ids (tf.Tensor): IDs of the segments.
        max_id (tf.Tensor): Max ID if the maximum ID is not in the segment_ids. Default is None.

    Returns:
        tf.Tensor: reduced segment data with method by segment_name.
    """
    if segment_name in ["segment_mean", "mean"]:
        pool = tf.math.segment_mean(data, segment_ids)
    elif segment_name in ["segment_sum", "sum", "add", "segment_add"]:
        pool = tf.math.segment_sum(data, segment_ids)
    elif segment_name in ["segment_max", "max"]:
        pool = tf.math.segment_max(data, segment_ids)
    elif segment_name in ["segment_min", "min"]:
        pool = tf.math.segment_min(data, segment_ids)
    # softmax does not really reduce tensor.
    # which is why it is not added to the list of segment operations for normal pooling.
    else:
        raise TypeError("Unknown segment operation, choose: 'segment_mean', 'segment_sum', ...")

    if max_id is not None:
        pool = pad_segments(pool, max_id=max_id)

    return pool
