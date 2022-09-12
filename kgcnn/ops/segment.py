import tensorflow as tf


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
def segment_ops_by_name(segment_name: str, data, segment_ids):
    """Segment operation chosen by string identifier.

    Args:
        segment_name (str): Name of the segment operation.
        data (tf.Tensor): Data tensor that has sorted segments.
        segment_ids (tf.Tensor): IDs of the segments.

    Returns:
        tf.Tensor: reduced segment data with method by segment_name.
    """
    if segment_name in ["segment_mean", "mean", "reduce_mean"]:
        pool = tf.math.segment_mean(data, segment_ids)
    elif segment_name in ["segment_sum", "sum", "reduce_sum"]:
        pool = tf.math.segment_sum(data, segment_ids)
    elif segment_name in ["segment_max", "max", "reduce_max"]:
        pool = tf.math.segment_max(data, segment_ids)
    elif segment_name in ["segment_min", "min", "reduce_min"]:
        pool = tf.math.segment_min(data, segment_ids)
    # softmax does not reduce tensor.
    # elif segment_name in ["segment_softmax", "segment_soft_max", "softmax", "soft_max", "reduce_softmax"]:
    #     pool = segment_softmax(data, segment_ids)
    else:
        raise TypeError("Unknown segment operation, choose: 'segment_mean', 'segment_sum', ...")
    return pool
