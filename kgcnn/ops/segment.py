import tensorflow as tf



@tf.function
def segment_softmax(data, segment_ids, normalize=True):
    if normalize:
        data_segment_max = tf.math.segment_max(data,segment_ids)
        data_max = tf.gather(data_segment_max,segment_ids)
        data = data-data_max

    data_exp = tf.math.exp(data)
    data_exp_segment_sum = tf.math.segment_sum(data_exp,segment_ids)
    data_exp_sum = tf.gather(data_exp_segment_sum,segment_ids)

    return data_exp/data_exp_sum


