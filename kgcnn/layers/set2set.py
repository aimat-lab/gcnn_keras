import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as ksb

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_partition_type
from kgcnn.ops.types import kgcnn_ops_check_tensor_type, kgcnn_ops_static_test_tensor_input_type


# Order Matters: Sequence to sequence for sets
# by Vinyals et al. 2016
# https://arxiv.org/abs/1511.06391


class Set2Set(ks.layers.Layer):
    """
    Set2Set layer. The Reading to Memory has to be handled seperately.
    Uses a keras LSTM layer for the updates.
    
    Args:
        channels (int): Number of channels for the LSTM update.
        T (int): Numer of iterations. Default is T=3.
        pooling_method : Pooling method for Set2Set. Default is 'mean'.
        init_qstar: How to generate the first q_star vector. Default is 'mean'.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
            is applied (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean (default `True`), whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix, used for
            the linear transformation of the inputs. Default: `glorot_uniform`.
            recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
            Default: `orthogonal`.
        bias_initializer: Initializer for the bias vector. Default: `zeros`.
            unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
            the forget gate at initialization. Setting it to true will also force
            `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
            bias_regularizer: Regularizer function applied to the bias vector. Default:
            `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel`
            weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector. Default:
            `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear
            transformation of the inputs. Default: 0.
            recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
            the linear transformation of the recurrent state. Default: 0.
        return_sequences: Boolean. Whether to return the last output. in the output
            sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition to the
            output. Default: `False`.
        go_backwards: Boolean (default `False`). If True, process the input sequence
            backwards and return the reversed sequence.
        stateful: Boolean (default `False`). If True, the last state for each sample
            at index i in a batch will be used as initial state for the sample of
            index i in the following batch.
        time_major: The shape format of the `inputs` and `outputs` tensors.
            If True, the inputs and outputs will be in shape
            `[timesteps, batch, feature]`, whereas in the False case, it will be
            `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
            efficient because it avoids transposes at the beginning and end of the
            RNN calculation. However, most TensorFlow data is batch-major, so by
            default this function accepts input and emits output in batch-major
            form.
        unroll: Boolean (default `False`). If True, the network will be unrolled,
            else a symbolic loop will be used. Unrolling can speed-up a RNN, although
            it tends to be more memory-intensive. Unrolling is only suitable for short
            sequences.
    """

    def __init__(self,
                 # Arg
                 channels,
                 T=3,
                 pooling_method='mean',
                 init_qstar='mean',
                 node_indexing="sample",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 # Args for LSTM
                 activation="tanh",
                 recurrent_activation="sigmoid",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 recurrent_initializer="orthogonal",
                 bias_initializer="zeros",
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 implementation=2,
                 return_sequences=False,  # Should not be changed here
                 return_state=False,  # Should not be changed here
                 go_backwards=False,  # Should not be changed here
                 stateful=False,
                 time_major=False,
                 unroll=False,
                 **kwargs):
        """Init layer."""
        super(Set2Set, self).__init__(**kwargs)
        # Number of Channels to use in LSTM
        self.channels = channels
        self.T = T  # Number of Iterations to work on memory
        self.pooling_method = pooling_method
        self.init_qstar = init_qstar
        self.partition_type = partition_type
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

        if self.pooling_method == 'mean':
            self._pool = ksb.mean
        elif self.pooling_method == 'sum':
            self._pool = ksb.sum
        else:
            raise TypeError("Unknown pooling, choose: 'mean', 'sum', ...")

        if self.init_qstar == 'mean':
            self.qstar0 = self.init_qstar_mean
        else:
            self.qstar0 = self.init_qstar_0
        # ...

        # LSTM Layer to run on m
        self.lay_lstm = ks.layers.LSTM(channels,
                                       activation=activation,
                                       recurrent_activation=recurrent_activation,
                                       use_bias=use_bias,
                                       kernel_initializer=kernel_initializer,
                                       recurrent_initializer=recurrent_initializer,
                                       bias_initializer=bias_initializer,
                                       unit_forget_bias=unit_forget_bias,
                                       kernel_regularizer=kernel_regularizer,
                                       recurrent_regularizer=recurrent_regularizer,
                                       bias_regularizer=bias_regularizer,
                                       activity_regularizer=activity_regularizer,
                                       kernel_constraint=kernel_constraint,
                                       recurrent_constraint=recurrent_constraint,
                                       bias_constraint=bias_constraint,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       implementation=implementation,
                                       return_sequences=return_sequences,
                                       return_state=return_state,
                                       go_backwards=go_backwards,
                                       stateful=stateful,
                                       time_major=time_major,
                                       unroll=unroll
                                       )

    def build(self, input_shape):
        """Build layer."""
        super(Set2Set, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
        and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
        disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape
        (batch, N, F) and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
        or a singe tensor for equally sized graphs (batch,N,F).

        Args:
            inputs: Embeddings to be encoded of shape (batch, [N], F)
        
        Returns:
            q_star (tf.tensor): Pooled tensor of shape (batch,1,2*channels)
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs, input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        x, batch_part = kgcnn_ops_dyn_cast(inputs, input_tensor_type=found_node_type,
                                           output_tensor_type="values_partition",
                                           partition_type=self.partition_type)
        batch_num = kgcnn_ops_change_partition_type(batch_part, self.partition_type, "row_length")
        batch_index = kgcnn_ops_change_partition_type(batch_part, self.partition_type, "value_rowids")

        # Reading to memory removed here, is to be done by seperately
        m = x  # (batch*None,feat)

        # Initialize q0 and r0
        qstar = self.qstar0(m, batch_index, batch_num)

        # start loop
        for i in range(0, self.T):
            q = self.lay_lstm(qstar)  # (batch,feat)
            qt = tf.repeat(q, batch_num, axis=0)  # (batch*num,feat)
            et = self.f_et(m, qt)  # (batch*num,)
            # get at = exp(et)/sum(et) with sum(et)
            at = ksb.exp(et - self.get_scale_per_sample(et, batch_index, batch_num))  # (batch*num,)
            norm = self.get_norm(at, batch_index, batch_num)  # (batch*num,)
            at = norm * at  # (batch*num,) x (batch*num,)
            # calculate rt
            at = ksb.expand_dims(at, axis=1)
            rt = m * at  # (batch*num,feat) x (batch*num,1)
            rt = tf.math.segment_sum(rt, batch_index)  # (batch,feat)
            # qstar = [q,r]
            qstar = ksb.concatenate([q, rt], axis=1)  # (batch,2*feat)
            qstar = ksb.expand_dims(qstar, axis=1)  # (batch,1,2*feat)

        return qstar

    def f_et(self, fm, fq):
        """
        Function to compute scalar from m and q. Can apply sum or mean etc.
        
        Args:
             fm (tf.tensor): of shape (batch*num,feat)
             fq (tf.tensor): of shape (batch*num,feat)
            
        Returns:
            et (tf.tensor): of shape (batch*num,)  
        """
        fet = self._pool(fm * fq, axis=1)  # (batch*num,1)
        return fet

    @staticmethod
    def get_scale_per_batch(x):
        """Get rescaleing for the batch."""
        return tf.keras.backend.max(x, axis=0, keepdims=True)

    @staticmethod
    def get_scale_per_sample(x, ind, rep):
        """Get rescaleing for the sample."""
        out = tf.math.segment_max(x, ind)  # (batch,)
        out = tf.repeat(out, rep)  # (batch*num,)
        return out

    @staticmethod
    def get_norm(x, ind, rep):
        """Compute Norm."""
        norm = tf.math.segment_sum(x, ind)  # (batch,)
        norm = tf.math.reciprocal_no_nan(norm)  # (batch,)
        norm = tf.repeat(norm, rep, axis=0)  # (batch*num,)
        return norm

    def init_qstar_0(self, m, batch_index, batch_num):
        """Initialize the q0 with zeros."""
        batch_shape = ksb.shape(batch_num)
        return tf.zeros((batch_shape[0], 1, 2 * self.channels))

    def init_qstar_mean(self, m, batch_index, batch_num):
        """Initialize the q0 with mean."""
        # use q0=avg(m) (or q0=0)
        # batch_shape = ksb.shape(batch_num)
        q = tf.math.segment_mean(m, batch_index)  # (batch,feat)
        # r0
        qt = tf.repeat(q, batch_num, axis=0)  # (batch*num,feat)
        et = self.f_et(m, qt)  # (batch*num,)
        # get at = exp(et)/sum(et) with sum(et)=norm
        at = ksb.exp(et - self.get_scale_per_sample(et, batch_index, batch_num))  # (batch*num,)
        norm = self.get_norm(at, batch_index, batch_num)  # (batch*num,)
        at = norm * at  # (batch*num,) x (batch*num,)
        # calculate rt
        at = ksb.expand_dims(at, axis=1)  # (batch*num,1)
        rt = m * at  # (batch*num,feat) x (batch*num,1)
        rt = tf.math.segment_sum(rt, batch_index)  # (batch,feat)
        # [q0,r0]
        qstar = ksb.concatenate([q, rt], axis=1)  # (batch,2*feat)
        qstar = ksb.expand_dims(qstar, axis=1)  # (batch,1,2*feat)
        return qstar

    def get_config(self):
        """Make config for layer."""
        config = super(Set2Set, self).get_config()
        config.update({"channels": self.channels,
                       "T": self.T,
                       "pooling_method": self.pooling_method,
                       "init_qstar": self.init_qstar,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        lstm_conf = self.lay_lstm.get_config()
        lstm_param = ["activation",
                      "recurrent_activation",
                      "use_bias",
                      "kernel_initializer",
                      "recurrent_initializer",
                      "bias_initializer",
                      "unit_forget_bias",
                      "kernel_regularizer",
                      "recurrent_regularizer",
                      "bias_regularizer",
                      "activity_regularizer",
                      "kernel_constraint",
                      "recurrent_constraint",
                      "bias_constraint",
                      "dropout",
                      "recurrent_dropout",
                      "implementation",
                      "return_sequences",  # Should not be changed here
                      "return_state",  # Should not be changed here
                      "go_backwards",  # Should not be changed here
                      "stateful",
                      "time_major",
                      "unroll"]
        for x in lstm_param:
            config.update({x: lstm_conf[x]})
        return config
