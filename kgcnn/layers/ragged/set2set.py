import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as ksb


# Order Matters: Sequence to sequence for sets
# by Vinyals et al. 2016
# https://arxiv.org/abs/1511.06391


class Set2Set(ks.layers.Layer):
    """
    Set2Set layer. The Reading to Memory has to be handled seperately.
    
    Uses a keras LSTM layer for the updates.
    The idea is to create a Seq2Seq using a “content” based attention via LSTMs.
    This should basically implement the encoder of chapter 4.2. in the paper.
    Note: Here it is used as a more sensitive averaging of e.g. node features.
    The labeling should match the paper.
    
    Args:
        channels (int): Number of channels for LSTM.
        T (int): iteratrions T=3
        pool_method : tensorflow.keras.backend.mean
        init_qstar: 'mean'
        
        activation : "tanh"
        recurrent_activation : "sigmoid"
        use_bias : True
        kernel_initializer : "glorot_uniform"
        recurrent_initializer : "orthogonal"
        bias_initializer : "zeros"
        unit_forget_bias : True
        kernel_regularizer : None
        recurrent_regularizer : None
        bias_regularizer : None
        activity_regularizer : None
        kernel_constraint : None
        recurrent_constraint : None
        bias_constraint : None
        dropout : 0.0
        recurrent_dropout : 0.0
        implementation : 2
        return_sequences : False
        return_state : False
        go_backwards : False
        stateful : False
        time_major : False
        unroll : False
        **kwargs
    """

    def __init__(self,
                 channels,
                 T=3,
                 pooling_method='mean',
                 init_qstar='mean',
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
        """Initialize layer."""
        super(Set2Set, self).__init__(**kwargs)
        # Number of Channels to use in LSTM
        self.channels = channels
        self.T = T  # Number of Iterations to work on memory
        self.pooling_method = pooling_method
        self.init_qstar = init_qstar

        if self.pooling_method == 'mean':
            self._pool = ksb.mean
        elif self.pooling_method == 'sum':
            self._pool = ksb.sum
        else:
            raise TypeError("Unknown pooling, choose: mean, sum, ...")

        if self.init_qstar == 'mean':
            self.qstar0 = self.init_qstar_mean
        else:
            self.qstar0 = self.init_qstar_0

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

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(Set2Set, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.
        
        Args:
            nodes (tf.ragged): Ragged nodelist of shape (batch,None,F)
        
        Returns:
            features (tf.ragged): Pooled node tensor of shape (batch,1,2*channels)
        """
        x_ragged = inputs
        x = x_ragged.values
        batch_num = x_ragged.row_lengths()
        batch_shape = ksb.shape(batch_num)

        # Reading to memory removed here, is to be done by seperately
        m = x  # (batch,feat)

        # Compute batchindex for averaging
        batch_index = tf.repeat(ksb.arange(0, batch_shape[0], 1), batch_num)  # (batch*num,) ex: [0,0,0,1,2,2,2,...]

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
        """Compute scalar from m and q. Can apply sum or mean etc.
        
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
        config.update({"channels": self.channels})
        config.update({"T": self.T})
        config.update({"pooling_method": self.pooling_method})
        config.update({"init_qstar": self.init_qstar})
        # For LSTM
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
