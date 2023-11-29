import keras as ks
from keras import ops
from kgcnn.ops.scatter import scatter_reduce_sum, scatter_reduce_max
from kgcnn.layers.aggr import Aggregate

# Order Matters: Sequence to sequence for sets
# by Vinyals et al. 2016
# https://arxiv.org/abs/1511.06391


class PoolingSet2SetEncoder(ks.layers.Layer):
    r"""Pooling Node or edge embeddings by the Set2Set encoder part from layer.
    This was first proposed by `NMPNN <http://arxiv.org/abs/1704.01212>`__ .
    The Reading to Memory has to be handled separately.
    Uses a keras LSTM layer for the updates.
    """

    def __init__(self,
                 # Args
                 channels,
                 T=3,  # noqa
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
                 # time_major=False,
                 unroll=False,
                 **kwargs):
        """Initialize layer.

        Args:
            channels (int): Number of channels for the LSTM update.
            T (int): Numer of iterations. Default is T=3.
            pooling_method : Pooling method for PoolingSet2SetEncoder. Default is 'mean'.
            init_qstar: How to generate the first q_star vector. Default is 'mean'.
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
            unroll: Boolean (default `False`). If True, the network will be unrolled,
                else a symbolic loop will be used. Unrolling can speed-up a RNN, although
                it tends to be more memory-intensive. Unrolling is only suitable for short
                sequences.
        """
        super(PoolingSet2SetEncoder, self).__init__(**kwargs)
        # Number of Channels to use in LSTM
        self.channels = channels
        self.T = T  # Number of Iterations to work on memory
        self.pooling_method = pooling_method
        self.init_qstar = init_qstar

        # Reduction of messages for f_et
        self._reduce_keys = {
            "sum": ops.sum,
            "mean": ops.mean,
            "max": ops.max,
            "min": ops.min,
            "var": ops.var,
        }
        if self.pooling_method not in self._reduce_keys:
            raise ValueError("ERROR:kgcnn: Unknown reduction '%s', choose one of '%s'." % (
                self.pooling_method, self._reduce_keys.keys()))
        self._reduce = self._reduce_keys[self.pooling_method]

        self._pool_init = None
        if self.init_qstar in ["0", "zeros", "zero"]:
            self.qstar0 = self.init_qstar_0
        elif self.init_qstar in ["ref", "reference", "input"]:
            self.qstar0 = self.init_qstar_ref
        else:
            self._pool_init = Aggregate(pooling_method=self.init_qstar)
            self.qstar0 = self.init_qstar_pool

        # LSTM Layer to run on m
        self.lay_lstm = ks.layers.LSTM(
            channels,
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
            # time_major=time_major,
            unroll=unroll
        )

    def build(self, input_shape):
        """Build layer."""
        assert len(input_shape) == 3
        ref_shape, attr_shape, index_shape = [list(x) for x in input_shape]
        if self._pool_init is not None:
            self._pool_init.build([attr_shape, index_shape, ref_shape])
        self.lay_lstm.build(tuple(ref_shape[:1] + [1] + [2*self.channels]))
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        ref_shape, attr_shape, index_shape = [list(x) for x in input_shape]
        return tuple(ref_shape[:1] + [1] + [2*self.channels])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [reference, nodes, batch_index]

                - reference (Tensor): Reference for aggregation of shape `(batch, ...)` .
                - nodes (Tensor): Node embeddings of shape `([N], F)` .
                - batch_index (Tensor): Batch assignment of shape `([N], )` .

        Returns:
            Tensor: Pooled tensor q_star of shape `(batch, 1, 2*channels)`
        """
        ref, x, batch_index = inputs

        # Reading to memory removed here, is to be done by separately
        m = x  # ([N], feat)

        # Initialize q0 and r0
        q_star = self.qstar0(m, batch_index, ref)

        # start loop
        for i in range(0, self.T):
            q = self.lay_lstm(q_star)  # (batch, feat)
            q_star = self.update_q(q, m, batch_index, ref)

        return q_star

    def update_q(self, q, m, batch_index, ref):
        qt = ops.take(q, batch_index, axis=0)
        et = self.f_et(m, qt)  # (batch*num,)
        # get at = exp(et)/sum(et) with sum(et)
        at = ops.exp(et - self._get_scale_per_sample(et, batch_index, ref))  # (batch*num,)
        norm = self._get_norm(at, batch_index, ref)  # (batch*num,)
        at = norm * at  # (batch*num,) x (batch*num,)
        # calculate rt
        # at = ops.expand_dims(at, axis=1)
        rt = m * at  # (batch*num,feat) x (batch*num,1)
        rt = self._pool_sum([rt, batch_index, ref])  # (batch,feat)
        # qstar = [q,r]
        q_star = ops.concatenate([q, rt], axis=1)  # (batch,2*feat)
        q_star = ops.expand_dims(q_star, axis=1)  # (batch,1,2*feat)
        return q_star

    def f_et(self, fm, fq):
        r"""Function to compute scalar from :math:`m` and :math:`q` .
        Uses :obj:`pooling_method` argument of the layer.

        Args:
             fm (Tensor): of shape `([N], F)` .
             fq (Tensor): of shape `([N], F)` .

        Returns:
            Tensor: et of shape `([N], )` .
        """
        return self._reduce(fm * fq, axis=1, keepdims=True)  # ([N], )

    @staticmethod
    def _get_scale_per_batch(x):
        """Get re-scaling for the batch."""
        return ops.max(x, axis=0, keepdims=True)

    def _get_scale_per_sample(self, x, ind, ref):
        """Get re-scaling for the sample."""
        out = self._pool_max([x, ind, ref])  # (batch,)
        out = ops.take(out, ind, axis=0)  # (batch*num,)
        return out

    def _get_norm(self, x, ind, ref):
        """Compute Norm."""
        norm = self._pool_sum([x, ind, ref])  # (batch,)
        norm = ops.divide(ops.convert_to_tensor(1.0, dtype=norm.dtype), norm)  # (batch,)
        norm = ops.where(ops.logical_or(ops.isnan(norm), ops.isinf(norm)), 0., norm)
        norm = ops.take(norm, ind, axis=0)  # (batch*num,)
        return norm

    def init_qstar_ref(self, m, batch_index, reference):
        return reference

    def init_qstar_0(self, m, batch_index, reference):
        """Initialize the q0 with zeros."""
        batch_shape = ops.shape(reference)[0]
        return ops.zeros((batch_shape, 1, 2 * self.channels), dtype=m.dtype)

    def init_qstar_pool(self, m, batch_index, reference):
        """Initialize the q0 with mean."""
        # batch_shape = ksb.shape(batch_num)
        q = self._pool_init([m, batch_index, reference])  # (batch,feat)
        qstar= self.update_q(q, m, batch_index, reference)
        return qstar

    def get_config(self):
        """Make config for layer."""
        config = super(PoolingSet2SetEncoder, self).get_config()
        config.update({"channels": self.channels, "T": self.T, "pooling_method": self.pooling_method,
                       "init_qstar": self.init_qstar})
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
                      # "time_major",
                      "unroll"]
        for x in lstm_param:
            if x in lstm_conf.keys():
                config.update({x: lstm_conf[x]})
        return config

    def _pool_sum(self, inputs):
        values, indices, ref = inputs
        shape_ = ops.shape(ref)[:1] + ops.shape(values)[1:]
        return scatter_reduce_sum(indices, values, shape=shape_)

    def _pool_max(self, inputs):
        values, indices, ref = inputs
        shape_ = ops.shape(ref)[:1] + ops.shape(values)[1:]
        return scatter_reduce_max(indices, values, shape=shape_)