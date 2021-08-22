import numpy as np
import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='EmbeddingDimeBlock')
class EmbeddingDimeBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,  # Vocabulary
                 output_dim,  # Embedding size
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        super(EmbeddingDimeBlock, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

        # self.embeddings_initializer = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))
        # embeddings_initializer = {'class_name': 'RandomUniform',
        #   'config': {'minval': -1.7320508075688772,
        #    'maxval': 1.7320508075688772,
        #    'seed': None}}
        self.embeddings = self.add_weight(name="embeddings", shape=(self.input_dim + 1, self.output_dim),
                                          dtype=self.dtype, initializer=self.embeddings_initializer,
                                          regularizer=self.embeddings_regularizer,
                                          constraint=self.embeddings_constraint,
                                          trainable=True)

    def call(self, inputs, **kwargs):
        """Embedding of inputs. Forward pass."""
        out = tf.gather(self.embeddings, tf.cast(inputs, dtype=tf.int32))
        return out

    def get_config(self):
        config = super(EmbeddingDimeBlock, self).get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim,
                       "embeddings_initializer": tf.keras.initializers.serialize(self.embeddings_initializer),
                       "embeddings_regularizer": tf.keras.regularizers.serialize(self.embeddings_regularizer),
                       "embeddings_constraint": tf.keras.constraints.serialize(self.embeddings_constraint)
                       })
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='SplitEmbedding')
class SplitEmbedding(GraphBaseLayer):
    def __init__(self,
                 num_or_size_splits,
                 axis=-1,
                 num=None,
                 **kwargs):
        super(SplitEmbedding, self).__init__(**kwargs)
        # self._supports_ragged_inputs = True
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.out_num = num

    def call(self, inputs, **kwargs):
        """Split embeddings across feature dimension e.g. `axis=-1`."""
        if isinstance(inputs, tf.RaggedTensor):
            if self.axis == -1 and inputs.shape[-1] is not None and inputs.ragged_rank == 1:
                value_tensor = inputs.values  # will be Tensor
                out_tensor = tf.split(value_tensor, self.num_or_size_splits, axis=self.axis, num=self.out_num)
                return [tf.RaggedTensor.from_row_splits(x, inputs.row_splits, validate=self.ragged_validate) for x in
                        out_tensor]
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting tf.split... ")

        out = tf.split(inputs, self.num_or_size_splits, axis=self.axis, num=self.out_num)
        return out

    def get_config(self):
        config = super(SplitEmbedding, self).get_config()
        config.update({"num_or_size_splits": self.num_or_size_splits, "axis": self.axis, "num": self.out_num
                       })
        return config
