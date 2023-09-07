import keras_core as ks
from keras_core import ops


class Embedding(ks.layers.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embeddings_initializer = ks.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = ks.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = ks.regularizers.get(activity_regularizer)
        self.embeddings_constraint = ks.constraints.get(embeddings_constraint)

        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(input_dim, output_dim),
            dtype=self.dtype,
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            trainable=True,
        )

    def build(self, input_shape):
        super(Embedding, self).build(input_shape)

    def call(self, inputs):
        return ops.take(self.embeddings, inputs, axis=0)

    def get_config(self):
        return super(Embedding, self).get_config()

