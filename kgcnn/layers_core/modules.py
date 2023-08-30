import keras_core as ks
from keras_core import ops


class Embedding(ks.layers.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 use_embedding=False,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(output_dim, input_dim),
            dtype=self.dtype,
            initializer=embeddings_initializer,
            trainable=True,
        )

    def call(self, inputs):
        out = ops.take(self.embeddings, inputs)
        return out

    def get_config(self):
        return super(Embedding, self).get_config()