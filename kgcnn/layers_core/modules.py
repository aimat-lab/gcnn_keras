import keras_core as ks


class Embedding(ks.layers.Layer):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, **kwargs):
        super().__init__(**kwargs)
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        emb_init = ops.initializers.RandomUniform(
            minval=-np.sqrt(3), maxval=np.sqrt(3)
        )
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(93, self.emb_size),
            dtype=self.dtype,
            initializer=emb_init,
            trainable=True,
        )

    def call(self, inputs):
        Z = inputs
        h = tf.gather(self.embeddings, Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class OptionalInputEmbedding(ks.layers.Layer):
    r"""Layer that optionally applies an :obj:`Embedding` to input tensor.
    This layer can only be used on positive integer inputs of a fixed range.
    The layer parameter :obj:`use_embedding` decides whether to actually use the embedding layer.
    """

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
        """Initialize layer.

        Args:
            input_dim: Integer. Size of the vocabulary,
                i.e. maximum integer index + 1.
            output_dim: Integer. Dimension of the dense embedding.
            use_embedding (bool): Whether to actually apply the embedding.
            embeddings_initializer: Initializer for the `embeddings`
                matrix (see `keras.initializers`).
            embeddings_regularizer: Regularizer function applied to
                the `embeddings` matrix (see `keras.regularizers`).
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
            embeddings_constraint: Constraint function applied to
                the `embeddings` matrix (see `keras.constraints`).
            mask_zero: Boolean, whether or not the input value 0 is a special "padding"
                value that should be masked out.
                This is useful when using recurrent layers
                which may take variable length input.
                If this is `True`, then all subsequent layers
                in the model need to support masking or an exception will be raised.
                If mask_zero is set to True, as a consequence, index 0 cannot be
                used in the vocabulary (input_dim should equal size of
                vocabulary + 1).
            input_length: Length of input sequences, when it is constant.
                This argument is required if you are going to connect
                `Flatten` then `Dense` layers upstream
                (without it, the shape of the dense outputs cannot be computed).
        """
        super(OptionalInputEmbedding, self).__init__(**kwargs)
        self.use_embedding = use_embedding

        if use_embedding:
            self._layer_embed = ks.layers.Embedding(input_dim=input_dim, output_dim=output_dim,
                                                    embeddings_initializer=embeddings_initializer,
                                                    embeddings_regularizer=embeddings_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    embeddings_constraint=embeddings_constraint,
                                                    mask_zero=mask_zero, input_length=input_length)
            self._add_layer_config_to_self = {"_layer_embed": ["input_dim", "output_dim", "embeddings_initializer",
                                                               "embeddings_regularizer", "activity_regularizer",
                                                               "embeddings_constraint", "mask_zero", "input_length"]}

    def build(self, input_shape):
        """Build layer."""
        super(OptionalInputEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass of embedding layer if `use_embedding=True`.

        Args:
            inputs (Tensor): Tensor of indices of shape `(batch, [N])`.

        Returns:
            Tensor: Embeddings of shape `(batch, [N], F)`.
        """
        if self.use_embedding:
            return self._layer_embed(inputs)
        return inputs

    def get_config(self):
        """Get config of layer."""
        config = super(OptionalInputEmbedding, self).get_config()
        config.update({"use_embedding": self.use_embedding})
        return config
