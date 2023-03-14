import tensorflow as tf
import kgcnn.ops.activ
ks = tf.keras
# from keras.engine.base_layer import Layer
# from tensorflow.keras.layers import Layer


class GraphBaseLayer(ks.layers.Layer):
    r"""Base layer for graph layers used in :obj:`kgcnn` .

    To work with ragged tensors, the layer adds the argument :obj:`ragged_validate` .
    Useful utility functions are methods of this class like e.g. :obj:`assert_ragged_input_rank` that
    can be used in graph layers for convenience.

    """

    def __init__(self,
                 ragged_validate: bool = False,
                 **kwargs):
        r"""Initialize layer.

        Args:
            ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        """
        super(GraphBaseLayer, self).__init__(**kwargs)
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._add_layer_config_to_self = {}

    def get_config(self):
        """Update layer config."""
        config = super(GraphBaseLayer, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        # Also add the config of a sub-layer to self.
        # Should only be done if sub-layer does not change config on built.
        for key, value in self._add_layer_config_to_self.items():
            if hasattr(self, key):
                if getattr(self, key) is not None:
                    layer_conf = getattr(self, key).get_config()
                    for x in value:
                        if x in layer_conf:
                            config.update({x: layer_conf[x]})
        return config

    def build(self, input_shape):
        """Build base layer."""
        super(GraphBaseLayer, self).build(input_shape)

    def assert_ragged_input_rank(self, inputs, mask=None, ragged_rank: int = 1):
        r"""Assert input to be ragged with a given ragged_rank. This function can be used to assert that the input
        is in ragged tensor form, if for example you want to access values and splits or ragged methods in a layer.
        If inputs are not ragged, then they are cast into ragged tensors if possible. Inputs can be a single tensor
        or a list of tensors.

        Args:
            inputs: Tensor or list of tensors to assert to be ragged and have given ragged rank.
            mask: Boolean mask for inputs. Default is None.
            ragged_rank (int): Assert ragged tensor to have ragged_rank. Default is 1.

        Returns:
            inputs: Tensor or list of tensors as ragged.
        """
        if mask is not None:
            raise ValueError("Using `mask` argument in `assert_ragged_input_rank` is not yet supported.")

        def validate_or_cast(x):
            if isinstance(x, tf.RaggedTensor):
                if ragged_rank is not None:
                    assert x.ragged_rank == ragged_rank, "'%s' must have input with ragged_rank=%s." % (
                        self.name, ragged_rank)
                return x
            elif isinstance(x, tf.Tensor) and mask is None:
                if ragged_rank is None:
                    raise ValueError("Casting to ragged without `ragged_rank` information is not supported.")
                if ragged_rank != 1:
                    raise ValueError("Casting to ragged is only supported for ragged_rank=1 at the moment.")
                if x.shape.rank <= ragged_rank:
                    raise ValueError(
                        "Rank of inputs must be > ragged_rank but found '%s <= %s' " % (x.shape.rank, ragged_rank))
                return tf.RaggedTensor.from_row_lengths(
                    tf.reshape(x, shape=tf.concat([tf.shape(x)[:1]*tf.shape(x)[1:2], tf.shape(x)[2:]], axis=0)),
                    tf.repeat(tf.shape(x)[1], tf.shape(x)[0]))
            else:
                raise ValueError("Unsupported tensor type '%s' in '%s'." % (type(x), self.name))

        if isinstance(inputs, (list, tuple)):
            return [validate_or_cast(input_item) for input_item in inputs]
        else:
            return validate_or_cast(inputs)

    def map_values(self, fun, inputs, **kwargs):
        r"""This is a helper function that attempts to call :obj:`fun` on the value tensor(s) of :obj:`inputs`.
        For ragged rank of one, the values is a :obj:`tf.Tensor` itself.
        The argument :obj:`inputs` must be a ragged tensors with ragged rank of one, or a list of ragged tensors.
        The fallback is to call :obj:`fun` directly on inputs.
        For list input, this corresponds to a "lazy operation" which requires :obj:`ragged_validate` is set to
        :obj:`False`, which means it is not checked if splits are equal.
        For the output of :obj:`fun` it is always assumed that the ragged partition does not change in :obj:`fun`.
        If `axis` is found in `kwargs`, the axis argument is adapted for the :obj:`values` tensor if possible, otherwise
        the tensor is passed as fallback to :obj:`fun` directly.

        Args:
            fun (callable): Callable function that accepts inputs and kwargs.
            inputs (tf.RaggedTensor, list): Tensor input or list of tensors.
            kwargs: Additional kwargs for fun.

        Returns:
            tf.RaggedTensor: Output of fun only on the :obj:`values` tensor of the ragged input.
        """
        if "axis" in kwargs:
            axis = kwargs["axis"]
            axis_values = None
            kwargs_values = None
            if isinstance(axis, int):
                if axis > 1:
                    axis_values = axis - 1
            elif isinstance(axis, (list, tuple)):
                if all([x > 1 for x in axis]):
                    axis_values = [x - 1 for x in axis]
            if axis_values is not None:
                kwargs_values = {key: value for key, value in kwargs.items()}
                kwargs_values.pop("axis")
                kwargs_values.update({"axis": axis_values})
        else:
            kwargs_values = {key: value for key, value in kwargs.items()}

        if isinstance(inputs, list) and kwargs_values is not None:
            if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
                if all([x.ragged_rank == 1 for x in inputs]) and not self.ragged_validate:
                    out = fun([x.values for x in inputs], **kwargs_values)
                    if isinstance(out, list):
                        return [tf.RaggedTensor.from_row_splits(x, inputs[i].row_splits, validate=self.ragged_validate)
                                for i, x in enumerate(out)]
                    else:
                        return tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
        elif isinstance(inputs, tf.RaggedTensor) and kwargs_values is not None:
            if inputs.ragged_rank == 1:
                out = fun(inputs.values, **kwargs_values)
                if isinstance(out, list):
                    return [tf.RaggedTensor.from_row_splits(x, inputs.row_splits, validate=self.ragged_validate) for x
                            in out]
                else:
                    return tf.RaggedTensor.from_row_splits(out, inputs.row_splits, validate=self.ragged_validate)

        if isinstance(inputs, tf.RaggedTensor):
            print("WARNING: Layer %s fail call on value Tensor of ragged Tensor." % self.name)
        if isinstance(inputs, list):
            if any([isinstance(x, tf.RaggedTensor) for x in inputs]):
                print("WARNING: Layer %s fail call on value Tensor for ragged Tensor in list." % self.name)
        return fun(inputs, **kwargs)

    # Deprecated Synonym for map values.
    call_on_values_tensor_of_ragged = map_values
