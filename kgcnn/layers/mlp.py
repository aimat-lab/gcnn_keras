import keras as ks
from keras.layers import Dense, Layer, Activation, Dropout
from keras.layers import LayerNormalization, GroupNormalization, BatchNormalization, UnitNormalization
from kgcnn.layers.norm import (GraphNormalization, GraphInstanceNormalization,
                               GraphBatchNormalization, GraphLayerNormalization)
from kgcnn.layers.norm import global_normalization_args as global_normalization_args_graph
from kgcnn.layers.relational import RelationalDense


global_normalization_args = {
    "UnitNormalization": (
        "axis"
    ),
    "BatchNormalization": (
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint", "momentum", "moving_mean_initializer",
        "moving_variance_initializer"
    ),
    "GroupNormalization": (
        "groups", "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint"
    ),
    "LayerNormalization": (
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint"
    )
}
global_normalization_args.update(global_normalization_args_graph)


class _MLPBase(Layer):  # noqa

    def __init__(self,
                 units,
                 use_bias=True,
                 activation=None,
                 activity_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_constraint=None,
                 bias_constraint=None,
                 # Normalization
                 use_normalization=False,
                 normalization_technique="BatchNormalization",
                 axis=-1,
                 momentum=0.99,
                 epsilon=0.001,
                 mean_shift=True,
                 center=True,
                 scale=True,
                 alpha_initializer="ones",
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 alpha_regularizer=None,
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 alpha_constraint=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 # Dropout
                 use_dropout=False,
                 rate=None,
                 noise_shape=None,
                 seed=None,
                 # Graph
                 padded_disjoint: bool = False,
                 **kwargs):
        r"""
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            use_normalization: Whether to use a normalization layer in between.
            normalization_technique: Which keras normalization technique to apply.
                This can be either 'batch', 'layer', 'group' etc.
            axis: Integer, the axis that should be normalized (typically the features
                axis). For instance, after a `Conv2D` layer with
                `data_format="channels_first"`, set `axis=1` in `GraphBatchNormalization`.
            momentum: Momentum for the moving average.
            epsilon: Small float added to variance to avoid dividing by zero.
            mean_shift: Whether to apply alpha.
            center: If True, add offset of `beta` to normalized tensor. If False, `beta`
                is ignored.
            scale: If True, multiply by `gamma`. If False, `gamma` is not used. When the
                next layer is linear (also e.g. `nn.relu`), this can be disabled since the
                scaling will be done by the next layer.
            alpha_initializer: Initializer for the alpha weight. Defaults to 'ones'.
            beta_initializer: Initializer for the beta weight.
            gamma_initializer: Initializer for the gamma weight.
            moving_mean_initializer: Initializer for the moving mean.
            moving_variance_initializer: Initializer for the moving variance.
            alpha_regularizer: Optional regularizer for the alpha weight.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.
            alpha_constraint: Optional constraint for the alpha weight.
            use_dropout: Whether to use dropout layers in between.
            rate: Float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
                For instance, if your inputs have shape`(batch_size, timesteps, features)` and
                you want the dropout mask to be the same for all timesteps,
                you can use `noise_shape=(batch_size, 1, features)`.
            seed: A Python integer to use as random seed.
        """
        super(_MLPBase, self).__init__(**kwargs)
        local_kw = locals()
        # List for groups of arguments.
        self._key_list_act = [
            "activation", "activity_regularizer"
        ]
        self._key_list_dense = [
            "units", "use_bias", "kernel_regularizer", "bias_regularizer", "kernel_initializer", "bias_initializer",
            "kernel_constraint", "bias_constraint"
        ]
        self._key_list_norm_all = [
            "axis", "momentum", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
            "moving_mean_initializer", "moving_variance_initializer", "beta_regularizer",
            "gamma_regularizer", "beta_constraint", "gamma_constraint", "alpha_initializer", "alpha_regularizer",
            "alpha_constraint", "mean_shift"
        ]
        self._key_list_dropout = ["rate", "noise_shape", "seed"]
        self._key_list_use = ["use_dropout", "use_normalization", "normalization_technique"]
        self._key_list_init = [
            "kernel_initializer", "bias_initializer", "beta_initializer", "gamma_initializer",
            "moving_mean_initializer", "moving_variance_initializer", "alpha_initializer"
        ]
        self._key_list_reg = [
            "activity_regularizer", "kernel_regularizer", "bias_regularizer", "beta_regularizer", "gamma_regularizer",
            "alpha_regularizer"
        ]
        self._key_list_const = [
            "kernel_constraint", "bias_constraint", "beta_constraint", "gamma_constraint", "alpha_constraint"
        ]
        self._key_list_general = [
            "padded_disjoint"
        ]
        self._key_dict_norm = global_normalization_args

        # Summarize all arguments.
        self._key_list = []
        self._key_list += self._key_list_act + self._key_list_dense + self._key_list_norm_all + self._key_list_dropout
        self._key_list += self._key_list_use + self._key_list_general
        self._key_list = list(set(self._key_list))

        # Dictionary of kwargs for MLP.
        mlp_kwargs = {key: local_kw[key] for key in self._key_list}

        # Everything should be defined by units.
        if isinstance(units, int):
            mlp_kwargs["units"] = [units]
        if not isinstance(mlp_kwargs["units"], list):
            raise ValueError("Units must be a list or a single int for `MLP`.")
        self._depth = len(mlp_kwargs["units"])
        # Special case, if axis is supposed to be multiple axis, use tuple here.
        if not isinstance(axis, list):
            mlp_kwargs["axis"] = [axis for _ in range(self._depth)]
        # Special case, for shape, use tuple here.
        if not isinstance(noise_shape, list):
            mlp_kwargs["noise_shape"] = [noise_shape for _ in range(self._depth)]

        # Assert matching number of args
        def assert_args_is_list(args):
            if not isinstance(args, (list, tuple)):
                return [args for _ in range(self._depth)]
            return args

        # Make every argument to list.
        for key, value in mlp_kwargs.items():
            mlp_kwargs[key] = assert_args_is_list(value)

        # Check correct length for all arguments.
        for key, value in mlp_kwargs.items():
            if self._depth != len(value):
                raise ValueError(
                    "Provide matching list of units '%s' and '%s' or simply a single value." % (
                        mlp_kwargs["units"], key))

        # Deserialize initializer, regularizes, constraints and activation.
        for sl, sm in [
            (self._key_list_init, ks.initializers.get), (self._key_list_reg, ks.regularizers.get),
            (self._key_list_const, ks.constraints.get), (["activation"], ks.activations.get)
        ]:
            for key in sl:
                mlp_kwargs[key] = [sm(x) for x in mlp_kwargs[key]]

        # Fix synonyms for normalization layer.
        replace_norm_identifier = [
            ("batch", "BatchNormalization"), ("layer", "LayerNormalization"), ("group", "GroupNormalization"),
            ("graph", "GraphNormalization"), ("graph_instance", "GraphInstanceNormalization"),
            ("unit_norm", "UnitNormalization"), ("norm", "Normalization"), ("graph_layer", "GraphLayerNormalization"),
            ("graph_batch", "GraphBatchNormalization")
        ]
        for i, x in enumerate(mlp_kwargs["normalization_technique"]):
            for key_rep, key in replace_norm_identifier:
                if x == key_rep:
                    mlp_kwargs["normalization_technique"][i] = key

        # Assign separate '_conf_' for use keys.
        # All '_conf_' kwargs in '_conf_mlp_kwargs'.
        for key in self._key_list_use:
            setattr(self, "_conf_" + key, mlp_kwargs[key])
        self._conf_mlp_kwargs = mlp_kwargs

    def _get_conf_for_keys(self, key_list_to_fetch: list, name_postfix: str, i_layer: int):
        out_kwargs = {key: self._conf_mlp_kwargs[key][i_layer] for key in key_list_to_fetch}
        out_kwargs.update({"name": self.name + "_" + name_postfix + "_" + str(i_layer)})
        return out_kwargs

    def build(self, input_shape):
        """Build layer."""
        super(_MLPBase, self).build(input_shape)

    def get_config(self):
        """Update config."""
        config = super(_MLPBase, self).get_config()
        for key in self._key_list:
            config.update({key: self._conf_mlp_kwargs[key]})

        # Serialize initializer, regularizes, constraints and activation.
        for sl, sm in [
            (self._key_list_init, ks.initializers.serialize), (self._key_list_reg, ks.regularizers.serialize),
            (self._key_list_const, ks.constraints.serialize), (["activation"], ks.activations.serialize)
        ]:
            for key in sl:
                config.update({key: [sm(x) for x in self._conf_mlp_kwargs[key]]})
        return config


class MLP(_MLPBase):  # noqa
    r"""Class for multilayer perceptron that consist of multiple feed-forward networks.

    The class contains arguments for :obj:`Dense` , :obj:`Dropout` and :obj:`BatchNormalization`
    or :obj:`LayerNormalization` or :obj:`GraphNormalization`
    since MLP is made up of stacked :obj:`Dense` layers with optional normalization and
    dropout to improve stability or regularization.
    Here, a list in place of arguments must be provided that applies
    to each layer. If not a list is given, then the single argument is used for each layer.
    The number of layers is determined by :obj:`units` argument, which should be list.

    This class holds arguments for batch-normalization which should be applied between kernel
    and activation. And dropout after the kernel output and before normalization.
    """

    # If child classes want to replace layers.
    _supress_dense = False

    def __init__(self, units, **kwargs):
        r"""Initialize with parameter for MLP layer that match :obj:`Dense` layer, including :obj:`Dropout` and
        :obj:`BatchNormalization` or :obj:`LayerNormalization` or :obj:`GraphNormalization` .

        Args:
            units: Positive integer, dimensionality of the output space.
            %s
        """
        super(MLP, self).__init__(units=units, **kwargs)
        norm_classes = {
            "UnitNormalization": UnitNormalization,
            "BatchNormalization": BatchNormalization,
            "GroupNormalization": GroupNormalization,
            "LayerNormalization": LayerNormalization,
            "GraphNormalization": GraphNormalization,
            "GraphInstanceNormalization": GraphInstanceNormalization,
            "GraphLayerNormalization": GraphLayerNormalization,
            "GraphBatchNormalization": GraphBatchNormalization,
        }
        if not self._supress_dense:
            self.mlp_dense_layer_list = [
                Dense(**self._get_conf_for_keys(
                    self._key_list_dense, "dense", i)) for i in range(self._depth)
            ]
        self.mlp_activation_layer_list = [
            Activation(**self._get_conf_for_keys(
                self._key_list_act, "act", i)) for i in range(self._depth)
        ]
        self.mlp_dropout_layer_list = [
            Dropout(**self._get_conf_for_keys(
                self._key_list_dropout, "drop", i)) if self._conf_use_dropout[i] else None for i
            in range(self._depth)
        ]
        self.mlp_norm_layer_list = [
            norm_classes[self._conf_normalization_technique[i]](
                **self._get_conf_for_keys(self._key_dict_norm[self._conf_normalization_technique[i]], "norm", i)
            ) if self._conf_use_normalization[i] else None for i in range(self._depth)
        ]
        self.is_graph_norm_layer = [
            "Graph" in self._conf_normalization_technique[i] if self._conf_use_normalization[i] else False for i in
            range(self._depth)
        ]

    def build(self, input_shape):
        """Build layer."""
        x_shape, x_graph = (input_shape[0], input_shape[1:]) if isinstance(input_shape, list) else (input_shape, [])
        for i in range(self._depth):
            self.mlp_dense_layer_list[i].build(x_shape)
            x_shape = self.mlp_dense_layer_list[i].compute_output_shape(x_shape)
            if self._conf_use_dropout[i]:
                self.mlp_dropout_layer_list[i].build(x_shape)
            if self._conf_use_normalization[i]:
                norm_shape = x_shape if not self.is_graph_norm_layer[i] else [x_shape] + x_graph
                self.mlp_norm_layer_list[i].build(norm_shape)
            self.mlp_activation_layer_list[i].build(x_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (Tensor): Input tensor with last dimension not `None` .

        Returns:
            Tensor: MLP forward pass.
        """
        x, batch = (inputs[0], inputs[1:]) if isinstance(inputs, list) else (inputs, [])

        for i in range(self._depth):
            x = self.mlp_dense_layer_list[i](x, **kwargs)
            if self._conf_use_dropout[i]:
                x = self.mlp_dropout_layer_list[i](x, **kwargs)
            if self._conf_use_normalization[i]:
                if self.is_graph_norm_layer[i]:
                    x = self.mlp_norm_layer_list[i]([x]+batch, **kwargs)
                else:
                    x = self.mlp_norm_layer_list[i](x, **kwargs)
            x = self.mlp_activation_layer_list[i](x, **kwargs)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(MLP, self).get_config()
        return config


MLP.__init__.__doc__ = MLP.__init__.__doc__ % _MLPBase.__init__.__doc__


# Normal MLP can pass additional tensors for normalization.
# Use as synonym here.
GraphMLP = MLP


class RelationalMLP(MLP):
    r"""Relational MLP which behaves like the standard MLP but uses :obj:`RelationalDense` , which
    applies a specific kernel transformation based on the provided relation.
    """

    _supress_dense = True

    def __init__(self, units, num_relations: int, num_bases: int = None, num_blocks: int = None, **kwargs):
        """Initialize with parameter for MLP layer that match :obj:`Dense` layer, including :obj:`Dropout` and
        :obj:`BatchNormalization` or :obj:`LayerNormalization` or :obj:`GraphNormalization` .

        Args:
            units: Positive integer, dimensionality of the output space.
            num_relations: Number of relations expected to construct weights.
            num_bases: Number of kernel basis functions to construct relations. Default is None.
            num_blocks: Number of block-matrices to get for parameter reduction. Default is None.
            %s
        """
        super(RelationalMLP, self).__init__(units=units, **kwargs)
        self._conf_num_relations = num_relations
        self._conf_num_bases = num_bases
        self._conf_num_blocks = num_blocks
        self._conf_relational_kwargs = {
            "num_relations": self._conf_num_relations, "num_bases": self._conf_num_bases,
            "num_blocks": self._conf_num_blocks
        }

        # Override dense list with RelationalDense layer.
        self.mlp_dense_layer_list = [RelationalDense(
            # **self._conf_mlp_dense_layer_kwargs[i],
            **self._get_conf_for_keys(self._key_list_dense, "dense", i),
            **self._conf_relational_kwargs) for i in range(self._depth)]

    def build(self, input_shape):
        """Build layer."""
        x_shape, r_shape, x_graph = (
            input_shape[0], input_shape[1], input_shape[2:]) if len(input_shape) > 2 else (
            input_shape[0], input_shape[1], [])
        for i in range(self._depth):
            self.mlp_dense_layer_list[i].build([x_shape, r_shape])
            x_shape = self.mlp_dense_layer_list[i].compute_output_shape([x_shape, r_shape])
            if self._conf_use_dropout[i]:
                self.mlp_dropout_layer_list[i].build(x_shape)
            if self._conf_use_normalization[i]:
                norm_shape = x_shape if not self.is_graph_norm_layer[i] else [x_shape] + x_graph
                self.mlp_norm_layer_list[i].build(norm_shape)
            self.mlp_activation_layer_list[i].build(x_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [features, relation]

                - features (Tensor): Input tensor with last dimension not `None` e.g. `(..., N)` .
                - relation (Tensor): Input tensor with relation information of shape e.g. `(..., )` of type 'int'.

        Returns:
            Tensor: MLP forward pass.
        """
        x, relations, batch = (inputs[0], inputs[1], inputs[2:]) if len(inputs) > 2 else (inputs[0], inputs[1], [])
        for i in range(self._depth):
            x = self.mlp_dense_layer_list[i]([x, relations], **kwargs)
            if self._conf_use_dropout[i]:
                x = self.mlp_dropout_layer_list[i](x, **kwargs)
            if self._conf_use_normalization[i]:
                if self.is_graph_norm_layer[i]:
                    x = self.mlp_norm_layer_list[i]([x]+batch, **kwargs)
                else:
                    x = self.mlp_norm_layer_list[i](x, **kwargs)
            x = self.mlp_activation_layer_list[i](x, **kwargs)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(RelationalMLP, self).get_config()
        config.update(self._conf_relational_kwargs)
        return config


RelationalMLP.__init__.__doc__ = RelationalMLP.__init__.__doc__ % _MLPBase.__init__.__doc__