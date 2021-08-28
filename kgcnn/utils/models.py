import tensorflow.keras as ks
import functools
import pprint
import importlib


def generate_embedding(inputs, input_shape: list, embedding_args: dict, embedding_rank: int = 1, **kwargs):
    """Optional embedding for tensor input. If there is no feature dimension, an embedding layer can be used.
    If the input tensor has without batch dimension the shape of e.g. `(None, F)` and `F` is the feature dimension,
    no embedding layer is required. However, for shape `(None, )` an embedding with `output_dim` assures a vector
    representation.

    Args:
        inputs (tf.Tensor): Input tensor to make embedding for.
        input_shape (list, tuple): Shape of input without batch dimension. Either (None, F) or (None, ).
        embedding_args (dict): Arguments for embedding layer which will be unpacked in layer constructor.
        embedding_rank (int): The rank of the input which requires embedding. Default is 1.

    Returns:
        tf.Tensor: Tensor embedding dependent on the input shape.
    """
    if len(kwargs) > 0:
        print("WARNING:kgcnn: Unknown embedding kwargs {0}. Will be reserved for future versions.".format(kwargs))

    if len(input_shape) == embedding_rank:
        n = ks.layers.Embedding(**embedding_args)(inputs)
    else:
        n = inputs
    return n


def update_model_kwargs_logic(default_kwargs: dict = None, user_kwargs: dict = None):
    """Make model kwargs dictionary with updated default values. This is essentially a nested version of update()
    for dicts. This is supposed to be more convenient if the values of kwargs are again layer kwargs to be unpacked,
    and do not need to be fully known to update them.

    Args:
        default_kwargs (dict): Dictionary of default values. Default is None.
        user_kwargs (dict): Dictionary of args to update. Default is None.

    Returns:
        dict: New dict and update with first default and then user args.
    """
    out = {}
    if default_kwargs is None:
        default_kwargs = {}
    if user_kwargs is None:
        user_kwargs = {}

    # Check valid kwargs
    for iter_key in user_kwargs.keys():
        if iter_key not in default_kwargs:
            raise ValueError("Model kwarg {0} not in default arguments {1}".format(iter_key, default_kwargs.keys()))

    out.update(default_kwargs)

    # Nested update of kwargs:
    def _nested_update(dict1, dict2):
        for key, values in dict2.items():
            if key not in dict1:
                print("WARNING:kgcnn: Unknown model kwarg {0} with value {1}".format(key, values))
                dict1[key] = values
            else:
                if isinstance(dict1[key], dict) and isinstance(values, dict):
                    # The value is a dict of model arguments itself. Update the same way.
                    dict1[key] = _nested_update(dict1[key], values)
                elif isinstance(dict1[key], dict) and not isinstance(values, dict):
                    # If values is None, means no information, keep dict1 values untouched.
                    if values is not None:
                        raise ValueError("Can not overwriting dictionary of {0} with {1}".format(key, values))
                else:
                    # Just any other value to update
                    dict1[key] = values
        return dict1

    return _nested_update(out, user_kwargs)


def update_model_kwargs(model_default):
    """Decorating function for update_model_kwargs_logic() ."""
    def model_update_decorator(func):

        @functools.wraps(func)
        def update_wrapper(*args, **kwargs):
            updated_kwargs = update_model_kwargs_logic(model_default, kwargs)
            if 'verbose' in updated_kwargs:
                if updated_kwargs['verbose'] > 0:
                    # Print out the full updated kwargs
                    print("INFO:kgcnn: Updated model kwargs:")
                    pprint.pprint(updated_kwargs)

            return func(*args, **updated_kwargs)

        return update_wrapper

    return model_update_decorator


class ModelSelection:
    """Select a model by string identifier. Since most models are created by the functional API, we do not register
    them but have an import by identifier here."""

    @classmethod
    def make_model(cls, model_id: str):
        """Return a make_model function that generates model instances from model-specific hyper-parameters.

        Args:
            model_id (str): Name of the model.

        Returns:
            func: Function to make model instances of tf.keras.Model.
        """
        try:
            make_model = getattr(importlib.import_module("kgcnn.literature.%s" % model_id), "make_model")

        except ModuleNotFoundError:
            raise NotImplementedError("ERROR:kgcnn: Unknown model identifier %s" % model_id)

        return make_model
