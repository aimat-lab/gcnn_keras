import tensorflow.keras as ks
import functools
import pprint


def generate_embedding(inputs, input_shape: list, embedding_args: dict, embedding_rank: int = 1, **kwargs):
    """Optional embedding for tensor input.

    Args:
        inputs (tf.Tensor): Input tensor to make embedding for.
        input_shape (list, tuple): Shape of input without batch dimension. Either (None, F) or (None, ).
        embedding_args (dict): Arguments for embedding layer.
        embedding_rank (int): The rank of the input which requires embedding. Default is 1.

    Returns:
        tf.Tensor: Tensor output.
    """
    print("WARNING:kgcnn: Unknown embedding kwargs {0}. Not supported yet.".format(kwargs))

    if len(input_shape) == embedding_rank:
        n = ks.layers.Embedding(**embedding_args)(inputs)
    else:
        n = inputs
    return n


def update_model_kwargs_logic(default_kwargs: dict = None, user_kwargs: dict = None):
    """Make model parameter dictionary with updated default values. This is a nested version of update() for dicts.

    Args:
        default_kwargs (dict): Dictionary of default values.
        user_kwargs (dict): Dictionary of args to update.

    Returns:
        dict: Make new dict and update with first default and then user args.
    """
    out = {}
    if default_kwargs is None:
        default_kwargs = {}
    if user_kwargs is None:
        user_kwargs = {}

    # Check valid args
    for iter_key in user_kwargs.keys():
        if iter_key not in default_kwargs:
            raise ValueError("Model arg", iter_key, "not in default arguments", default_kwargs.keys())

    out.update(default_kwargs)

    # Nested update of args:
    def _nested_update(dict1, dict2):
        for key, values in dict2.items():
            if key not in dict1:
                print("WARNING: Unknown model argument:", key, "with value", values)
                dict1[key] = values
            else:
                if isinstance(dict1[key], dict) and isinstance(values, dict):
                    # The value is a dict of model arguments itself. Update the same way.
                    dict1[key] = _nested_update(dict1[key], values)
                elif isinstance(dict1[key], dict) and not isinstance(values, dict):
                    # If values is None, means no information, keep dict1 values untouched.
                    if values is not None:
                        raise ValueError("Error: Can not overwriting dictionary of", key, "with", values)
                else:
                    # Just any other value to update
                    dict1[key] = values
        return dict1

    return _nested_update(out, user_kwargs)


def update_model_kwargs(model_default):
    """Decorating function for a kwargs input to be updated."""
    def model_update_decorator(func):
        """Decorate function."""
        @functools.wraps(func)
        def update_wrapper(*args, **kwargs):
            updated_kwargs = update_model_kwargs_logic(model_default, kwargs)
            if updated_kwargs['verbose'] > 0:
                print("INFO:kgcnn: Updated functional make model kwargs:")
                pprint.pprint(updated_kwargs)
            return func(*args, **updated_kwargs)

        return update_wrapper

    return model_update_decorator


class ModelSelection:
    """Select a model by string identifier. Since most models are created by the functional API, we do not register
    them but have an import by identifier here."""

    @classmethod
    def make_model(cls, model_id: str):
        """Return a make_model function that generates model instances.

        Args:
            model_id (str): Name of the model.

        Returns:
            func: Function to make model instances.
        """
        if model_id == "Schnet":
            from kgcnn.literature.Schnet import make_model
        elif model_id == "GraphSAGE":
            from kgcnn.literature.GraphSAGE import make_model
        elif model_id == "INorp":
            from kgcnn.literature.INorp import make_model
        elif model_id == "Unet":
            from kgcnn.literature.Unet import make_model
        elif model_id == "PAiNN":
            from kgcnn.literature.PAiNN import make_model
        elif model_id == "Megnet":
            from kgcnn.literature.Megnet import make_model
        elif model_id == "GIN":
            from kgcnn.literature.GIN import make_model
        elif model_id == "GraphSAGE":
            from kgcnn.literature.GraphSAGE import make_model
        elif model_id == "AttentiveFP":
            from kgcnn.literature.AttentiveFP import make_model
        elif model_id == "GCN":
            from kgcnn.literature.GCN import make_model
        elif model_id == "GAT":
            from kgcnn.literature.GAT import make_model
        elif model_id == "DimeNetPP":
            from kgcnn.literature.DimeNetPP import make_model
        else:
            raise NotImplementedError("ERROR:kgcnn: Unknown model identifier %s" % model_id)

        return make_model
