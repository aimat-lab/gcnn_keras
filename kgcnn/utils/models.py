import tensorflow.keras as ks
import functools
import pprint


def generate_embedding(inputs, input_shape, embedding_args, embedding_rank=1, **kwargs):
    """Optional node embedding for tensor input.

    Args:
        node_input (tf.Tensor): Input tensor to make embedding for.
        input_node_shape (list): Shape of node input without batch dimension. Either (None, F) or (None, )
        embedding_args (dict): Arguments for embedding layer.

    Returns:
        tf.Tensor: Tensor output.
    """
    if len(input_shape) == embedding_rank:
        n = ks.layers.Embedding(**embedding_args)(inputs)
    else:
        n = inputs
    return n


def generate_node_embedding(node_input, input_node_shape, embedding_args, **kwargs):
    """Optional node embedding for tensor input.

    Args:
        node_input (tf.Tensor): Input tensor to make embedding for.
        input_node_shape (list): Shape of node input without batch dimension. Either (None, F) or (None, )
        embedding_args (dict): Arguments for embedding layer.

    Returns:
        tf.Tensor: Tensor output.
    """
    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(**embedding_args)(node_input)
    else:
        n = node_input
    return n


def generate_edge_embedding(edge_input, input_edge_shape, embedding_args, **kwargs):
    """Optional edge embedding for tensor input.

    Args:
        edge_input (tf.Tensor): Input tensor to make embedding for.
        input_edge_shape (list): Shape of edge input without batch dimension. Either (None, F) or (None, )
        embedding_args (dict): Arguments for embedding layer.

    Returns:
        tf.Tensor: Tensor output.
    """
    if len(input_edge_shape) == 1:
        ed = ks.layers.Embedding(**embedding_args)(edge_input)
    else:
        ed = edge_input
    return ed


def generate_state_embedding(env_input, input_state_shape, embedding_args, **kwargs):
    """Optional state embedding for tensor input.

    Args:
        env_input (tf.Tensor): Input tensor to make embedding for.
        input_state_shape: Shape of state input without batch dimension. Either (F, ) or (, )
        embedding_args (dict): Arguments for embedding layer.

    Returns:
        tf.Tensor: Tensor output.
    """
    if len(input_state_shape) == 0:
        uenv = ks.layers.Embedding(**embedding_args)(env_input)
    else:
        uenv = env_input
    return uenv


def update_model_kwargs_logic(default_kwargs=None, user_kwargs=None):
    """Make model parameter dictionary with updated default values.

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
    def model_update_decorator(func):
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

    @classmethod
    def make_model(cls, model_id, dataset_name=None):
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
        else:
            raise NotImplementedError("ERROR:kgcnn: Unknown model identifier %s" % model_id)

        return make_model
