import importlib


class ModelSelection:
    """Select a model by string identifier. Since most models are created by the functional API, we do not register
    them but have an import by identifier here.

    """

    def __init__(self, model_name: str = None, make_function: str = None):
        """Initialize a Model-selection for a specific model.

        Args:
            model_name (str): Model name. Default is None
            make_function (str): Name of the make function.
        """
        self._model_name = None
        self._make_function = "make_model"
        if model_name is not None:
            self._model_name = model_name
        if make_function is not None:
            self._make_function = make_function

    def make_model(self, model_id: str = None, make_function: str = None):
        r"""Return a make_model function that generates model instances from model-specific hyperparameter.

        Args:
            model_id (str): Name of the model.
            make_function (str): Name of the make function for `model_id`.

        Returns:
            func: Function to make model instances of :obj:`tf.keras.Model`.
        """
        if model_id is None:
            model_id = self._model_name
        if make_function is None:
            make_function = self._make_function

        try:
            make_model = getattr(importlib.import_module("kgcnn.literature.%s" % model_id), make_function)

        except ModuleNotFoundError:
            raise NotImplementedError("Unknown model identifier %s for a model in kgcnn.literature" % model_id)

        return make_model

    def __call__(self, **model_kwargs):
        r"""Make a model from :obj:`self` with model kwargs.

        Args:
            model_kwargs (dict): Model kwargs.

        Returns:
            :obj:`tf.keras.models.Model`
        """
        return self.make_model()(**model_kwargs)