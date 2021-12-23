import importlib


class ModelSelection:
    """Select a model by string identifier. Since most models are created by the functional API, we do not register
    them but have an import by identifier here."""

    def __init__(self, model_name: str = None):
        """Initialize a Model-selection for a specific model.

        Args:
            model_name (str):  Model name. Default is None
        """
        self._model_name = None
        if model_name is not None:
            self._model_name = model_name

    def make_model(self, model_id: str = None):
        """Return a make_model function that generates model instances from model-specific hyper-parameters.

        Args:
            model_id (str): Name of the model.

        Returns:
            func: Function to make model instances of tf.keras.Model.
        """
        if model_id is None:
            model_id = self._model_name

        try:
            make_model = getattr(importlib.import_module("kgcnn.literature.%s" % model_id), "make_model")

        except ModuleNotFoundError:
            raise NotImplementedError("Unknown model identifier %s for a model in kgcnn.literature" % model_id)

        return make_model

    def __call__(self, **model_kwargs):
        """Make a model for with model kwargs.

        Args:
            model_kwargs (dict): Model kwargs.

        Returns:
            tf.keras.models.Model
        """
        return self.make_model()(**model_kwargs)