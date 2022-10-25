import os.path
import numpy as np
from sklearn.preprocessing import StandardScaler as StandardScalerSklearn
from kgcnn.data.utils import save_json_file, load_json_file


class StandardScaler(StandardScalerSklearn):

    _attributes_list_sklearn = ["n_features_in_", "mean_", "scale_", "var_", "feature_names_in_", "n_samples_seen_"]

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super(StandardScaler, self).__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X, *, y=None, sample_weight=None, atomic_number=None):
        return super(StandardScaler, self).fit(X=X, y=y, sample_weight=sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None, atomic_number=None):
        # For partial fit internally uses args and not kwargs.
        return super(StandardScaler, self).partial_fit(X=X, y=y, sample_weight=sample_weight)

    def fit_transform(self, X, *, y=None, atomic_number=None, **fit_params):
        return super(StandardScaler, self).fit_transform(X=X, y=y, **fit_params)

    def transform(self, X, *, copy=None, atomic_number=None):
        return super(StandardScaler, self).transform(X=X, copy=copy)

    def inverse_transform(self, X, *, copy=None, atomic_number=None):
        return super(StandardScaler, self).inverse_transform(X=X, copy=copy)

    def get_scaling(self):
        if not hasattr(self, "scale_"):
            return
        scale = np.array(self.scale_)
        scale = np.expand_dims(scale, axis=0)
        return scale

    def get_config(self):
        config = super(StandardScaler, self).get_params()
        return config

    def set_config(self, config):
        self.set_params(**config)

    def get_weights(self) -> dict:
        weight_dict = dict()
        for x in self._attributes_list_sklearn:
            if hasattr(self, x):
                value = getattr(self, x)
                value_update = {x: np.array(value).tolist()} if value is not None else {x: value}
                weight_dict.update(value_update)
        return weight_dict

    def set_weights(self, weights: dict):
        for item, value in weights.items():
            if item in self._attributes_list_sklearn:
                setattr(self, item, np.array(value))
            else:
                print("`StandardScaler` got unknown weight '%s'." % item)

    def save_weights(self, file_path: str):
        weights = self.get_weights()
        # Make them all numpy arrays for save.
        for key, value in weights.items():
            weights[key] = np.array(value)
        if len(weights) > 0:
            np.savez(os.path.splitext(file_path)[0] + ".npz", **weights)
        else:
            print("Error no weights to save for `StandardScaler`.")

    def save(self, file_path: str):
        conf = self.get_config()
        weights = self.get_weights()
        full_info = {"class_name": type(self).__name__, "module_name": type(self).__module__,
                     "config": conf, "weights": weights}
        save_json_file(full_info, os.path.splitext(file_path)[0] + ".json")

    def load(self, file_path: str):
        full_info = load_json_file(file_path)
        # Could verify class_name and module_name here.
        self.set_config(full_info["config"])
        self.set_weights(full_info["weights"])