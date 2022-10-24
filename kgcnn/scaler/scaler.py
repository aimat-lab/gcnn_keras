import os.path
import numpy as np
from sklearn.preprocessing import StandardScaler as StandardScalerSklearn
from kgcnn.data.utils import save_json_file


class StandardScaler(StandardScalerSklearn):

    _attributes_list_sklearn = ["n_features_in_"]
    _attributes_list_sklearn_np_array = ["mean_", "scale_", "var_", "feature_names_in_", "n_samples_seen_"]

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super(StandardScaler, self).__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X, *, y=None, sample_weight=None, atomic_number=None):
        return super(StandardScaler, self).fit(X=X, y=y, sample_weight=sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None, atomic_number=None):
        return super(StandardScaler, self).partial_fit(X=X, y=y, sample_weight=sample_weight)

    def fit_transform(self, X, y=None, *, atomic_number=None, **fit_params):
        return super(StandardScaler, self).fit_transform(X=X, y=y, **fit_params)

    def transform(self, X, *, copy=None, atomic_number=None):
        return super(StandardScaler, self).transform(X=X, copy=copy)

    def inverse_transform(self, X, *, copy=None, atomic_number=None):
        return super(StandardScaler, self).inverse_transform(X=X, copy=copy)

    def get_config(self):
        config = super(StandardScaler, self).get_params()
        return config

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
                setattr(self, item, value)
            elif item in self._attributes_list_sklearn_np_array:
                setattr(self, item, np.array(value))
            else:
                print("`StandardScaler` got unknown weight '%s'." % item)

    def save_config(self, file_path: str):
        config = self.get_config()
        save_json_file(config, os.path.splitext(file_path)[0] + ".json")

    def save_weights(self, file_path: str):
        weights = self.get_weights()
        # Make them all numpy arrays for save.
        for key, value in weights.items():
            weights[key] = np.array(value)
        if len(weights) > 0:
            np.savez(os.path.splitext(file_path)[0] + ".npz", **weights)
        else:
            print("Error no weights to save.")

    def save(self, file_path: str):
        self.save_config(file_path)
        self.save_weights(file_path)

    def get_scaling(self):
        if not hasattr(self, "scale_"):
            return
        scale = np.array(self.scale_)
        scale = np.expand_dims(scale, axis=0)
        return scale
