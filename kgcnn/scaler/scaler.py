import numpy as np
from sklearn.preprocessing import StandardScaler as StandardScalerSklearn
from kgcnn.data.utils import save_json_file


class StandardScaler(StandardScalerSklearn):

    _attributes_list_sklearn = ["mean_", "scale_", "var_", "n_features_in_", "feature_names_in_", "n_samples_seen_"]

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super(StandardScaler, self).__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def get_config(self):
        config = super(StandardScaler, self).get_params()
        return config

    def get_weights(self) -> dict:
        weight_dict = dict()
        for x in self._attributes_list_sklearn:
            if hasattr(self, x):
                weight_dict.update({x: np.array(getattr(self, x))})
        return weight_dict

    def set_weights(self, weights: dict):
        for item, value in weights.items():
            if item in self._attributes_list_sklearn:
                setattr(self, item, value)
            else:
                print("`StandardScaler` got unknown weight '%s'" % item)

    def save_config(self, file_path: str):
        config = self.get_config()
        save_json_file(config, file_path + ".json")

    def save_weights(self, file_path: str):
        weights = self.get_weights()
        if len(weights) > 0:
            np.savez(file_path + ".npz", **weights)
