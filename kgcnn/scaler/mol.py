import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Union
from sklearn.linear_model import Ridge
from kgcnn.scaler.scaler import StandardScaler
from kgcnn.data.utils import save_json_file, load_json_file


class ExtensiveMolecularScaler:
    r"""Scaler for extensive properties like energy to remove a simple linear behaviour with additive atom
    contributions. Interface is designed after scikit-learn scaler. Internally Ridge regression ist used.
    Only the atomic number is used as extensive scaler. This could be further improved by also taking bonds and
    interactions into account, e.g. as energy contribution.

    .. code-block:: python

        import numpy as np
        from kgcnn.scaler.mol import ExtensiveMolecularScaler
        data = np.random.rand(5).reshape((5,1))
        mol_num = [np.array([6, 1, 1, 1, 1]), np.array([7, 1, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1]), np.array([6, 6, 1, 1]), np.array([6, 6, 1, 1, 1, 1, 1, 1])
        ]
        scaler = ExtensiveMolecularScaler()
        scaler.fit(X=data, atomic_number=mol_num)
        print(scaler.get_weights())
        print(scaler.get_config())
        scaler._plot_predict(data, mol_num)  # For debugging.
        print(scaler.inverse_transform(scaler.transform(X=data, atomic_number=mol_num), atomic_number=mol_num))
        print(data)
        scaler.save("example.json")
        new_scaler = ExtensiveMolecularScaler()
        new_scaler.load("example.json")
        print(new_scaler.inverse_transform(scaler.transform(X=data, atomic_number=mol_num), atomic_number=mol_num))

    """

    _attributes_list_sklearn = ["n_features_in_", "coef_", "intercept_", "n_iter_", "feature_names_in_"]
    _attributes_list_mol = ["scale_", "_fit_atom_selection", "_fit_atom_selection_mask"]
    max_atomic_number = 95

    def __init__(self, alpha: float = 1e-9, fit_intercept: bool = False, standardize_scale: bool = True, **kwargs):
        r"""Initialize scaler with parameters directly passed to scikit-learns :obj:`Ridge()`.

        Args:
            alpha (float): Regularization parameter for regression.
            fit_intercept (bool): Whether to allow a constant offset per target.
            standardize_scale (bool): Whether to standardize output after offset removal.
            kwargs: Additional arguments passed to :obj:`Ridge()`.
        """

        self.ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        self._standardize_scale = standardize_scale
        self._fit_atom_selection_mask = None
        self._fit_atom_selection = None
        self.scale_ = None

    def fit(self, X, *, y: Union[None, np.ndarray] = None, sample_weight=None, atomic_number=None):
        r"""Fit atomic number to the molecular properties.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            y (np.ndarray): Ignored.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            self
        """
        molecular_property = X
        if atomic_number is None or not isinstance(atomic_number, (list, tuple, np.ndarray)):
            raise ValueError("Please specify kwarg 'atomic_number' for calling fit. Got '%s'." % atomic_number)
        if len(atomic_number) != len(molecular_property):
            raise ValueError(
                "`ExtensiveMolecularScaler` different input shape '{0}' vs. '{1}'.".format(
                    len(atomic_number), len(molecular_property))
            )

        unique_number = [np.unique(x, return_counts=True) for x in atomic_number]
        all_unique = np.unique(np.concatenate([x[0] for x in unique_number], axis=0))
        self._fit_atom_selection = all_unique
        atom_mask = np.zeros(self.max_atomic_number, dtype="bool")
        atom_mask[all_unique] = True
        self._fit_atom_selection_mask = atom_mask
        total_number = []
        for unique_per_mol, num_unique in unique_number:
            array_atoms = np.zeros(self.max_atomic_number)
            array_atoms[unique_per_mol] = num_unique
            positives = array_atoms[atom_mask]
            total_number.append(positives)
        total_number = np.array(total_number)
        self.ridge.fit(total_number, molecular_property, sample_weight=sample_weight)
        diff = molecular_property - self.ridge.predict(total_number)
        if self._standardize_scale:
            self.scale_ = np.std(diff, axis=0)
        else:
            self.scale_ = np.ones(diff.shape[1:], dtype="float")
        return self

    def predict(self, atomic_number):
        """Predict the offset form atomic numbers. Requires :obj:`fit()` called previously.

        Args:
            atomic_number (list): List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].

        Returns:
            np.ndarray: Offset of atomic properties fitted previously. Shape is `(n_samples, n_properties)`.
        """
        if self._fit_atom_selection_mask is None:
            raise ValueError("`ExtensiveMolecularScaler` has not been fitted yet. Can not predict.")
        unique_number = [np.unique(x, return_counts=True) for x in atomic_number]
        total_number = []
        for unique_per_mol, num_unique in unique_number:
            array_atoms = np.zeros(self.max_atomic_number)
            array_atoms[unique_per_mol] = num_unique
            positives = array_atoms[self._fit_atom_selection_mask]
            if np.sum(positives) != np.sum(num_unique):
                print("`ExtensiveMolecularScaler` got unknown atom species in transform.")
            total_number.append(positives)
        total_number = np.array(total_number)
        offset = self.ridge.predict(total_number)
        return offset

    def _plot_predict(self, molecular_property, atomic_number):
        """Debug function to check prediction."""
        molecular_property = np.array(molecular_property)
        if len(molecular_property.shape) <= 1:
            molecular_property = np.expand_dims(molecular_property, axis=-1)
        predict_prop = self.predict(atomic_number)
        if len(predict_prop.shape) <= 1:
            predict_prop = np.expand_dims(predict_prop, axis=-1)
        mae = np.mean(np.abs(molecular_property - predict_prop), axis=0)
        plt.figure()
        for i in range(predict_prop.shape[-1]):
            plt.scatter(predict_prop[:, i], molecular_property[:, i], alpha=0.3,
                        label="Pos: " + str(i) + " MAE: {0:0.4f} ".format(mae[i]))
        plt.plot(np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05),
                 np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05), color='red')
        plt.xlabel('Fitted')
        plt.ylabel('Actual')
        plt.legend(loc='upper left', fontsize='x-small')
        plt.show()

    def transform(self, X, *, y=None, copy=True, atomic_number):
        """Transform any atomic number list with matching properties based on previous fit. Also std-scaled.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            y (np.ndarray): Ignored.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].

        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        if copy:
            X = X - self.predict(atomic_number)
            if self._standardize_scale:
                X = X / np.expand_dims(self.scale_, axis=0)
        else:
            X -= self.predict(atomic_number)
            if self._standardize_scale:
                X /= np.expand_dims(self.scale_, axis=0)
        return X

    def fit_transform(self, X, *, y=None, copy=True, sample_weight=None, atomic_number=None):
        """Combine fit and transform methods in one call.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            y (np.ndarray): Ignored.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        if atomic_number is None:
            raise ValueError("`ExtensiveMolecularScaler` requires 'atomic_number' argument.")
        self.fit(X=X, y=y, atomic_number=atomic_number, sample_weight=sample_weight)
        return self.transform(X=X, copy=copy, atomic_number=atomic_number)

    def inverse_transform(self, X, *, y=None, copy=True, atomic_number):
        """Reverse the transform method to original properties without offset and scaled to original units.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`
            y (np.ndarray): Ignored.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].

        Returns:
            np.ndarray: Original atomic properties. Shape is `(n_samples, n_properties)`.
        """
        if copy:
            if self._standardize_scale:
                X = X * np.expand_dims(self.scale_, axis=0)
            X = X + self.predict(atomic_number)
        else:
            if self._standardize_scale:
                X *= np.expand_dims(self.scale_, axis=0)
            X += self.predict(atomic_number)
        return X

    def get_config(self):
        """Get configuration for scaler."""
        config = {}
        config.update(self.ridge.get_params())
        config.update({"standardize_scale": self._standardize_scale})
        return config

    def set_config(self, config):
        """Set configuration for scaler.

        Args:
            config (dict): Config dictionary.
        """
        self._standardize_scale = config["standardize_scale"]
        config_ridge = {key: value for key, value in config.items() if key not in ["standardize_scale"]}
        self.ridge.set_params(**config_ridge)
        return self

    def get_weights(self) -> dict:
        """Get weights for this scaler after fit."""
        weights = dict()
        for x in self._attributes_list_mol:
            weights.update({x: np.array(getattr(self, x)).tolist()})
        for x in self._attributes_list_sklearn:
            if hasattr(self.ridge, x):
                weights.update({x: np.array(getattr(self.ridge, x)).tolist()})
        return weights

    def set_weights(self, weights: dict):
        """Set weights for this scaler.

        Args:
            weights (dict): Weight dictionary.
        """
        for item, value in weights.items():
            if item in self._attributes_list_mol:
                setattr(self, item, np.array(value))
            elif item in self._attributes_list_sklearn:
                setattr(self.ridge, item, np.array(value))
            else:
                print("`ExtensiveMolecularScaler` got unknown weight '%s'." % item)

    def save_weights(self, file_path: str):
        """Save weights as numpy to file.

        Args:
            file_path: Filepath to save weights.
        """
        weights = self.get_weights()
        # Make them all numpy arrays for save.
        for key, value in weights.items():
            weights[key] = np.array(value)
        if len(weights) > 0:
            np.savez(os.path.splitext(file_path)[0] + ".npz", **weights)
        else:
            print("Error no weights to save for `ExtensiveMolecularScaler`.")

    def get_scaling(self):
        """Get scale of shape (1, n_properties)."""
        if self.scale_ is None:
            return
        return np.expand_dims(self.scale_, axis=0)

    def save(self, file_path: str):
        """Save scaler serialization to file.

        Args:
            file_path: Filepath to save scaler serialization.
        """
        conf = self.get_config()
        weights = self.get_weights()
        full_info = {"class_name": type(self).__name__, "module_name": type(self).__module__,
                     "config": conf, "weights": weights}
        save_json_file(full_info, os.path.splitext(file_path)[0] + ".json")

    def load(self, file_path: str):
        """Load scaler serialization from file.

        Args:
            file_path: Filepath to load scaler serialization.
        """
        full_info = load_json_file(file_path)
        # Could verify class_name and module_name here.
        self.set_config(full_info["config"])
        self.set_weights(full_info["weights"])
        return self


class QMGraphLabelScaler:
    r"""A scaler that scales QM targets differently. For now, the main difference is that intensive and extensive
    properties are scaled differently. In principle, also dipole, polarizability or rotational constants
    could to be standardized differently. Interface is designed after scikit-learn scaler.

    The class is simply a list of separate scaler and scales each target of shape [N_samples, target] with a scaler
    from its list. :obj:`QMGraphLabelScaler` is intended as a scaler list class.

    The scaler uses `y` argument but for compatibility reason will also work with labels assigned to `X`
    argument in :obj:`fit`, :obj:`transform` etc. Each label is passed to the corresponding scaler in list simply
    as first argument without keyword `X` or `y`.

    .. code-block:: python

        import numpy as np
        from kgcnn.scaler.mol import QMGraphLabelScaler, ExtensiveMolecularScaler
        from kgcnn.scaler.scaler import StandardScaler
        data = np.random.rand(10).reshape((5,2))
        mol_num = [np.array([6, 1, 1, 1, 1]), np.array([7, 1, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1]), np.array([6, 6, 1, 1]), np.array([6, 6, 1, 1, 1, 1, 1, 1])
        ]
        scaler = QMGraphLabelScaler([ExtensiveMolecularScaler(), StandardScaler()])
        scaler.fit(y=data, atomic_number=mol_num)
        print(scaler.get_weights())
        print(scaler.get_config())
        print(scaler.inverse_transform(scaler.transform(y=data, atomic_number=mol_num), atomic_number=mol_num))
        print(data)
        scaler.save("example.json")
        new_scaler = QMGraphLabelScaler([ExtensiveMolecularScaler(), StandardScaler()])
        new_scaler.load("example.json")
        print(new_scaler.inverse_transform(scaler.transform(y=data, atomic_number=mol_num), atomic_number=mol_num))

    """

    def __init__(self, scaler: list):

        if not isinstance(scaler, list):
            raise TypeError("Scaler information for `QMGraphLabelScaler` must be list, got '%s'." % scaler)

        self.scaler_list = []
        for x in scaler:
            # TODO: Make a general list and add deserialization.
            if isinstance(x, StandardScaler) or isinstance(x, ExtensiveMolecularScaler):
                self.scaler_list.append(x)
                continue

            # Otherwise, must be serialized version of a scaler.
            if not isinstance(x, dict):
                raise TypeError("Single scaler for `QMGraphLabelScaler` deserialization must be dict, got '%s'." % x)
            if "class_name" not in x:
                raise ValueError("Scaler class for single target must be defined, got '%s'." % x)

            # Pick allowed scaler.
            if x["class_name"] == "StandardScaler":
                self.scaler_list.append(StandardScaler(**x["config"]))
            elif x["class_name"] == "ExtensiveMolecularScaler":
                self.scaler_list.append(ExtensiveMolecularScaler(**x["config"]))
            else:
                raise ValueError("Unsupported scaler '%s'." % x["name"])

    def fit_transform(self, y=None, *, X=None, copy=True, sample_weight=None, atomic_number=None):
        r"""Fit and transform all target labels for QM.

        Args:
            y (np.ndarray): Array of atomic labels of shape `(n_samples, n_labels)`.
            X (np.ndarray): Not used.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            np.ndarray: Transformed labels of shape `(n_samples, n_labels)`.
        """
        self.fit(X=X, y=y, atomic_number=atomic_number, sample_weight=sample_weight)
        return self.transform(X=X, y=y, copy=copy, atomic_number=atomic_number)

    def transform(self, y=None, *, X=None, copy=True, atomic_number: list = None):
        r"""Transform all target labels for QM. Requires :obj:`fit()` called previously.

        Args:
            y (np.ndarray): Array of QM unscaled labels of shape `(n_samples, n_labels)`.
            X (np.ndarray): Not used.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Transformed labels of shape `(n_samples, n_labels)`.
        """
        labels = self._check_input(atomic_number,  X, y)

        if copy:
            out_labels = []
            for i, x in enumerate(self.scaler_list):
                out_labels.append(x.transform(X=labels[:, i:i+1], atomic_number=atomic_number, copy=copy))
            out_labels = np.concatenate(out_labels, axis=-1)
        else:
            for i, x in enumerate(self.scaler_list):
                x.transform(X=labels[:, i:i+1], atomic_number=atomic_number, copy=copy)
            out_labels = labels
        return out_labels

    def fit(self, y=None, *, X=None, sample_weight=None, atomic_number=None):
        r"""Fit scaling of QM graph labels or targets.

        Args:
            y (np.ndarray): Array of atomic labels of shape `(n_samples, n_labels)`.
            X (np.ndarray): Ignored.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            self
        """
        labels = self._check_input(atomic_number, X, y)

        for i, x in enumerate(self.scaler_list):
            x.fit(labels[:, i:i + 1], atomic_number=atomic_number, sample_weight=sample_weight)

        return self

    def inverse_transform(self, y=None, *, X=None, copy: bool = True, atomic_number: list = None):
        r"""Back-transform all target labels for QM.

        Args:
            y (np.ndarray): Array of atomic labels of shape `(n_samples, n_labels)`.
            X (np.ndarray): Not used.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].

        Returns:
            np.ndarray: Back-transformed labels of shape `(n_samples, n_labels)`.
        """
        labels = self._check_input(atomic_number, X, y)

        if copy:
            out_labels = []
            for i, x in enumerate(self.scaler_list):
                out_labels.append(x.inverse_transform(labels[:, i:i + 1], atomic_number=atomic_number, copy=copy))
            out_labels = np.concatenate(out_labels, axis=-1)
        else:
            for i, x in enumerate(self.scaler_list):
                x.inverse_transform(labels[:, i:i + 1], atomic_number=atomic_number, copy=copy)
            out_labels = labels

        return out_labels

    def _check_input(self, node_number, X, y):
        assert X is not None or y is not None, "`QMGraphLabelScaler` did not get properties or labels."
        graph_labels = X if (y is None and X is not None) else y
        assert len(node_number) == len(graph_labels), "`QMGraphLabelScaler` input length does not match."
        assert graph_labels.shape[-1] == len(self.scaler_list), "`QMGraphLabelScaler` got wrong number of labels."
        return graph_labels

    @property
    def scale_(self):
        """Composite scale of all scaler in list."""
        return np.concatenate([x.scale_ for x in self.scaler_list], axis=0)

    def get_scaling(self):
        """Get scale of shape (1, n_properties)."""
        return np.expand_dims(self.scale_, axis=0)

    def get_weights(self):
        """Get weights for this scaler after fit."""
        weights = {"scaler": [x.get_weights() for x in self.scaler_list]}
        return weights

    def set_weights(self, weights: dict):
        """Set weights for this scaler.

        Args:
            weights (dict): Weight dictionary.
        """
        scaler_weights = weights["scaler"]
        for i, x in enumerate(scaler_weights):
            self.scaler_list[i].set_weights(x)

    def save_weights(self, file_path: str):
        """Save weights as numpy to file.

        Args:
            file_path: Filepath to save weights.
        """
        all_weights = {}
        for i, x in enumerate(self.scaler_list):
            w = x.get_weights()
            for key, value in w.items():
                all_weights[str(key) + "_%i" % i] = np.array(value)
        np.savez(os.path.splitext(file_path)[0] + ".npz", **all_weights)

    def get_config(self):
        """Get configuration for scaler."""
        config = {"scaler": [
            {"class_name": type(x).__name__, "module_name": type(x).__module__,
             "config": x.get_config()} for x in self.scaler_list]
        }
        return config

    def set_config(self, config):
        """Set configuration for scaler.

        Args:
            config (dict): Config dictionary.
        """
        scaler_conf = config["scaler"]
        for i, x in enumerate(scaler_conf):
            self.scaler_list[i].set_config(x["config"])
        return self

    def save(self, file_path: str):
        """Save scaler serialization to file.

        Args:
            file_path: Filepath to save scaler serialization.
        """
        conf = self.get_config()
        weights = self.get_weights()
        full_info = {"class_name": type(self).__name__, "module_name": type(self).__module__,
                     "config": conf, "weights": weights}
        save_json_file(full_info, os.path.splitext(file_path)[0] + ".json")

    def load(self, file_path: str):
        """Load scaler serialization from file.

        Args:
            file_path: Filepath to load scaler serialization.
        """
        full_info = load_json_file(file_path)
        # Could verify class_name and module_name here.
        self.set_config(full_info["config"])
        self.set_weights(full_info["weights"])
        return self