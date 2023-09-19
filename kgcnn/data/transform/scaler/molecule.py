import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Union, List, Dict
from sklearn.linear_model import Ridge
from kgcnn.data.utils import save_json_file, load_json_file
from kgcnn.data.transform.scaler.serial import deserialize


class _ExtensiveMolecularScalerBase:
    """Scaler base class for extensive properties like energy to remove a simple linear behaviour with additive atom
    contributions.
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
        self._molecular_property = None
        self._atomic_number = None
        self._sample_weight = None

    def _fit(self, molecular_property, atomic_number, sample_weight=None):
        r"""Fit atomic number to the molecular properties.

        Args:
            molecular_property (np.ndarray): Molecular properties of shape `(n_samples, n_properties)` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            self
        """
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

    def _predict(self, atomic_number):
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

    def _plot_predict(self, molecular_property: np.ndarray, atomic_number: List[np.ndarray]):
        """Debug function to check prediction."""
        molecular_property = np.array(molecular_property)
        if len(molecular_property.shape) <= 1:
            molecular_property = np.expand_dims(molecular_property, axis=-1)
        predict_prop = self._predict(atomic_number)
        if len(predict_prop.shape) <= 1:
            predict_prop = np.expand_dims(predict_prop, axis=-1)
        mae = np.mean(np.abs(molecular_property - predict_prop), axis=0)
        fig = plt.figure()
        for i in range(predict_prop.shape[-1]):
            plt.scatter(predict_prop[:, i], molecular_property[:, i], alpha=0.3,
                        label="Pos: " + str(i) + " MAE: {0:0.4f} ".format(mae[i]))
        plt.plot(np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05),
                 np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05), color='red')
        plt.xlabel('Fitted')
        plt.ylabel('Actual')
        plt.legend(loc='upper left', fontsize='x-small')
        plt.show()
        return fig

    def _transform(self, molecular_property, atomic_number, copy=True):
        """Transform any atomic number list with matching properties based on previous fit with sequential std-scaling.

        Args:
            molecular_property (np.ndarray): Molecular properties of shape `(n_samples, n_properties)` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.

        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        if copy:
            molecular_property = molecular_property - self._predict(atomic_number)
            if self._standardize_scale:
                molecular_property = molecular_property / np.expand_dims(self.scale_, axis=0)
        else:
            molecular_property -= self._predict(atomic_number)
            if self._standardize_scale:
                molecular_property /= np.expand_dims(self.scale_, axis=0)
        return molecular_property

    def _fit_transform(self, molecular_property, atomic_number, copy=True, sample_weight=None):
        """Combine fit and transform methods in one call.

        Args:
            molecular_property (np.ndarray): Molecular properties of shape `(n_samples, n_properties)` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        self._fit(molecular_property=molecular_property, atomic_number=atomic_number, sample_weight=sample_weight)
        return self._transform(molecular_property=molecular_property, atomic_number=atomic_number, copy=copy)

    def _inverse_transform(self, molecular_property: np.ndarray, atomic_number: List[np.ndarray],
                           copy: bool = True) -> np.ndarray:
        """Reverse the transform method to original properties without offset removed and scaled to original units.

        Args:
            molecular_property (np.ndarray): Molecular properties of shape `(n_samples, n_properties)` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.

        Returns:
            np.ndarray: Original atomic properties. Shape is `(n_samples, n_properties)`.
        """
        if copy:
            if self._standardize_scale:
                molecular_property = molecular_property * np.expand_dims(self.scale_, axis=0)
            molecular_property = molecular_property + self._predict(atomic_number)
        else:
            if self._standardize_scale:
                molecular_property *= np.expand_dims(self.scale_, axis=0)
            molecular_property += self._predict(atomic_number)
        return molecular_property

    def get_config(self) -> dict:
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

    # Similar functions that work on dataset plus property names.
    # noinspection PyPep8Naming
    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]]):
        r"""Fit to dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.

        Returns:
            self.
        """
        return self._fit(
            molecular_property=np.array([item[self._molecular_property] for item in dataset]),
            atomic_number=[item[self._atomic_number] for item in dataset],
            sample_weight=[item[self._sample_weight] for item in dataset] if self._sample_weight is not None else None
        )

    # noinspection PyPep8Naming
    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          copy: bool = True,
                          copy_dataset: bool = False,
                          ) -> List[Dict[str, np.ndarray]]:
        r"""Transform dataset with relevant `X` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.
            copy (bool): Whether to copy data for transformation. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Transformed dataset.
        """
        if copy_dataset:
            dataset = dataset.copy()
        out = self._transform(
            molecular_property=np.array([item[self._molecular_property] for item in dataset]),
            atomic_number=[item[self._atomic_number] for item in dataset],
            copy=copy,
        )
        for graph, out_value in zip(dataset, out):
            graph[self._molecular_property] = out_value
        return dataset

    # noinspection PyPep8Naming
    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  copy: bool = True,
                                  copy_dataset: bool = False,
                                  ) -> List[Dict[str, np.ndarray]]:
        r"""Inverse transform dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.
            copy (bool): Whether to copy data for transformation. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Inverse-transformed dataset.
        """
        if copy_dataset:
            dataset = dataset.copy()
        out = self._inverse_transform(
            molecular_property=np.array([item[self._molecular_property] for item in dataset]),
            atomic_number=[item[self._atomic_number] for item in dataset],
            copy=copy,
        )
        for graph, out_value in zip(dataset, out):
            graph[self._molecular_property] = out_value
        return dataset

    # noinspection PyPep8Naming
    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                              copy: bool = True,
                              copy_dataset: bool = False
                              ) -> List[Dict[str, np.ndarray]]:
        r"""Fit and transform to dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.
            copy (bool): Whether to copy data for transformation. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Transformed dataset.
        """
        self.fit_dataset(dataset=dataset)
        return self.transform_dataset(dataset=dataset, copy=copy, copy_dataset=copy_dataset)


class ExtensiveMolecularScaler(_ExtensiveMolecularScalerBase):
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
        print(scaler.inverse_transform(scaler.transform(X=data, atomic_number=mol_num), atomic_number=mol_num))

    """
    # noinspection PyPep8Naming
    def __init__(self, X: str = "graph_attributes", atomic_number: str = "atomic_number", sample_weight: str = None,
                 **kwargs):
        super(ExtensiveMolecularScaler, self).__init__(**kwargs)
        self._molecular_property = X
        self._atomic_number = atomic_number
        self._sample_weight = sample_weight

    # noinspection PyPep8Naming
    def fit(self, X, *, y: Union[None, np.ndarray] = None, sample_weight=None, atomic_number=None):
        r"""Fit atomic number to the molecular properties.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            y (np.ndarray): Ignored.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            self.
        """
        return super(ExtensiveMolecularScaler, self)._fit(
            molecular_property=X, atomic_number=atomic_number, sample_weight=sample_weight)

    # noinspection PyPep8Naming
    def transform(self, X, *, y=None, copy=True, atomic_number=None):
        """Transform any atomic number list with matching properties based on previous fit with sequential std-scaling.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            y (np.ndarray): Ignored.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.

        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        return super(ExtensiveMolecularScaler, self)._transform(
            molecular_property=X, atomic_number=atomic_number, copy=copy)

    # noinspection PyPep8Naming
    def fit_transform(self, X, *, y=None, copy=True, atomic_number=None, sample_weight=None):
        r"""Fit and transform.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            y (np.ndarray): Ignored.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            np.ndarray: Transformed properties.
        """
        self.fit(X=X, y=y, sample_weight=sample_weight, atomic_number=atomic_number)
        return self.transform(X=X, y=y, copy=copy, atomic_number=atomic_number)

    # noinspection PyPep8Naming
    def inverse_transform(self, X, *, y=None, copy=True, atomic_number=None):
        """Reverse the transform method to original properties without offset removed and scaled to original units.

        Args:
            X (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            y (np.ndarray): Ignored.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.

        Returns:
            np.ndarray: Original atomic properties. Shape is `(n_samples, n_properties)`.
        """

        return super(ExtensiveMolecularScaler, self)._inverse_transform(
            molecular_property=X, atomic_number=atomic_number, copy=copy)

    def get_config(self):
        config = super(ExtensiveMolecularScaler, self).get_config()
        config.update({"X": self._molecular_property, "atomic_number": self._atomic_number,
                       "sample_weight": self._sample_weight})
        return config

    def set_config(self, config):
        super(ExtensiveMolecularScaler, self).set_config(config)
        self._molecular_property = config["X"]
        self._atomic_number = config["atomic_number"]
        self._sample_weight = config["sample_weight"]


class ExtensiveMolecularLabelScaler(_ExtensiveMolecularScalerBase):
    r"""Equivalent of :obj:`ExtensiveMolecularScaler` for labels, which uses the `y` argument for labels.
    For `X` the atomic numbers can be passed.

    .. code-block:: python

        import numpy as np
        from kgcnn.scaler.mol import ExtensiveMolecularLabelScaler
        data = np.random.rand(5).reshape((5,1))
        mol_num = [np.array([6, 1, 1, 1, 1]), np.array([7, 1, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1]), np.array([6, 6, 1, 1]), np.array([6, 6, 1, 1, 1, 1, 1, 1])
        ]
        scaler = ExtensiveMolecularLabelScaler()
        scaler.fit(X=mol_num, y=data)
        print(scaler.get_weights())
        print(scaler.get_config())
        scaler._plot_predict(data, mol_num)  # For debugging.
        print(scaler.inverse_transform(X=mol_num, y=scaler.transform(X=mol_num, y=data)))
        print(data)
        scaler.save("example.json")
        new_scaler = ExtensiveMolecularLabelScaler()
        new_scaler.load("example.json")
        print(scaler.inverse_transform(X=mol_num, y=scaler.transform(X=mol_num, y=data)))

    """
    # noinspection PyPep8Naming
    def __init__(self, y: str = "graph_labels", atomic_number: str = "atomic_number", sample_weight: str = None,
                 **kwargs):
        super(ExtensiveMolecularLabelScaler, self).__init__(**kwargs)
        self._molecular_property = y
        self._atomic_number = atomic_number
        self._sample_weight = sample_weight

    def _assert_has_y(self, y):
        if y is None:
            raise ValueError(
                "Require labels in `y` for `%s`. Input must be e.g. 'fit(y=data)'." % type(self).__name__)

    # noinspection PyPep8Naming
    def fit(self, y: Union[None, list, np.ndarray] = None, *, X=None, sample_weight=None, atomic_number=None):
        """Fit labels with atomic number information.

        Args:
            y: Array of atomic labels of shape `(n_samples, n_labels)`.
            X: List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Optional, since they should be contained in `X` . Note that if assigning `atomic_numbers`
                then `X` is ignored.
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            np.ndarray: Transformed y.
        """
        self._assert_has_y(y)
        atomic_number = atomic_number if atomic_number else X
        return super(ExtensiveMolecularLabelScaler, self)._fit(
            molecular_property=y, sample_weight=sample_weight, atomic_number=atomic_number)

    # noinspection PyPep8Naming
    def transform(self, y=None, *, X, copy=True, atomic_number=None):
        """Transform any atomic number list with matching labels based on previous fit with sequential std-scaling.

        Args:
            y: Array of atomic labels of shape `(n_samples, n_labels)`.
            X: List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Optional, since they should be contained in `X` . Note that if assigning `atomic_numbers`
                then `X` is ignored.

        Returns:
            np.ndarray: Transformed y.
        """
        self._assert_has_y(y)
        atomic_number = atomic_number if atomic_number else X
        return super(ExtensiveMolecularLabelScaler, self)._transform(
            molecular_property=y, atomic_number=atomic_number, copy=copy)

    # noinspection PyPep8Naming
    def fit_transform(self, y=None, *, X=None, copy=True, atomic_number=None, sample_weight=None):
        """Fit and transform.

        Args:
            y: Array of atomic labels of shape `(n_samples, n_labels)`.
            X: List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Optional, since they should be contained in `X` . Note that if assigning `atomic_numbers`
                then `X` is ignored.
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:

        """
        self.fit(y=y, X=X, sample_weight=sample_weight, atomic_number=atomic_number)
        return self.transform(y=y, X=X, copy=copy, atomic_number=atomic_number)

    # noinspection PyPep8Naming
    def inverse_transform(self, y=None, *, X=None, copy=True, atomic_number=None):
        """Reverse the transform method to original labels without offset removed and scaled to original units.

        Args:
            y: Array of atomic labels of shape `(n_samples, n_labels)`.
            X: List of array of atomic numbers. Example [np.array([7,1,1,1]), ...].
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Optional, since they should be contained in `X` . Note that if assigning `atomic_numbers`
                then `X` is ignored.

        Returns:
            np.ndarray: Transformed y.
        """
        self._assert_has_y(y)
        atomic_number = atomic_number if atomic_number else X
        return super(ExtensiveMolecularLabelScaler, self)._inverse_transform(
            molecular_property=y, atomic_number=atomic_number, copy=copy)

    def get_config(self):
        config = super(ExtensiveMolecularLabelScaler, self).get_config()
        config.update({"y": self._molecular_property, "atomic_number": self._atomic_number,
                       "sample_weight": self._sample_weight})
        return config

    def set_config(self, config):
        super(ExtensiveMolecularLabelScaler, self).set_config(config)
        self._molecular_property = config["y"]
        self._atomic_number = config["atomic_number"]
        self._sample_weight = config["sample_weight"]


class QMGraphLabelScaler:
    r"""A scaler that scales QM targets differently. For now, the main difference is that intensive and extensive
    properties are scaled differently. In principle, also dipole, polarizability or rotational constants
    could to be standardized differently. Interface is designed after scikit-learn scaler.

    The class is simply a list of separate scaler and scales each target of shape [N_samples, target] with a scaler
    from its list. :obj:`QMGraphLabelScaler` is intended as a scaler list class.

    .. note::

        The scaler uses `y` argument

    Each label is passed to the corresponding scaler in list simply as first argument without keyword `X` or `y`.

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

    # noinspection PyPep8Naming
    def __init__(self, scaler: list, y: str = "graph_labels", X: str = None, atomic_number: str = "atomic_number",
                 sample_weight: str = None):

        if not isinstance(scaler, list):
            raise TypeError("Scaler information for `QMGraphLabelScaler` must be list, got '%s'." % scaler)

        self.scaler_list = []
        for x in scaler:
            if hasattr(x, "transform") and hasattr(x, "fit"):
                self.scaler_list.append(x)
            elif isinstance(x, dict):
                # Otherwise, must be serialized version of a scaler.
                self.scaler_list.append(deserialize(x))
            else:
                raise ValueError("Unsupported scaler type '%s'." % x)
        self._n_X = X
        self._n_y = y
        self._n_atomic_number = atomic_number
        self._n_sample_weight = sample_weight

    # noinspection PyPep8Naming
    def fit_transform(self, y: Union[np.ndarray, List[np.ndarray]] = None,
                      X: Union[np.ndarray, List[np.ndarray], None] = None,
                      atomic_number: Union[np.ndarray, List[np.ndarray], None] = None,
                      copy: bool = True,
                      sample_weight=None):
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
        self.fit(y=y, X=X, atomic_number=atomic_number, sample_weight=sample_weight)
        return self.transform(y=y, X=X, copy=copy, atomic_number=atomic_number)

    # noinspection PyPep8Naming
    def transform(self, y: Union[np.ndarray, List[np.ndarray]] = None,
                  X: Union[np.ndarray, List[np.ndarray], None] = None,
                  atomic_number: Union[np.ndarray, List[np.ndarray], None] = None,
                  copy=True):
        r"""Transform all target labels for QM. Requires :obj:`fit()` called previously.

        Args:
            y (np.ndarray): Array of QM unscaled labels of shape `(n_samples, n_labels)`.
            X (np.ndarray): Not used.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Transformed labels of shape `(n_samples, n_labels)`.
        """
        labels, atomic_number = self._check_input(atomic_number, X, y)

        if copy:
            out_labels = []
            for i, x in enumerate(self.scaler_list):
                out_labels.append(x.transform([d[i:i + 1] for d in labels], atomic_number=atomic_number, copy=copy))
            out_labels = np.concatenate(out_labels, axis=-1)
        else:
            for i, x in enumerate(self.scaler_list):
                x.transform([d[i:i + 1] for d in labels], atomic_number=atomic_number, copy=copy)
            out_labels = labels
        return out_labels

    # noinspection PyPep8Naming
    def fit(self, y: Union[np.ndarray, List[np.ndarray]] = None,
            X: Union[np.ndarray, List[np.ndarray], None] = None,
            atomic_number: Union[np.ndarray, List[np.ndarray], None] = None,
            sample_weight=None):
        r"""Fit scaling of QM graph labels or targets.

        Args:
            y (np.ndarray): Array of atomic labels of shape `(n_samples, n_labels)`.
            X (np.ndarray): Ignored.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            self
        """
        labels, atomic_number = self._check_input(atomic_number, X, y)

        for i, x in enumerate(self.scaler_list):
            x.fit([d[i:i + 1] for d in labels], atomic_number=atomic_number, sample_weight=sample_weight)

        return self

    # noinspection PyPep8Naming
    def inverse_transform(self, y: Union[np.ndarray, List[np.ndarray]] = None,
                          X: Union[np.ndarray, List[np.ndarray], None] = None,
                          atomic_number: Union[np.ndarray, List[np.ndarray], None] = None,
                          copy: bool = True):
        r"""Back-transform all target labels for QM.

        Args:
            y (np.ndarray): Array of atomic labels of shape `(n_samples, n_labels)`.
            X (np.ndarray): Not used.
            copy (bool): Whether to copy or change in place.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].

        Returns:
            np.ndarray: Back-transformed labels of shape `(n_samples, n_labels)`.
        """
        labels, atomic_number = self._check_input(atomic_number, X, y)

        if copy:
            out_labels = []
            for i, x in enumerate(self.scaler_list):
                out_labels.append(x.inverse_transform([d[i:i + 1] for d in labels], atomic_number=atomic_number, copy=copy))
            out_labels = np.concatenate(out_labels, axis=-1)
        else:
            for i, x in enumerate(self.scaler_list):
                x.inverse_transform([d[i:i + 1] for d in labels], atomic_number=atomic_number, copy=copy)
            out_labels = labels

        return out_labels

    # noinspection PyPep8Naming
    def _check_input(self, node_number, X, y):
        assert X is not None or y is not None, "`QMGraphLabelScaler` did not get properties or labels."
        graph_labels = X if (y is None and node_number is not None) else y
        node_number = node_number if node_number is not None else X
        assert len(node_number) == len(graph_labels), "`QMGraphLabelScaler` input length does not match."
        assert len(graph_labels[0]) == len(self.scaler_list), "`QMGraphLabelScaler` got wrong number of labels."
        return graph_labels, node_number

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
        config.update({"X": self._n_X, "y": self._n_y, "atomic_number": self._n_atomic_number,
                       "sample_weight": self._n_sample_weight})
        return config

    def set_config(self, config):
        """Set configuration for scaler.

        Args:
            config (dict): Config dictionary.
        """
        scaler_conf = config["scaler"]
        for i, x in enumerate(scaler_conf):
            self.scaler_list[i].set_config(x["config"])
        self._n_X = config["X"]
        self._n_y = config["y"]
        self._n_atomic_number = config["atomic_number"]
        self._n_sample_weight = config["sample_weight"]
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

    # Similar functions that work on dataset plus property names.
    # noinspection PyPep8Naming
    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]]):
        r"""Fit to dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.


        Returns:
            self.
        """
        return self.fit(
            y=[item[self._n_y] for item in dataset],
            X=[item[self._n_X] for item in dataset] if self._n_X is not None else None,
            atomic_number=[
                item[self._n_atomic_number] for item in dataset] if self._n_atomic_number is not None else None,
            sample_weight=[
                item[self._n_sample_weight] for item in dataset] if self._n_sample_weight is not None else None
        )

    # noinspection PyPep8Naming
    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          copy: bool = True,
                          copy_dataset: bool = False,
                          ) -> List[Dict[str, np.ndarray]]:
        r"""Transform dataset with relevant `X` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.
            copy (bool): Whether to copy data for transformation. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Transformed dataset.
        """
        if copy_dataset:
            dataset = dataset.copy()
        out = self.transform(
            y=[graph[self._n_y] for graph in dataset],
            X=[item[self._n_X] for item in dataset] if self._n_X is not None else None,
            atomic_number=[
                item[self._n_atomic_number] for item in dataset] if self._n_atomic_number is not None else None,
            copy=copy,
        )
        for graph, out_value in zip(dataset, out):
            graph[self._n_y] = out_value
        return dataset

    # noinspection PyPep8Naming
    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  copy: bool = True,
                                  copy_dataset: bool = False,
                                  ) -> List[Dict[str, np.ndarray]]:
        r"""Inverse transform dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.
            copy (bool): Whether to copy data for transformation. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Inverse-transformed dataset.
        """
        if copy_dataset:
            dataset = dataset.copy()
        out = self.inverse_transform(
            y=[graph[self._n_y] for graph in dataset],
            X=[item[self._n_X] for item in dataset] if self._n_X is not None else None,
            atomic_number=[
                item[self._n_atomic_number] for item in dataset] if self._n_atomic_number is not None else None,
            copy=copy,
        )
        for graph, out_value in zip(dataset, out):
            graph[self._n_y] = out_value
        return dataset

    # noinspection PyPep8Naming
    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                              copy: bool = True,
                              copy_dataset: bool = False
                              ) -> List[Dict[str, np.ndarray]]:
        r"""Fit and transform to dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` with dictionary of numpy arrays.
            copy (bool): Whether to copy data for transformation. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Transformed dataset.
        """
        self.fit_dataset(dataset=dataset)
        return self.transform_dataset(
            dataset=dataset, copy=copy, copy_dataset=copy_dataset)
