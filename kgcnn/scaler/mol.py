from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np

from kgcnn.scaler.scaler import StandardScaler


class ExtensiveMolecularScaler:
    """Scaler for extensive properties like energy to remove a simple linear behaviour with additive atom
    contributions. Interface is designed after scikit-learn standard scaler. Internally Ridge regression ist used.
    Only the atomic number is used as extensive scaler. This could be further improved by also taking bonds and
    interactions into account, e.g. as energy contribution.

    """
    _attributes_list_sklearn = ["coef_", "intercept_", "n_iter_", "n_features_in_", "feature_names_in_"]
    _attributes_list_mol = ["scale_", "_fit_atom_selection", "_fit_atom_selection_mask"]
    max_atomic_number = 95

    def __init__(self, alpha: float = 1e-9, fit_intercept: bool = False, **kwargs):
        r"""Initialize scaler with parameters directly passed to scikit-learns :obj:`Ridge()`.

        Args:
            alpha (float): Regularization parameter for regression.
            fit_intercept (bool): Whether to allow a constant offset per target.
            kwargs: Additional arguments passed to :obj:`Ridge()`.
        """

        self.ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)

        self._fit_atom_selection_mask = None
        self._fit_atom_selection = None
        self.scale_ = None

    def fit(self, atomic_number, molecular_property, sample_weight=None):
        r"""Fit atomic number to the molecular properties.

        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, #atoms)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            self
        """
        if len(atomic_number) != len(molecular_property):
            raise ValueError(
                "`ExtensiveMolecularScaler` different input shape {0} vs. {1}".format(
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
        self.scale_ = np.std(diff, axis=0, keepdims=True)
        return self

    def predict(self, atomic_number):
        """Predict the offset form atomic numbers. Requires :obj:`fit()` called previously.

        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.

        Returns:
            np.ndarray: Offset of atomic properties fitted previously. Shape is `(n_samples, n_properties)`.
        """
        if self._fit_atom_selection_mask is None:
            raise ValueError("ERROR: `ExtensiveMolecularScaler` has not been fitted yet. Can not predict.")
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

    def _plot_predict(self, atomic_number, molecular_property):
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

    def transform(self, atomic_number, molecular_property):
        """Transform any atomic number list with matching properties based on previous fit. Also std-scaled.

        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.

        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        return (molecular_property - self.predict(atomic_number)) / self.scale_

    def fit_transform(self, atomic_number, molecular_property, sample_weight=None):
        """Combine fit and transform methods in one call.

        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.

        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        self.fit(atomic_number, molecular_property, sample_weight)
        return self.transform(atomic_number, molecular_property)

    def inverse_transform(self, atomic_number, molecular_property):
        """Reverse the transform method to original properties without offset and scaled to original units.

        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`

        Returns:
            np.ndarray: Original atomic properties. Shape is `(n_samples, n_properties)`.
        """
        return molecular_property * self.scale_ + self.predict(atomic_number)

    def get_config(self):
        """Get configuration for scaler."""
        return self.ridge.get_params()

    def get_weights(self) -> dict:
        weights = dict()
        for x in self._attributes_list_mol:
            weights.update({x: np.array(getattr(self, x))})
        for x in self._attributes_list_sklearn:
            if hasattr(self.ridge, x):
                weights.update({x: np.array(getattr(self.ridge, x))})
        return weights


class QMGraphLabelScaler:
    """A scaler that scales QM targets differently. For now, the main difference is that intensive and extensive
    properties are scaled differently. In principle, also dipole, polarizability or rotational constants
    could to be standardized differently.

    """

    def __init__(self, scaler: list):
        if not isinstance(scaler, list):
            raise TypeError("Scaler information for `QMGraphLabelScaler` must be list, got %s." % scaler)

        self.scaler_list = []
        for x in scaler:
            # If x is already a scaler, add it directly to the scaler list.
            if hasattr(x, "fit") and hasattr(x, "transform") and hasattr(x, "inverse_transform"):
                self.scaler_list.append(x)
                continue
            # Otherwise, must be serialized version of a scaler.
            if not isinstance(x, dict):
                raise TypeError("Single scaler for `QMGraphLabelScaler` must be dict, got %s." % x)

            if "class_name" not in x:
                raise ValueError("Scaler class for single target must be defined, got %s" % x)

            if x["class_name"] == "StandardScaler":
                self.scaler_list.append(StandardScaler(**x["config"]))
            elif x["class_name"] == "ExtensiveMolecularScaler":
                self.scaler_list.append(ExtensiveMolecularScaler(**x["config"]))
            else:
                raise ValueError("Unsupported scaler %s" % x["name"])

        self.scale_ = None

    def _input_for_each_scaler_type(self, scaler, graph_labels, node_number):
        if isinstance(scaler, StandardScaler):
            return [graph_labels]
        elif isinstance(scaler, ExtensiveMolecularScaler):
            return node_number, graph_labels
        raise TypeError("Unsupported scaler %s" % scaler)

    def _scale_for_each_scaler_type(self, scaler):
        if isinstance(scaler, StandardScaler):
            return scaler.scale_
        elif isinstance(scaler, ExtensiveMolecularScaler):
            return scaler.scale_[0]
        raise TypeError("Unsupported scaler %s" % scaler)

    def fit_transform(self, graph_labels, node_number):
        r"""Fit and transform all target labels for QM9.

        Args:
            graph_labels (np.ndarray): Array of QM9 labels of shape `(N, 15)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, 15)`.
        """
        self.fit(graph_labels, node_number)
        return self.transform(graph_labels, node_number)

    def transform(self, graph_labels, node_number):
        r"""Transform all target labels for QM. Requires :obj:`fit()` called previously.

        Args:
            graph_labels (np.ndarray): Array of QM unscaled labels of shape `(N, #labels)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, #labels)`.
        """
        self._check_input(node_number, graph_labels)

        out_labels = []
        for i, x in enumerate(self.scaler_list):
            labels = graph_labels[:, i:i+1]
            out_labels.append(x.transform(*self._input_for_each_scaler_type(x, labels, node_number)))

        out_labels = np.concatenate(out_labels, axis=-1)
        return out_labels

    def fit(self, graph_labels, node_number):
        r"""Fit scaling of QM9 graph labels or targets.

        Args:
            graph_labels (np.ndarray): Array of QM labels of shape `(N, #labels)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            self
        """
        self._check_input(node_number, graph_labels)

        for i, x in enumerate(self.scaler_list):
            labels = graph_labels[:, i:i + 1]
            x.fit(*self._input_for_each_scaler_type(x, labels, node_number))

        self.scale_ = np.concatenate([self._scale_for_each_scaler_type(x) for x in self.scaler_list], axis=0)
        return self

    def inverse_transform(self, graph_labels, node_number):
        r"""Back-transform all target labels for QM9.

        Args:
            graph_labels (np.ndarray): Array of QM scaled labels of shape `(N, #labels)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Back-transformed labels of shape `(N, 15)`.
        """
        self._check_input(node_number, graph_labels)

        out_labels = []
        for i, x in enumerate(self.scaler_list):
            labels = graph_labels[:, i:i + 1]
            out_labels.append(x.inverse_transform(*self._input_for_each_scaler_type(x, labels, node_number)))

        out_labels = np.concatenate(out_labels, axis=-1)
        return out_labels

    def _check_input(self, node_number, graph_labels):
        assert len(node_number) == len(graph_labels), "`QMGraphLabelScaler` input length does not match."
        assert graph_labels.shape[-1] == len(self.scaler_list), "`QMGraphLabelScaler` got wrong number of labels."
