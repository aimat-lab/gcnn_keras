import numpy as np
import logging
from typing import Union, List, Dict
from kgcnn.scaler.mol import ExtensiveMolecularLabelScaler

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class EnergyForceExtensiveLabelScaler(ExtensiveMolecularLabelScaler):
    r"""Extensive scaler for scaling jointly energy, forces and optionally coordinates.

    Inherits from :obj:`kgcnn.scaler.mol.ExtensiveMolecularLabelScaler` but makes use of `X` , `y` , as
    `atomic_number` and (`energy` , `force` ).
    In contrast to :obj:`kgcnn.scaler.mol.ExtensiveMolecularLabelScaler` which uses only
    `y` as for example `energy` .

    Interface is designed after scikit-learn scaler.

    .. note::

        Units for energy and forces must match.

    Code example for scaler:

    .. code-block:: python

        import numpy as np
        from kgcnn.scaler.force import EnergyForceExtensiveLabelScaler
        energy = np.random.rand(5).reshape((5,1))
        mol_num = [np.array([6, 1, 1, 1, 1]), np.array([7, 1, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1]), np.array([6, 6, 1, 1]), np.array([6, 6, 1, 1, 1, 1, 1, 1])
        ]
        force = [np.random.rand(len(m)*3).reshape((len(m),3)) for m in mol_num]
        scaler = EnergyForceExtensiveLabelScaler()
        scaler.fit(X=mol_num, y=[energy, force])
        print(scaler.get_weights())
        print(scaler.get_config())
        scaler._plot_predict(energy, mol_num)  # For debugging.
        y, f = scaler.transform(X=mol_num, y=[energy, force])
        print(energy, y)
        print(scaler.inverse_transform(X=mol_num, y=[y, f])[1][1][0], f[0])
        scaler.save("example.json")
        new_scaler = EnergyForceExtensiveLabelScaler()
        new_scaler.load("example.json")
        print(scaler.inverse_transform(X=mol_num, y=[y, f])[1][1][0], f[0])

    """

    def __init__(self, standardize_coordinates: bool = False, **kwargs):
        super(EnergyForceExtensiveLabelScaler, self).__init__(**kwargs)
        self._standardize_coordinates = standardize_coordinates
        if self._standardize_coordinates:
            raise NotImplementedError("Scaling of coordinates is not supported. This class is pure label scaler.")
        # Backward compatibility.
        self._use_separate_input_arguments = False

    def fit(self, *, X=None, y: Union[tuple, List, np.ndarray] = None, sample_weight: Union[List, np.ndarray] = None,
            force=None, atomic_number=None):
        """Fit Scaler to data.

        Args:
            X (tuple, list, np.ndarray): Atomic number `atomic_number` are a list of
                arrays of atomic numbers. Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            y (tuple, list, np.ndarray): Tuple of `(energy, forces)` .
                Array or list of energy of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as numpy arrays.
                Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            sample_weight (list, np.ndarray): Weights for each sample.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        return super(EnergyForceExtensiveLabelScaler, self).fit(
            X=y, y=None, sample_weight=sample_weight, atomic_number=atomic_number)

    def fit_transform(self, *, X=None, y=None, copy=True, force=None, atomic_number=None,
                      sample_weight: Union[List, np.ndarray] = None):
        """Fit Scaler to data.

        Args:
            X (tuple, list, np.ndarray): Atomic number `atomic_number` are a list of
                arrays of atomic numbers. Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            y (tuple, list, np.ndarray): Tuple of `(energy, forces)` .
                Array or list of energy of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as numpy arrays.
                Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            sample_weight (list, np.ndarray): Weights for each sample.
            copy (bool): Not yet implemented.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .

        Returns:
            list: Scaled y.
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        self.fit(X=X, y=y, atomic_number=atomic_number, force=force, sample_weight=sample_weight)
        return self.transform(X=X, y=y, copy=copy, force=force, atomic_number=atomic_number)

    def transform(self, *, X=None, y=None, copy=True, force=None, atomic_number=None):
        """Perform scaling of atomic energies and forces.

        Args:
            X (tuple, list, np.ndarray): Atomic number `atomic_number` are a list of
                arrays of atomic numbers. Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            y (tuple, list, np.ndarray): Tuple of `(energy, forces)` .
                Array or list of energy of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as numpy arrays.
                Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            copy (bool): Whether to copy array or change inplace.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .

        Returns:
            tuple: Scaled y.
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        if copy:
            y = np.array(y)
            force = [np.array(f) for f in force]
        y -= self.predict(atomic_number)
        if self._standardize_scale:
            y /= np.expand_dims(self.scale_, axis=0)
            for i in range(len(force)):
                force[i][:] = force[i] / np.expand_dims(self.scale_, axis=0)
        return self._verify_output(X, y, force, atomic_number)

    def inverse_transform(self, *, X=None, y=None, copy=True, force=None,
                          atomic_number=None):
        """Scale back data for atoms.

        Args:
            X (tuple, list, np.ndarray): Atomic number `atomic_number` are a list of
                arrays of atomic numbers. Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            y (tuple, list, np.ndarray): Tuple of `(energy, forces)` .
                Array or list of energy of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as numpy arrays.
                Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            copy (bool): Whether to copy array or change inplace.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .

        Returns:
            tuple: Rescaled y.
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        if copy:
            y = np.array(y)
            force = [np.array(f) for f in force]
        if self._standardize_scale:
            y *= np.expand_dims(self.scale_, axis=0)
            for i in range(len(force)):
                force[i][:] = force[i] * np.expand_dims(self.scale_, axis=0)
        y += self.predict(atomic_number)
        return self._verify_output(X, y, force, atomic_number)

    # Needed for backward compatibility.
    def _verify_input(self, X, y, force, atomic_number):
        # Verify the input format.
        if y is None:
            raise ValueError("`EnergyForceExtensiveLabelScaler` requires 'y' argument, but got 'None'.")
        if force is not None:
            self._use_separate_input_arguments = True
            module_logger.warning(
                "Preferred input is `(energy, force)` for 'y', since `force` argument is deprecated.")
            energy, forces = y, force
            if len(energy) != len(forces):
                raise ValueError("Length of energy '%s' do not match force '%s'." % (len(energy), len(forces)))
        else:
            self._use_separate_input_arguments = False
            energy, forces = y
        if atomic_number is not None:
            atoms = atomic_number
            x_input = X
        else:
            atoms = X
            x_input = None
        return x_input, energy, forces, atoms

    # Needed for backward compatibility.
    def _verify_output(self, X, y, force, atomic_number):
        if self._use_separate_input_arguments:
            return X, y, force
        else:
            return y, force

    def get_config(self):
        """Get configuration for scaler."""
        config = super(EnergyForceExtensiveLabelScaler, self).get_config()
        config.update({"standardize_coordinates": self._standardize_coordinates})
        return config

    def set_config(self, config):
        """Set configuration for scaler.

        Args:
            config (dict): Config dictionary.
        """
        self._standardize_coordinates = config["standardize_coordinates"]
        config_super = {key: value for key, value in config.items() if key not in ["standardize_coordinates"]}
        return super(EnergyForceExtensiveLabelScaler, self).set_config(config_super)

    # Similar functions that work on dataset plus property names.

    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]],
                       X: List[str] = None, y: List[str] = None,
                       sample_weight: str = None, **fit_params):
        coord, atoms = X
        energy, force = y
        return self.fit(
            X=[item[atoms] for item in dataset],
            y=([item[energy] for item in dataset], [item[force] for item in dataset]),
            sample_weight=[item[sample_weight] for item in dataset] if sample_weight is not None else None,
            **fit_params
        )

    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          X: List[str] = None, y: List[str] = None, copy: bool = True):
        coord, atoms = X
        energy, force = y
        if copy:
            dataset.copy()
        self.transform(
            X=[item[atoms] for item in dataset],
            y=([item[energy] for item in dataset], [item[force] for item in dataset]),
            copy=False,
        )
        return dataset

    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  X: List[str] = None, y: List[str] = None, copy: bool = True):
        coord, atoms = X
        energy, force = y
        if copy:
            dataset.copy()
        self.inverse_transform(
            X=[item[atoms] for item in dataset],
            y=([item[energy] for item in dataset], [item[force] for item in dataset]),
            copy=False,
        )
        return dataset

    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                       X: List[str] = None, y: List[str] = None,
                       sample_weight: str = None, copy: bool = True):
        self.fit_dataset(dataset=dataset, X=X, y=y, sample_weight=sample_weight)
        return self.transform_dataset(dataset=dataset, X=X, y=y, copy=copy)
