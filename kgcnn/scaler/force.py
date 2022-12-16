import numpy as np
from typing import Union
from kgcnn.scaler.mol import ExtensiveMolecularScaler


class EnergyForceExtensiveScaler(ExtensiveMolecularScaler):
    r"""Extensive scaler for scaling jointly energy, forces and optionally coordinates.

    Inherits from :obj:`kgcnn.scaler.mol.ExtensiveMolecularScaler` but makes use of `X`, `y`, `force`, `atomic_number`
    input parameters, in contrast to :obj:`kgcnn.scaler.mol.ExtensiveMolecularScaler` which uses only
    `X` and `atomic_number`. The coordinates are expected to be the `X` argument and the output, that is an energy, as
    the `y` argument to match the convention of the `Scaler` classes.
    Interface is designed after scikit-learn scaler.

    .. note::

        Units for energy and forces must match.

    Code example for scaler:

    .. code-block:: python

        import numpy as np
        from kgcnn.scaler.force import EnergyForceExtensiveScaler
        energy = np.random.rand(5).reshape((5,1))
        mol_num = [np.array([6, 1, 1, 1, 1]), np.array([7, 1, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1]), np.array([6, 6, 1, 1]), np.array([6, 6, 1, 1, 1, 1, 1, 1])
        ]
        force = [np.random.rand(len(m)*3).reshape((len(m),3)) for m in mol_num]
        coord = [np.random.rand(len(m)*3).reshape((len(m),3)) for m in mol_num]
        scaler = EnergyForceExtensiveScaler()
        scaler.fit(X=coord, y=energy, force=force, atomic_number=mol_num)
        print(scaler.get_weights())
        print(scaler.get_config())
        scaler._plot_predict(energy, mol_num)  # For debugging.
        x, y, f = scaler.transform(X=coord, y=energy, force=force, atomic_number=mol_num)
        print(energy, y)
        print(scaler.inverse_transform(X=x, y=y, force=f, atomic_number=mol_num)[2][0], f[0])
        scaler.save("example.json")
        new_scaler = EnergyForceExtensiveScaler()
        new_scaler.load("example.json")
        print(scaler.inverse_transform(X=x, y=y, force=f, atomic_number=mol_num)[2][0], f[0])

    """

    def __init__(self, standardize_coordinates: bool = False, **kwargs):
        super(EnergyForceExtensiveScaler, self).__init__(**kwargs)
        self._standardize_coordinates = standardize_coordinates
        if self._standardize_coordinates:
            raise NotImplementedError("Scaling of coordinates is not yet supported.")

    def fit(self, *, X=None, y=None, sample_weight=None, force=None, atomic_number=None):
        """Fit Scaler to data.

        Args:
            X (list): List of coordinates as numpy arrays.
            y (np.ndarray): Array of energy of shape `(n_samples, n_states)`.
            sample_weight (np.ndarray): Weights for each sample.
            force (list): List of forces as numpy arrays
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
        """
        return super(EnergyForceExtensiveScaler, self).fit(
            X=y, y=None, sample_weight=sample_weight, atomic_number=atomic_number)

    def fit_transform(self, *, X=None, y=None, copy=True, force=None, atomic_number=None, **fit_params):
        """Fit Scaler to data.

        Args:
            X (list, np.ndarray): List of coordinates as numpy arrays.
            y (np.ndarray): Array of energy of shape `(n_samples, n_states)`.
            copy (bool): Not yet implemented.
            force (list): List of forces as numpy arrays.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
            fit_params: Additional parameters for fit.

        Returns:
            list: Scaled [X, y, force].
        """
        self._verify_input(y, force, atomic_number)
        self.fit(X=X, y=y, atomic_number=atomic_number, force=force, **fit_params)
        return self.transform(X=X, y=y, copy=copy, force=force, atomic_number=atomic_number)

    def transform(self, *, X=None, y=None, copy=True, force=None, atomic_number=None):
        """Perform scaling of atomic energies and forces.

        Args:
            X (list, np.ndarray): List of coordinates as numpy arrays.
            y (np.ndarray): Array of energy of shape `(n_samples, n_states)`.
            copy (bool): Whether to copy array or change inplace.
            force (list): List of forces as numpy arrays.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].

        Returns:
            list: Scaled [X, y, force].
        """
        self._verify_input(y, force, atomic_number)
        if copy:
            y = np.array(y)
            force = [np.array(f) for f in force]
        y -= self.predict(atomic_number)
        if self._standardize_scale:
            y /= np.expand_dims(self.scale_, axis=0)
            for i in range(len(force)):
                force[i][:] = force[i] / np.expand_dims(self.scale_, axis=0)
        return X, y, force

    def inverse_transform(self, *, X=None, y=None, copy=True, force=None,
                          atomic_number=None):
        """Scale back data for atoms.

        Args:
            X (list, np.ndarray): List of coordinates as numpy arrays.
            y (np.ndarray): Array of energy of shape `(n_samples, n_states)`.
            copy (bool): Whether to copy array or change inplace.
            force (list): List of forces as numpy arrays.
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].

        Returns:
            list: Rescaled [X, y, force].
        """
        self._verify_input(y, force, atomic_number)
        if copy:
            y = np.array(y)
            force = [np.array(f) for f in force]
        if self._standardize_scale:
            y *= np.expand_dims(self.scale_, axis=0)
            for i in range(len(force)):
                force[i][:] = force[i] * np.expand_dims(self.scale_, axis=0)
        y += self.predict(atomic_number)
        return X, y, force

    @staticmethod
    def _verify_input(y, force, atomic_number):
        for name, value in zip(["y", "force", "atomic_number"], [y, force, atomic_number]):
            if value is None:
                raise ValueError("`EnergyForceExtensiveScaler` requires '%s' argument, but got 'None'." % name)

    def get_config(self):
        """Get configuration for scaler."""
        config = super(EnergyForceExtensiveScaler, self).get_config()
        config.update({"standardize_coordinates": self._standardize_coordinates})
        return config

    def set_config(self, config):
        """Set configuration for scaler.

        Args:
            config (dict): Config dictionary.
        """
        self._standardize_coordinates = config["standardize_coordinates"]
        config_super = {key: value for key, value in config.items() if key not in ["standardize_coordinates"]}
        return super(EnergyForceExtensiveScaler, self).set_config(config_super)
