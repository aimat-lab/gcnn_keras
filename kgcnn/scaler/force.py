import numpy as np
import logging
from typing import Union, List, Dict, Tuple
from kgcnn.scaler.mol import ExtensiveMolecularScalerBase

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class EnergyForceExtensiveLabelScaler(ExtensiveMolecularScalerBase):
    r"""Extensive scaler for scaling jointly energy, forces.

    Inherits from :obj:`kgcnn.scaler.mol.ExtensiveMolecularScalerBase` but makes use of `X` , `y` , as
    `atomic_number` and (`energy` , `force` ).
    In contrast to :obj:`kgcnn.scaler.mol.ExtensiveMolecularLabelScaler` which uses only
    `y` as `energy` .

    Interface is designed after scikit-learn scaler and has additional functions to apply on datasets with
    :obj:`fit_dataset()` and :obj:`transform_dataset()`

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
        r"""Initialize layer with arguments for :obj:`kgcnn.scaler.mol.ExtensiveMolecularScalerBase` .

        Args:
            standardize_coordinates (bool): Whether to standardize coordinates. Must always be False.
            kwargs: Kwargs for :obj:`kgcnn.scaler.mol.ExtensiveMolecularScalerBase` .
        """
        super(EnergyForceExtensiveLabelScaler, self).__init__(**kwargs)
        self._standardize_coordinates = standardize_coordinates
        if self._standardize_coordinates:
            raise NotImplementedError("Scaling of coordinates is not supported. This class is a pure label scaler.")
        # Backward compatibility.
        self._use_separate_input_arguments = False

    def fit(self, y: Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]] = None, *,
            X: List[np.ndarray] = None,
            sample_weight: Union[None, np.ndarray] = None,
            force: Union[None, List[np.ndarray]] = None,
            atomic_number: Union[None, List[np.ndarray]] = None
            ) -> Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]]:
        """Fit Scaler to data.

        Args:
            y (tuple): Tuple of `(energy, forces)` .
                Energies must be a single array or list of energies of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as with each force stored in a
                numpy array. Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            X (list): Atomic number `atomic_number` are a list of arrays of atomic numbers.
                Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            sample_weight (list, np.ndarray): Weights for each sample.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .

        Returns:
            self.
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        return super(EnergyForceExtensiveLabelScaler, self).fit(
            molecular_property=y, sample_weight=sample_weight, atomic_number=atomic_number)

    def fit_transform(self, y: Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]] = None, *,
            X: List[np.ndarray] = None,
            sample_weight: Union[None, np.ndarray] = None,
            force: Union[None, List[np.ndarray]] = None,
            atomic_number: Union[None, List[np.ndarray]] = None,
            copy: bool = True
            ) -> Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]]:
        """Fit Scaler to data and subsequently transform data.

        Args:
            y (tuple): Tuple of `(energy, forces)` .
                Energies must be a single array or list of energies of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as with each force stored in a
                numpy array. Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            X (list): Atomic number `atomic_number` are a list of arrays of atomic numbers.
                Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            sample_weight (list, np.ndarray): Weights for each sample.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .
            copy (bool): Not yet implemented.

        Returns:
            tuple: Tuple of transformed `(energy, forces)` .
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        self.fit(X=X, y=y, atomic_number=atomic_number, force=force, sample_weight=sample_weight)
        return self.transform(X=X, y=y, copy=copy, force=force, atomic_number=atomic_number)

    def transform(self, y: Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]] = None, *,
                  X: List[np.ndarray] = None,
                  force: Union[None, List[np.ndarray]] = None,
                  atomic_number: Union[None, List[np.ndarray]] = None,
                  copy: bool = True
                  ) -> Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]]:
        """Perform scaling of atomic energies and forces.

        Args:
            y (tuple): Tuple of `(energy, forces)` .
                Energies must be a single array or list of energies of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as with each force stored in a
                numpy array. Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            X (list): Atomic number `atomic_number` are a list of arrays of atomic numbers.
                Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .
            copy (bool): Not yet implemented.

        Returns:
            tuple: Tuple of transformed `(energy, forces)` .
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        if copy:
            y = np.array(y)
            force = [np.array(f) for f in force]
        for i in range(len(y)):
            y[i][:] = y[i] - self.predict(atomic_number)[i]
            if self._standardize_scale:
                y[i][:] = y[i] / self.scale_
                force[i][:] = force[i] / np.expand_dims(self.scale_, axis=0)
        return y, force

    def inverse_transform(self, y: Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]] = None, *,
                          X: List[np.ndarray] = None,
                          force: Union[None, List[np.ndarray]] = None,
                          atomic_number: Union[None, List[np.ndarray]] = None,
                          copy: bool = True
                          ) -> Union[Tuple, List][Union[List[np.ndarray], np.ndarray], List[np.ndarray]]:
        """Scale back data for atoms.

        Args:
            y (tuple): Tuple of `(energy, forces)` .
                Energies must be a single array or list of energies of shape `(n_samples, n_states)` .
                For one energy this must still be `(n_samples, 1)` . List of forces as with each force stored in a
                numpy array. Note that you can also pass the forces separately to function argument `force` , in
                which case `y` should be only energies (not a tuple).
            X (list): Atomic number `atomic_number` are a list of arrays of atomic numbers.
                Example: `[np.array([7,1,1,1]), ...]` . They must match in length.
                Note that you can also pass the atomic numbers separately to function argument `atomic_number` , in
                which case `X` is ignored.
            force (list): List of forces as numpy arrays. Deprecated, since they can be contained in `y` .
            atomic_number (list): List of arrays of atomic numbers. Example [np.array([7,1,1,1]), ...].
                Deprecated, since they can be contained in `X` .
            copy (bool): Not yet implemented.

        Returns:
            tuple: Tuple of reverse-transformed `(energy, forces)` .
        """
        X, y, force, atomic_number = self._verify_input(X, y, force, atomic_number)
        if copy:
            y = np.array(y)
            force = [np.array(f) for f in force]
        for i in range(len(y)):
            if self._standardize_scale:
                y[i][:] = y[i][:] * self.scale_
                force[i][:] = force[i] * np.expand_dims(self.scale_, axis=0)
            y[i][:] = y[i][:] + self.predict(atomic_number)
        return y, force

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
        else:
            self._use_separate_input_arguments = False
            energy, forces = y
        if len(energy) != len(forces):
            raise ValueError("Length of energy '%s' do not match force '%s'." % (len(energy), len(forces)))
        if atomic_number is not None:
            atoms = atomic_number
            x_input = X
        else:
            atoms = X
            x_input = None
        return x_input, energy, forces, atoms

    def get_config(self) -> dict:
        """Get configuration for scaler."""
        config = super(EnergyForceExtensiveLabelScaler, self).get_config()
        config.update({"standardize_coordinates": self._standardize_coordinates})
        return config

    def set_config(self, config: dict):
        """Set configuration for scaler.

        Args:
            config (dict): Config dictionary.
        """
        self._standardize_coordinates = config["standardize_coordinates"]
        config_super = {key: value for key, value in config.items() if key not in ["standardize_coordinates"]}
        return super(EnergyForceExtensiveLabelScaler, self).set_config(config_super)

    # Similar functions that work on dataset plus property names.
    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]], y: List[str] = None, X: str = None,
                    sample_weight: str = None, **fit_params):
        """

        Args:
            dataset (list):
            y (str):
            X (str):
            sample_weight (str):
            **fit_params:

        Returns:

        """
        atoms = X
        energy, force = y
        return self.fit(
            X=[item[atoms] for item in dataset],
            y=([item[energy] for item in dataset], [item[force] for item in dataset]),
            sample_weight=[item[sample_weight] for item in dataset] if sample_weight is not None else None,
            **fit_params
        )

    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          X: str = None, y: List[str] = None, copy: bool = True):
        atoms = X
        energy, force = y
        if copy:
            dataset = dataset.copy()
        self.transform(
            X=[item[atoms] for item in dataset],
            y=([item[energy] for item in dataset], [item[force] for item in dataset]),
            copy=False,
        )
        return dataset

    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  X: str = None, y: List[str] = None, copy: bool = True):
        atoms = X
        energy, force = y
        if copy:
            dataset = dataset.copy()
        self.inverse_transform(
            X=[item[atoms] for item in dataset],
            y=([item[energy] for item in dataset], [item[force] for item in dataset]),
            copy=False,
        )
        return dataset

    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                       X: str = None, y: List[str] = None,
                       sample_weight: str = None, copy: bool = True):
        self.fit_dataset(dataset=dataset, X=X, y=y, sample_weight=sample_weight)
        return self.transform_dataset(dataset=dataset, X=X, y=y, copy=copy)
