import numpy as np
import logging
from typing import Union, List, Dict, Tuple
from kgcnn.data.transform.scaler.molecule import ExtensiveMolecularScalerBase

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
        from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
        energy = np.random.rand(5).reshape((5,1))
        mol_num = [np.array([6, 1, 1, 1, 1]), np.array([7, 1, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1]), np.array([6, 6, 1, 1]), np.array([6, 6, 1, 1, 1, 1, 1, 1])
        ]
        force = [np.random.rand(len(m)*3).reshape((len(m),3)) for m in mol_num]
        scaler = EnergyForceExtensiveLabelScaler()
        scaler.fit(y=[energy, force], X=mol_num)
        print(scaler.get_weights())
        print(scaler.get_config())
        scaler._plot_predict(energy, mol_num)  # For debugging.
        y, f = scaler.transform(y=[energy, force], X=mol_num)
        print(energy, y)
        print(scaler.inverse_transform(y=[y, f], X=mol_num)[1][1][0], f[0])
        scaler.save("example.json")
        new_scaler = EnergyForceExtensiveLabelScaler()
        new_scaler.load("example.json")
        print(scaler.inverse_transform(y=[y, f], X=mol_num)[1][1][0], f[0])

    """

    def __init__(self, standardize_coordinates: bool = False, **kwargs):
        r"""Initialize layer with arguments for :obj:`kgcnn.scaler.mol.ExtensiveMolecularScalerBase` .

        Args:
            standardize_coordinates (bool): Whether to standardize coordinates. Must always be False.
            kwargs: Kwargs for :obj:`kgcnn.scaler.mol.ExtensiveMolecularScalerBase` parent class.
                See docs for this class.
        """
        super(EnergyForceExtensiveLabelScaler, self).__init__(**kwargs)
        self._standardize_coordinates = standardize_coordinates
        if self._standardize_coordinates:
            raise NotImplementedError("Scaling of coordinates is not supported. This class is a pure label scaler.")
        # Backward compatibility.
        self._use_separate_input_arguments = False

    # noinspection PyPep8Naming
    def fit(self, y: Tuple[List[np.ndarray], List[np.ndarray]] = None, *,
            X: List[np.ndarray] = None,
            sample_weight: Union[None, np.ndarray] = None,
            force: Union[None, List[np.ndarray]] = None,
            atomic_number: Union[None, List[np.ndarray]] = None
            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    # noinspection PyPep8Naming
    def fit_transform(self, y: Tuple[List[np.ndarray], List[np.ndarray]] = None, *,
                      X: List[np.ndarray] = None,
                      sample_weight: Union[None, np.ndarray] = None,
                      force: Union[None, List[np.ndarray]] = None,
                      atomic_number: Union[None, List[np.ndarray]] = None,
                      copy: bool = True
                      ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    # noinspection PyPep8Naming
    def transform(self, y: Tuple[List[np.ndarray], List[np.ndarray]] = None, *,
                  X: List[np.ndarray] = None,
                  force: Union[None, List[np.ndarray]] = None,
                  atomic_number: Union[None, List[np.ndarray]] = None,
                  copy: bool = True
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
            y = np.array(y) - self.predict(atomic_number)
            if self._standardize_scale:
                y = y / np.expand_dims(self.scale_, axis=0)
                force = [np.array(f) / np.expand_dims(self.scale_, axis=0) for f in force]
            else:
                force = [np.array(f) for f in force]
        else:
            for i in range(len(y)):
                y[i][:] = y[i] - self.predict(atomic_number)[i]
                if self._standardize_scale:
                    y[i][:] = y[i] / self.scale_
                    force[i][:] = force[i] / np.expand_dims(self.scale_, axis=0)
        return y, force

    # noinspection PyPep8Naming
    def inverse_transform(self, y: Tuple[List[np.ndarray], List[np.ndarray]] = None, *,
                          X: List[np.ndarray] = None,
                          force: Union[None, List[np.ndarray]] = None,
                          atomic_number: Union[None, List[np.ndarray]] = None,
                          copy: bool = True
                          ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
            if self._standardize_scale:
                y = y * np.expand_dims(self.scale_, axis=0)
                force = [np.array(f) * np.expand_dims(self.scale_, axis=0) for f in force]
            else:
                force = [np.array(f) for f in force]
            y = y + self.predict(atomic_number)
        else:
            for i in range(len(y)):
                if self._standardize_scale:
                    y[i][:] = y[i][:] * self.scale_
                    force[i][:] = force[i] * np.expand_dims(self.scale_, axis=0)
                y[i][:] = y[i][:] + self.predict(atomic_number)[i]
        return y, force

    # Needed for backward compatibility.
    # noinspection PyPep8Naming
    def _verify_input(self, X, y, force, atomic_number):
        # Verify the input format.
        if y is None:
            raise ValueError("`EnergyForceExtensiveLabelScaler` requires 'y' argument, but got 'None'.")
        if force is not None:
            self._use_separate_input_arguments = True
            if len(force) == len(y):
                energy, forces = y, force
            elif len(y) == 2:
                energy, forces = y[0], force
            else:
                raise ValueError("Energy and forces do not match.")
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
    # noinspection PyPep8Naming
    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]], y: List[str], X: str,
                    atomic_number: str = None, sample_weight: str = None,  **fit_params):
        r"""Fit to dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` containing energies and forces and atomic numbers.
            y (list): List of names for energy and force in dataset. For example ("energy", "forces") .
            X (str): Name of atomic number information in dataset. For example "atomic_number".
            atomic_number (str): Name of atomic number information in dataset. For example "atomic_number".
                Optional, if used, `X` will be ignored.
            sample_weight (str): Name of sample weight information in dataset. For example "sample_weight".
            fit_params: Fit parameters handed to :obj:`fit()`

        Returns:
            self.
        """
        atoms = atomic_number if atomic_number is not None else X
        energy, force = y
        return self.fit(
            X=[item[atoms] for item in dataset],
            y=([item[energy] for item in dataset], [item[force] for item in dataset]),
            sample_weight=[item[sample_weight] for item in dataset] if sample_weight is not None else None,
            **fit_params
        )

    # noinspection PyPep8Naming
    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          y: List[str], X: str, atomic_number: str = None, copy: bool = True,
                          copy_dataset: bool = False,
                          ) -> List[Dict[str, np.ndarray]]:
        r"""Transform dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` containing energies and forces and atomic numbers.
            y (list): List of names for energy and force in dataset. For example ("energy", "forces") .
            X (str): Name of atomic number information in dataset. For example "atomic_number".
            atomic_number (str): Name of atomic number information in dataset. For example "atomic_number".
                Optional, if used, `X` will be ignored.
            copy (bool): Whether to copy data for transformation. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Transformed dataset.
        """
        atoms = atomic_number if atomic_number is not None else X
        energy, force = y
        if copy_dataset:
            dataset = dataset.copy()
        out_energy, out_force = self.transform(
            atomic_number=[graph[atoms] for graph in dataset],
            y=([graph[energy] for graph in dataset], [graph[force] for graph in dataset]),
            copy=copy,
        )
        for graph, graph_energy, graph_force in zip(dataset, out_energy, out_force):
            graph[energy] = graph_energy
            graph[force] = graph_force
        return dataset

    # noinspection PyPep8Naming
    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  y: List[str], X: str, atomic_number: str = None, copy: bool = True,
                                  copy_dataset: bool = False,
                                  ) -> List[Dict[str, np.ndarray]]:
        r"""Inverse transform dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` containing energies and forces and atomic numbers.
            y (list): List of names for energy and force in dataset. For example ("energy", "forces") .
            X (str): Name of atomic number information in dataset. For example "atomic_number".
            atomic_number (str): Name of atomic number information in dataset. For example "atomic_number".
                Optional, if used, `X` will be ignored.
            copy (bool): Whether to copy dataset. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            dataset: Inverse-transformed dataset.
        """
        atoms = atomic_number if atomic_number is not None else X
        energy, force = y
        if copy_dataset:
            dataset = dataset.copy()
        out_energy, out_force = self.inverse_transform(
            atomic_number=[graph[atoms] for graph in dataset],
            y=([graph[energy] for graph in dataset], [graph[force] for graph in dataset]),
            copy=copy,
        )
        for graph, graph_energy, graph_force in zip(dataset, out_energy, out_force):
            graph[energy] = graph_energy
            graph[force] = graph_force
        return dataset

    # noinspection PyPep8Naming
    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                              y: List[str], X: str,
                              atomic_number: str = None, sample_weight: str = None, copy: bool = True,
                              copy_dataset: bool = False,
                              **fit_params) -> List[Dict[str, np.ndarray]]:
        r"""Fit and transform to dataset with relevant `X` , `y` information.

        Args:
            dataset (list): Dataset of type `List[Dict]` containing energies and forces and atomic numbers.
            y (list): List of names for energy and force in dataset. For example ("energy", "forces") .
            X (str): Name of atomic number information in dataset. For example "atomic_number".
            atomic_number (str): Name of atomic number information in dataset. For example "atomic_number".
                Optional, if used, `X` will be ignored.
            sample_weight (str): Name of sample weight information in dataset. For example "sample_weight".
            copy (bool): Whether to copy dataset. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.
            fit_params: Fit parameters handed to :obj:`fit()`

        Returns:
            dataset: Transformed dataset.
        """
        self.fit_dataset(dataset=dataset, X=X, y=y, sample_weight=sample_weight, atomic_number=atomic_number,
                         **fit_params)
        return self.transform_dataset(dataset=dataset, X=X, y=y,atomic_number=atomic_number, copy=copy,
                                      copy_dataset=copy_dataset)
