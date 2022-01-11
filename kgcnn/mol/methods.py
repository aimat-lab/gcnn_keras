import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

global_proton_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
                      'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                      'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                      'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
                      'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47,
                      'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
                      'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
                      'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
                      'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
                      'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
                      'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                      'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
                      'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
                      'Og': 118, 'Uue': 119}
inverse_global_proton_dict = {value: key for key, value in global_proton_dict.items()}


def get_connectivity_from_inverse_distance_matrix(inv_dist_mat, protons, radii_dict=None, k1=16.0, k2=4.0 / 3.0,
                                                  cutoff=0.85, force_bonds=True):
    r"""Get connectivity table from inverse distance matrix defined at last dimensions `(..., N, N)` and
    corresponding bond-radii. Keeps shape with `(..., N, N)`.
    Covalent radii, from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197. 
    Values for metals decreased by 10% according to Robert Paton's Sterimol implementation. 
    Partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code.
    Vectorized version of the original code for numpy arrays that take atomic numbers as input.
    
    Args:
        inv_dist_mat (np.ndarray): Inverse distance matrix defined at last dimensions `(..., N, N)`
            distances must be in Angstrom not in Bohr.
        protons (np.ndarray): An array of atomic numbers matching the inv_dist_mat `(..., N)`,
            for which the radii are to be computed.
        radii_dict (np.ndarray): Covalent radii for each element. If ``None``, stored values are used.
            Otherwise expected numpy array with covalent bonding radii.
            Example: ``np.array([0, 0.34, 0.46, 1.2, ...])`` for atomic number ``np.array([0, 1, 2, ...])``
            that would match ``[None, 'H', 'He', 'Li', ...]``.
        k1 (float): K1-value. Defaults to 16
        k2 (float): K2-value. Defaults to 4.0/3.0
        cutoff (float): Cutoff value to set values to Zero (no bond). Defaults to 0.85.
        force_bonds (bool): Whether to force at least one bond in the bond table per atom. Default is True.
        
    Returns:
        np.ndarray: Connectivity table with 1 for chemical bond and zero otherwise of shape `(..., N, N)`.
    """
    # Dictionary of bond radii
    proton_radii_dict = np.array(
        [0, 0.34, 0.46, 1.2, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.4, 1.25, 1.13, 1.04, 1.1, 1.02, 0.99, 0.96,
         1.76, 1.54, 1.33, 1.22, 1.21, 1.1, 1.07, 1.04, 1.0, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.1, 1.14, 1.17, 1.89,
         1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.19, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09,
         1.76, 1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.5, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37, 1.31,
         1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.3, 1.3, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52, 1.53,
         1.54, 1.55])
    if radii_dict is None:
        radii_dict = proton_radii_dict  # index matches atom number
    # Get Radii
    protons = np.array(protons, dtype="int")
    radii = radii_dict[protons]
    # Calculate
    shape_rad = radii.shape
    r1 = np.expand_dims(radii, axis=len(shape_rad) - 1)
    r2 = np.expand_dims(radii, axis=len(shape_rad))
    r_mat = r1 + r2
    r_mat = k2 * r_mat
    rr = r_mat * inv_dist_mat
    damp = (1.0 + np.exp(-k1 * (rr - 1.0)))
    damp = 1.0 / damp
    if force_bonds:  # Have at least one bond
        max_vals = np.expand_dims(np.argmax(damp, axis=-1), axis=-1)
        np.put_along_axis(damp, max_vals, 1, axis=-1)
        # To make it symmetric transpose last two axis
        damp = np.swapaxes(damp, -2, -1)
        np.put_along_axis(damp, max_vals, 1, axis=-1)
        damp = np.swapaxes(damp, -2, -1)
    damp[damp < cutoff] = 0
    bond_tab = np.round(damp)
    return bond_tab


class ExtensiveMolecularScaler:
    """Scaler for extensive properties like energy to remove a simple linear behaviour with additive atom
    contributions. Interface is designed after scikit-learn standard scaler. Internally Ridge regression ist used.
    Only the atomic number is used as extensive scaler. This could be further improved by also taking bonds and
    interactions into account, e.g. as energy contribution.

    """

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
        self._fit_coef = None
        self._fit_intercept = None
        self.scale_ = None

    def fit(self, atomic_number, molecular_property, sample_weight=None):
        """Fit atomic number to the molecular properties.

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
        self._fit_coef = self.ridge.coef_
        self._fit_intercept = self.ridge.intercept_
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
            raise ValueError("ERROR:kgcnn: `ExtensiveMolecularScaler` has not been fitted yet. Can not predict.")
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
