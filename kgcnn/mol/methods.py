import numpy as np


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
    proton_raddi_dict = np.array(
        [0, 0.34, 0.46, 1.2, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.4, 1.25, 1.13, 1.04, 1.1, 1.02, 0.99, 0.96,
         1.76, 1.54, 1.33, 1.22, 1.21, 1.1, 1.07, 1.04, 1.0, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.1, 1.14, 1.17, 1.89,
         1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.19, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09,
         1.76, 1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.5, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37, 1.31,
         1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.3, 1.3, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52, 1.53,
         1.54, 1.55])
    if radii_dict is None:
        radii_dict = proton_raddi_dict  # index matches atom number
    # Get Radii
    protons = np.array(protons, dtype=np.int)
    radii = radii_dict[protons]
    # Calculate
    shape_rad = radii.shape
    r1 = np.expand_dims(radii, axis=len(shape_rad) - 1)
    r2 = np.expand_dims(radii, axis=len(shape_rad))
    rmat = r1 + r2
    rmat = k2 * rmat
    rr = rmat * inv_dist_mat
    damp = (1.0 + np.exp(-k1 * (rr - 1.0)))
    damp = 1.0 / damp
    if force_bonds:  # Have at least one bond
        maxvals = np.expand_dims(np.argmax(damp, axis=-1), axis=-1)
        np.put_along_axis(damp, maxvals, 1, axis=-1)
        # To make it symmetric transpose last two axis
        damp = np.swapaxes(damp, -2, -1)
        np.put_along_axis(damp, maxvals, 1, axis=-1)
        damp = np.swapaxes(damp, -2, -1)
    damp[damp < cutoff] = 0
    bond_tab = np.round(damp)
    return bond_tab


