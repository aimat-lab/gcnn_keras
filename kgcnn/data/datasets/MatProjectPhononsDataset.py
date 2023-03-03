from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectPhononsDataset(MatBenchDataset2020):
    """Store and process :obj:`MatProjectPhononsDataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database. Name within Matbench: 'matbench_phonons'.

    Matbench test dataset for predicting vibration properties from crystal structure. Original data retrieved
    from Petretto et al. Original calculations done via ABINIT in the harmonic approximation based on density
    functional perturbation theory. Removed entries having a formation energy (or energy above the convex hull)
    more than 150meV. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the
    retrieved data; refer to the Automatminer/Matbench publication for more details.

        * Number of samples: 1265
        * Task type: regression
        * Input type: structure

    last phdos peak: Target variable. Frequency of the highest frequency optical phonon mode peak, in units of 1/cm;
    may be used as an estimation of dominant longitudinal optical phonon frequency.

    """

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize 'matbench_mp_e_form' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        # Use default base class init()
        super(MatProjectPhononsDataset, self).__init__("matbench_phonons", reload=reload, verbose=verbose)
        self.label_names = "omega_max"
        self.label_units = "1/cm"
