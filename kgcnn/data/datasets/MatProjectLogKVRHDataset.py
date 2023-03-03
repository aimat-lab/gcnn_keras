from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectLogKVRHDataset(MatBenchDataset2020):
    """Store and process :obj:`MatProjectLogKVRHDataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database. Name within Matbench: 'matbench_log_kvrh'.

    Matbench v0.1 test dataset for predicting DFT log10 VRH-average bulk modulus from structure.
    Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull)
    more than 150meV and those having negative G_Voigt, G_Reuss, G_VRH, K_Voigt, K_Reuss, or K_VRH and those failing
    G_Reuss <= G_VRH <= G_Voigt or K_Reuss <= K_VRH <= K_Voigt and those containing noble gases.
    Retrieved April 2, 2019. For benchmarking w/ nested cross validation, the order of the dataset must be identical
    to the retrieved data; refer to the Automatminer/Matbench publication for more details.

        * Number of samples: 10987
        * Task type: regression
        * Input type: structure

    """

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize 'matbench_mp_e_form' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        # Use default base class init()
        super(MatProjectLogKVRHDataset, self).__init__("matbench_log_kvrh", reload=reload, verbose=verbose)
        self.label_names = "log10(K_VRH) "
        self.label_units = "GPa"
