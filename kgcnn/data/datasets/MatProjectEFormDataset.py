from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectEFormDataset(MatBenchDataset2020):
    r"""Store and process :obj:`MatProjectEFormDataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database.  Name within Matbench: 'matbench_mp_e_form'.

    Matbench test dataset for predicting DFT formation energy from structure.
    Adapted from Materials Project database. Removed entries having formation energy more than 2.5eV and those
    containing noble gases. Retrieved April 2, 2019. For benchmarking w/ nested cross validation,
    the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication
    for more details.

        * Number of samples: 132752.
        * Task type: regression.
        * Input type: structure.

    """

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize 'matbench_mp_e_form' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        # Use default base class init()
        super(MatProjectEFormDataset, self).__init__("matbench_mp_e_form", reload=reload, verbose=verbose)
        self.label_names = "e_form"
        self.label_units = "eV/atom"
