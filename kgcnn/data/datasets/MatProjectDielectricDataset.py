from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectDielectricDataset(MatBenchDataset2020):
    """Store and process :obj:`MatProjectDielectricDataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database. Name within Matbench: 'matbench_dielectric'.

    Matbench test dataset for predicting refractive index from structure. Adapted from Materials Project database.
    Removed entries having a formation energy (or energy above the convex hull) more than 150meV and
    those having refractive indices less than 1 and those containing noble gases. Retrieved April 2, 2019.
    For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data;
    refer to the Automatminer/Matbench publication for more details.

        * Number of samples: 4764
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
        super(MatProjectDielectricDataset, self).__init__("matbench_dielectric", reload=reload, verbose=verbose)
        self.label_names = "n_r"
        self.label_units = ""
