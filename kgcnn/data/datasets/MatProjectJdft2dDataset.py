from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectJdft2dDataset(MatBenchDataset2020):
    """Store and process :obj:`MatProjectJdft2dDataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database. Name within Matbench: 'matbench_jdft2d'.

    Matbench test dataset for predicting exfoliation energies from crystal structure
    (computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT database.
    For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data;
    refer to the Automatminer/Matbench publication for more details.

        * Number of samples: 636
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
        super(MatProjectJdft2dDataset, self).__init__("matbench_jdft2d", reload=reload, verbose=verbose)
        self.label_names = "exfoliation_en "
        self.label_units = "meV/atom"
