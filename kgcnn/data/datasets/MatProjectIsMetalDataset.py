from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectIsMetalDataset(MatBenchDataset2020):
    r"""Store and process :obj:`MatProjectEFormDataset` from `MatBench <https://matbench.materialsproject.org/>`_
    database.

    """

    def __init__(self, reload=False, verbose=1):
        r"""Initialize 'matbench_mp_e_form' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        super(MatProjectIsMetalDataset, self).__init__("matbench_mp_is_metal", reload=reload, verbose=verbose)
        self.label_names = "is_metal"
        self.label_units = ""
