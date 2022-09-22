from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectPhononsDataset(MatBenchDataset2020):
    """Store and process 'matbench_mp_e_form' dataset."""

    def __init__(self, reload=False, verbose=1):
        r"""Initialize 'matbench_mp_e_form' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        super(MatProjectPhononsDataset, self).__init__("matbench_phonons", reload=reload, verbose=verbose)
        self.label_names = "omega_max"
        self.label_units = "1/cm"


# data = MatProjectIsMetalDataset()
# data.map_list(method="set_range_periodic", max_distance=4, max_neighbours=65)
