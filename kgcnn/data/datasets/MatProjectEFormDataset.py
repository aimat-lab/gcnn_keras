from kgcnn.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectEFormDataset(MatBenchDataset2020):
    """Store and process 'matbench_mp_e_form' dataset."""

    def __init__(self, reload=False, verbose=1):
        r"""Initialize 'matbench_mp_e_form' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        super(MatProjectEFormDataset, self).__init__("matbench_mp_e_form", reload=reload, verbose=verbose)
        self.label_names = "e_form"
        self.label_units = "eV/atom"


# data = MatProjectEFormDataset()
# data.map_list(method="set_range_periodic", max_distance=4, max_neighbours=65)
