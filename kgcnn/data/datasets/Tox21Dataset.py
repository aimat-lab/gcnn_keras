import numpy as np

from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class Tox21Dataset(MoleculeNetDataset2018):
    """Store and process full Tox21 dataset."""

    def __init__(self, reload=False, verbose=1):
        """Initialize Tox21 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(Tox21Dataset, self).__init__("Tox21", reload=reload, verbose=verbose)

    def read_in_memory(self, **kwargs):
        super(Tox21Dataset, self).read_in_memory(**kwargs)
        graph_labels = self.obtain_property("graph_labels")
        graph_labels = np.nan_to_num(graph_labels)
        self.assign_property("graph_labels", [x for x in graph_labels])


# data = Tox21Dataset(reload=False)

