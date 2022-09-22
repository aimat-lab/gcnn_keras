import numpy as np
from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class Tox21MolNetDataset(MoleculeNetDataset2018):
    """Store and process full Tox21 dataset."""

    def __init__(self, reload=False, verbose=1, remove_nan: bool = False):
        """Initialize Tox21 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        self._remove_nan_label = remove_nan
        super(Tox21MolNetDataset, self).__init__("Tox21", reload=reload, verbose=verbose)

    def read_in_memory(self, **kwargs):
        super(Tox21MolNetDataset, self).read_in_memory(**kwargs)

        # One could set unknown classes to zero, but better go with NaN compatible metrics and loss.
        graph_labels = self.obtain_property("graph_labels")
        graph_labels = [np.array(x, dtype="float") for x in graph_labels]
        if self._remove_nan_label:
            graph_labels = [np.nan_to_num(x) for x in graph_labels]
        self.assign_property("graph_labels", graph_labels)


# data = Tox21MolNetDataset(reload=False)
