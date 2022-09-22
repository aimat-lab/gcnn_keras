import numpy as np
from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class ClinToxDataset(MoleculeNetDataset2018):
    """Store and process full ClinTox dataset."""

    def __init__(self, reload=False, verbose=1):
        """Initialize ClinTox dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(ClinToxDataset, self).__init__("ClinTox", reload=reload, verbose=verbose)

    def read_in_memory(self, **kwargs):
        super(ClinToxDataset, self).read_in_memory(**kwargs)

        # Strictly one class we can use second class here for labeling.
        graph_labels = self.obtain_property("graph_labels")
        graph_labels = [np.array(x[1:], dtype="float") if x is not None else np.array([None]) for x in graph_labels]
        self.assign_property("graph_labels", graph_labels)


# data = ClinToxDataset(reload=False)