import numpy as np
from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class SIDERDataset(MoleculeNetDataset2018):
    """Store and process full SIDER dataset."""

    def __init__(self, reload=False, verbose=1):
        """Initialize SIDER dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(SIDERDataset, self).__init__("SIDER", reload=reload, verbose=verbose)

    def read_in_memory(self, **kwargs):
        super(SIDERDataset, self).read_in_memory(**kwargs)
        graph_labels = self.obtain_property("graph_labels")
        graph_labels = [np.array(x, dtype="float") for x in graph_labels]
        self.assign_property("graph_labels", graph_labels)


# data = SIDERDataset(reload=False)
