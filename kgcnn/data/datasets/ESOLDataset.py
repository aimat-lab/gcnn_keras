from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class ESOLDataset(MoleculeNetDataset2018):
    """Store and process full ESOL dataset."""

    def __init__(self, reload=False, verbose=1):
        """Initialize ESOL dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(ESOLDataset, self).__init__("ESOL", reload=reload, verbose=verbose)


# data = ESOLDataset(reload=True)
# data.set_attributes()
