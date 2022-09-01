from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class FreeSolvDataset(MoleculeNetDataset2018):
    """Store and process full LipopDataset dataset."""

    def __init__(self, reload=False, verbose=1):
        r"""Initialize Lipop dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(FreeSolvDataset, self).__init__("FreeSolv", reload=reload, verbose=verbose)


# data = FreeSolvDataset(reload=True)

