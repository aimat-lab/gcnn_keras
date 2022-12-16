from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class ESOLDataset(MoleculeNetDataset2018):
    r"""Store and process 'ESOL' dataset from `MoleculeNet <https://moleculenet.org/>`_ database.
    Class inherits from :obj:`MoleculeNetDataset2018` and downloads dataset on class initialization.

    Compare reference: `DeepChem <https://deepchem.readthedocs.io/en/latest/api_reference/
    moleculenet.html#delaney-datasets>`_  reading:

    Water solubility data(log solubility in mols per litre) for common organic small molecules.
    Random or Scaffold splitting is recommended for this dataset.
    Description in DeepChem reads: 'The Delaney (ESOL) dataset a regression dataset containing structures and water
    solubility data for 1128 compounds. The dataset is widely used to validate machine learning models on
    estimating solubility directly from molecular structures (as encoded in SMILES strings).'

    References:

        (1) Delaney, John S. ESOL: estimating aqueous solubility directly from molecular structure.
            Journal of chemical information and computer sciences 44.3 (2004): 1000-1005.

    """

    def __init__(self, reload=False, verbose=1):
        """Initialize ESOL dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(ESOLDataset, self).__init__("ESOL", reload=reload, verbose=verbose)
