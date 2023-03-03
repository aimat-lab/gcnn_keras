from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class LipopDataset(MoleculeNetDataset2018):
    r"""Store and process 'Lipop' dataset from `MoleculeNet <https://moleculenet.org/>`_ database.
    Class inherits from :obj:`MoleculeNetDataset2018` and downloads dataset on class initialization.
    Compare reference:
    `DeepChem <https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#lipo-datasets>`__ reading:
    Experimental results of octanol/water distribution coefficient(logD at pH 7.4).
    Description in DeepChem reads: 'Lipophilicity is an important feature of drug molecules that affects both
    membrane permeability and solubility. The lipophilicity dataset, curated from ChEMBL database, provides
    experimental results of octanol/water distribution coefficient (logD at pH 7.4) of 4200 compounds.'
    Random or Scaffold splitting is recommended for this dataset.

    References:

        (1) Hersey, A. ChEMBL Deposited Data Set - AZ dataset; 2015. `<https://doi.org/10.6019/chembl3301361>`_

    """

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize Lipop dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        super(LipopDataset, self).__init__("Lipop", reload=reload, verbose=verbose)
