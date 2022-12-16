from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class FreeSolvDataset(MoleculeNetDataset2018):
    r"""Store and process 'FreeSolv' dataset from `MoleculeNet <https://moleculenet.org/>`_ database.
    Class inherits from :obj:`MoleculeNetDataset2018` and downloads dataset on class initialization.
    Compare reference: `DeepChem <https://deepchem.readthedocs.io/en/latest/api_reference/
    moleculenet.html#freesolv-dataset>`_  reading:
    Experimental and calculated hydration free energy of small molecules in water.
    Description in DeepChem reads: 'The FreeSolv dataset is a collection of experimental and calculated hydration
    free energies for small molecules in water, along with their experiemental values. Here, we are using a modified
    version of the dataset with the molecule smile string and the corresponding experimental hydration free energies.'

    Random splitting is recommended for this dataset.

    References:

        (1) Lukasz Maziarka, et al. Molecule Attention Transformer.
            NeurIPS 2019 arXiv:2002.08264v1 [cs.LG].
        (2) Mobley DL, Guthrie JP. FreeSolv: a database of experimental and calculated hydration free energies,
            with input files. J Comput Aided Mol Des. 2014;28(7):711-720. doi:10.1007/s10822-014-9747-x

    """

    def __init__(self, reload=False, verbose=1):
        r"""Initialize Lipop dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(FreeSolvDataset, self).__init__("FreeSolv", reload=reload, verbose=verbose)
