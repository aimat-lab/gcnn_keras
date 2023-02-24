import numpy as np
from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class ClinToxDataset(MoleculeNetDataset2018):
    r"""Store and process 'ClinTox' dataset from `MoleculeNet <https://moleculenet.org/>`_ database.
    Class inherits from :obj:`MoleculeNetDataset2018` and downloads dataset on class initialization.

    Compare reference: `DeepChem <https://deepchem.readthedocs.io/en/latest/api_reference/
    moleculenet.html#clintox-datasets>`_ reading:
    Qualitative data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons.
    Random splitting is recommended for this dataset.

    The ClinTox dataset compares drugs approved by the FDA and drugs that have failed clinical trials for toxicity.
    The dataset includes two classification tasks for 1491 drug compounds with known chemical structures:

        - clinical trial toxicity (or absence of toxicity)
        - FDA approval status.

    List of FDA-approved drugs are compiled from the SWEETLEAD database, and list of drugs that failed clinical trials
    for toxicity reasons are compiled from the Aggregate Analysis of ClinicalTrials.gov(AACT) database.

    References:

        (1) Gayvert, Kaitlyn M., Neel S. Madhukar, and Olivier Elemento.
            A data-driven approach to predicting successes and failures of clinica trials.
            Cell chemical biology 23.10 (2016): 1294-1301.
        (2) Artemov, Artem V., et al. Integrated deep learned transcriptomic and
            structure-based predictor of clinical trials outcomes. bioRxiv (2016): 095653.
        (3) Novick, Paul A., et al. SWEETLEAD: an in silico database of approved drugs,
            regulated chemicals, and herbal isolates for computer-aided drug discovery.
            PloS one 8.11 (2013): e79568.
        (4) Aggregate Analysis of ClincalTrials.gov (AACT) Database.
            `<https://www.ctti-clinicaltrials.org/aact-database>`_ .

    """

    def __init__(self, reload=False, verbose: int = 10, label_index: int = 0):
        """Initialize ClinTox dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
            label_index (int): Which information should be taken as binary label. Default is 0.
                Determines the positive class label, which is 'approved' by default.
        """
        self.label_index = label_index
        super(ClinToxDataset, self).__init__("ClinTox", reload=reload, verbose=verbose)

    def read_in_memory(self, **kwargs):
        super(ClinToxDataset, self).read_in_memory(**kwargs)

        # Strictly one class so we pick drug approved for positive label.
        graph_labels = self.obtain_property("graph_labels")
        graph_labels = [np.array([x[self.label_index]], dtype="float") if x is not None else None for x in graph_labels]
        self.assign_property("graph_labels", graph_labels)
