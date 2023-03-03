import numpy as np
from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class SIDERDataset(MoleculeNetDataset2018):
    r"""Store and process 'SIDER' dataset from `MoleculeNet <https://moleculenet.org/>`__ database.

    Compare reference:
    `DeepChem <https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#freesolv-dataset>`__ reading:
    The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug reactions (ADR).
    The version of the SIDER dataset in DeepChem has grouped drug side effects into 27 system organ classes following
    MedDRA classifications measured for 1427 approved drugs.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

        - “smiles”: SMILES representation of the molecular structure
        - “Hepatobiliary disorders” ~ “Injury, poisoning and procedural complications”: Recorded side effects for the
           drug. Please refer to `<http://sideeffects.embl.de/se/?page=98>`_ for details on ADRs.

    References:

        (1) Kuhn, Michael, et al. “The SIDER database of drugs and side effects.”
            Nucleic acids research 44.D1 (2015): D1075-D1079.
        (2) Altae-Tran, Han, et al. “Low data drug discovery with one-shot learning.”
            ACS central science 3.4 (2017): 283-293.
        (3) Medical Dictionary for Regulatory Activities. http://www.meddra.org/

    """

    def __init__(self, reload=False, verbose: int = 10):
        """Initialize SIDER dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        super(SIDERDataset, self).__init__("SIDER", reload=reload, verbose=verbose)

    def read_in_memory(self, **kwargs):
        super(SIDERDataset, self).read_in_memory(**kwargs)
        graph_labels = self.obtain_property("graph_labels")
        graph_labels = [np.array(x, dtype="float") for x in graph_labels]
        self.assign_property("graph_labels", graph_labels)
