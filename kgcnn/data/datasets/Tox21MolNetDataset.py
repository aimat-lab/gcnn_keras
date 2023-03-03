import numpy as np
from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class Tox21MolNetDataset(MoleculeNetDataset2018):
    r"""Store and process 'TOX21' dataset from `MoleculeNet <https://moleculenet.org/>`__ database.

    Compare reference:
    `DeepChem <https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#freesolv-dataset>`__ reading:
    The “Toxicology in the 21st Century” (Tox21) initiative created a public database measuring toxicity of compounds,
    which has been used in the 2014 Tox21 Data Challenge. This dataset contains qualitative toxicity measurements for
    8k compounds on 12 different targets, including nuclear receptors and stress response pathways.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

        - “smiles”: SMILES representation of the molecular structure
        - “NR-XXX”: Nuclear receptor signaling bioassays results
        - “SR-XXX”: Stress response bioassays results

    please refer to https://tripod.nih.gov/tox21/challenge/data.jsp for details.

    References:

        (1) Tox21 Challenge. https://tripod.nih.gov/tox21/challenge/

    """

    def __init__(self, reload=False, verbose: int = 10, remove_nan: bool = False):
        """Initialize Tox21 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
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
