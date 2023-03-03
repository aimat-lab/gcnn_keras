import os

from kgcnn.data.tudataset import GraphTUDataset
from kgcnn.data.download import DownloadDataset


class GraphTUDataset2020(GraphTUDataset, DownloadDataset):
    r"""Base class for loading graph datasets published by `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_ .

    This general base class has functionality to load TUDatasets in a generic way.

    .. note::

        Note that subclasses of `GraphTUDataset` in :obj:``kgcnn.data.datasets`` should still be made,
        if the dataset needs more refined post-precessing. Not all datasets can provide all types of graph
        properties like `edge_attributes` etc.

    References:

        (1) TUDataset: A collection of benchmark datasets for learning with graphs.
            Christopher Morris and Nils M. Kriege and Franka Bause and Kristian Kersting and Petra Mutzel and
            Marion Neumann, ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020)
            `<www.graphlearning.io>`_ .

    """

    # List of datasets in TUDatasets.
    tudataset_ids = [
        # Molecules
        "AIDS", "alchemy_full", "aspirin", "benzene", "BZR", "BZR_MD", "COX2", "COX2_MD", "DHFR", "DHFR_MD", "ER_MD",
        "ethanol", "FRANKENSTEIN", "malonaldehyde", "MCF-7", "MCF-7H", "MOLT-4", "MOLT-4H", "Mutagenicity", "MUTAG",
        "naphthalene", "NCI1", "NCI109", "NCI-H23", "NCI-H23H", "OVCAR-8", "OVCAR-8H", "P388", "P388H", "PC-3", "PC-3H",
        "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "QM9", "salicylic_acid", "SF-295", "SF-295H", "SN12C", "SN12CH",
        "SW-620", "SW-620H", "toluene", "Tox21_AhR_training", "Tox21_AhR_testing", "Tox21_AhR_evaluation",
        "Tox21_AR_training", "Tox21_AR_testing", "Tox21_AR_evaluation", "Tox21_AR-LBD_training", "Tox21_AR-LBD_testing",
        "Tox21_AR-LBD_evaluation", "Tox21_ARE_training", "Tox21_ARE_testing", "Tox21_ARE_evaluation",
        "Tox21_aromatase_training", "Tox21_aromatase_testing", "Tox21_aromatase_evaluation", "Tox21_ATAD5_training",
        "Tox21_ATAD5_testing", "Tox21_ATAD5_evaluation", "Tox21_ER_training", "Tox21_ER_testing", "Tox21_ER_evaluation",
        "Tox21_ER-LBD_training", "Tox21_ER-LBD_testing", "Tox21_ER-LBD_evaluation", "Tox21_HSE_training",
        "Tox21_HSE_testing", "Tox21_HSE_evaluation", "Tox21_MMP_training", "Tox21_MMP_testing", "Tox21_MMP_evaluation",
        "Tox21_p53_training", "Tox21_p53_testing", "Tox21_p53_evaluation", "Tox21_PPAR-gamma_training",
        "Tox21_PPAR-gamma_testing", "Tox21_PPAR-gamma_evaluation", "UACC257", "UACC257H", "uracil", "Yeast", "YeastH",
        "ZINC_full", "ZINC_test", "ZINC_train", "ZINC_val",
        # Bioinformatics
        "DD", "ENZYMES", "KKI", "OHSU", "Peking_1", "PROTEINS", "PROTEINS_full",
        # Computer vision
        "COIL-DEL", "COIL-RAG", "Cuneiform", "Fingerprint", "FIRSTMM_DB", "Letter-high", "Letter-low", "Letter-med",
        "MSRC_9", "MSRC_21", "MSRC_21C",
        # Social networks
        "COLLAB", "dblp_ct1", "dblp_ct2", "DBLP_v1", "deezer_ego_nets", "facebook_ct1", "facebook_ct2",
        "github_stargazers", "highschool_ct1", "highschool_ct2", "IMDB-BINARY", "IMDB-MULTI", "infectious_ct1",
        "infectious_ct2", "mit_ct1", "mit_ct2", "REDDIT-BINARY", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K",
        "reddit_threads", "tumblr_ct1", "tumblr_ct2", "twitch_egos", "TWITTER-Real-Graph-Partial",
        # Synthetic
        "COLORS-3", "SYNTHETIC", "SYNTHETICnew", "Synthie", "TRIANGLES"
    ]

    def __init__(self, dataset_name: str, reload: bool = False, verbose: int = 10):
        """Initialize a `GraphTUDataset` instance from string identifier.

        Args:
            dataset_name (str): Name of a dataset.
            reload (bool): Download the dataset again and prepare data on disk.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        if not isinstance(dataset_name, str):
            raise ValueError("Please provide string identifier for TUDataset.")

        GraphTUDataset.__init__(self, verbose=verbose, dataset_name=dataset_name)

        # Prepare download
        if dataset_name in self.tudataset_ids:
            self.download_info = {
                "data_directory_name": dataset_name,
                "download_url": "https://www.chrsmrrs.com/graphkerneldatasets/" + dataset_name + ".zip",
                "download_file_name": dataset_name + ".zip",
                "unpack_zip": True,
                "unpack_directory_name": dataset_name,
                "dataset_name": dataset_name
            }
        else:
            raise ValueError("Can not resolve %s as a TUDataset." % dataset_name,
                             "Add to `all_tudataset_identifier` list manually.")

        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_directory = os.path.join(self.unpack_directory_name, dataset_name)
        self.fits_in_memory = True

        if self.fits_in_memory:
            self.read_in_memory()

    @staticmethod
    def _debug_read_list():
        line_ids = []
        with open("datasets.md", 'r') as f:
            for line in f.readlines():
                if line[:3] == "|**":
                    line_ids.append(line.split("**")[1])
        return line_ids
