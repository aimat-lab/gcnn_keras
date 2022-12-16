"""
Module for handling Visual Graph Datasets (VGD).
"""
import os
import typing as t

import numpy as np

from visual_graph_datasets.config import Config
from visual_graph_datasets.util import get_dataset_path, ensure_folder
from visual_graph_datasets.web import get_file_share
from visual_graph_datasets.web import PROVIDER_CLASS_MAP, AbstractFileShare
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.graph.base import GraphDict


class VisualGraphDataset(MemoryGraphDataset):
    r"""Visual graph datasets from external :obj:`visual_graph_datasets` python package.

    Downloading and processing is given in :obj:`visual_graph_datasets` python package.
    For documentation please find `<https://github.com/awa59kst120df/visual_graph_datasets>`_ .
    This class constitutes an interface to kgcnn :obj:`MemoryGraphDataset` data representation.
    """
    
    def __init__(self,
                 name: str,
                 config: Config = Config()):
        super(VisualGraphDataset, self).__init__(
            dataset_name=name,
            file_directory=None,
            file_name=None,
            data_directory=None
        )
        self.vgd_config = config
        self.index_data_map: t.Dict[int, dict] = {}

        self._has_importances: t.Dict[str, bool] = {}

    def ensure(self) -> None:
        """
        This method ensures that the raw dataset exists on the disk. After this method has been called it
        can be certain that the dataset folder exists on the disk and that the folder path is known.

        This is either because an existing dataset folder has been found or because the dataset was
        downloaded.

        Returns:
            None
        """
        # First of all we try to load the dataset, as it might already exist on the system.
        if os.path.exists(os.path.join(self.vgd_config.get_datasets_path(), self.dataset_name)):
            # This function will try to locate a dataset with the given name inside the system's global
            # default folder where all the visual graph datasets are stored. If it does not find a
            # corresponding dataset folder there, an exception is raised.
            self.data_directory = get_dataset_path(
                dataset_name=self.dataset_name,
                datasets_path=self.vgd_config.get_datasets_path()
            )
            return

        # At this point we know that the folder does not already exist which means we need to download the
        # dataset.

        # For this we will first check if a dataset with the given name is even available at the remote
        # file share provider.
        ensure_folder(self.vgd_config.get_datasets_path())
        file_share: AbstractFileShare = get_file_share(self.vgd_config)
        file_share.check_dataset(self.dataset_name)

        # If the dataset is available, then we can download it and then finally load the path
        file_share.download_dataset(self.dataset_name, self.vgd_config.get_datasets_path())
        self.data_directory = get_dataset_path(
            dataset_name=self.dataset_name,
            datasets_path=self.vgd_config.get_datasets_path(),
        )
        self.logger.info(f'visual graph dataset found @ {self.data_directory}')

    def read_in_memory(self) -> None:
        """
        Actually loads the dataset from the file representations into the working memory as GraphDicts
        within the internal MemoryGraphList.

        Returns:
            None
        """
        name_data_map, self.index_data_map = load_visual_graph_dataset(
            self.data_directory,
            logger=self.logger,
            metadata_contains_index=True
        )
        dataset_length = len(self.index_data_map)

        self.empty(dataset_length)
        self.logger.info(f'initialized empty list with {len(self.index_data_map)} elements')

        for index, data in sorted(self.index_data_map.items(), key=lambda t: t[0]):
            g = data['metadata']['graph']

            # In the visual_graph_dataset framework, the train and test split indications are part of the
            # metadata and not the graph itself. We need to set them as graph properties with these
            # specific names however, such that later on the existing base method "get_train_test_indices"
            # of the dataset class can be used.
            if 'train_split' in data['metadata']:
                g['train'] = data['metadata']['train_split']
            if 'test_split' in data['metadata']:
                g['test'] = data['metadata']['test_split']

            # Otherwise the basic structure of the dict from the visual graph dataset should be compatible
            # such that it can be directly used as a GraphDict
            graph_dict = GraphDict(g)
            self[index] = graph_dict

        self.logger.info(f'loaded dataset as MemoryGraphList')

    def visualize_importances(self,
                              output_path: str,
                              gt_importances_suffix: str,
                              node_importances_list: t.List[np.ndarray],
                              edge_importances_list: t.List[np.ndarray],
                              indices: t.List[int],
                              title: str = 'Model',
                              ) -> None:
        data_list = [self.index_data_map[index] for index in indices]

        suffix = gt_importances_suffix
        create_importances_pdf(
            output_path=output_path,
            graph_list=[data['metadata']['graph'] for data in data_list],
            image_path_list=[data['image_path'] for data in data_list],
            node_positions_list=[data['metadata']['graph']['image_node_positions'] for data in data_list],
            importances_map={
                'Ground Truth': (
                    [data['metadata']['graph'][f'node_importances_{suffix}'] for data in data_list],
                    [data['metadata']['graph'][f'edge_importances_{suffix}'] for data in data_list],
                ),
                title: (
                    node_importances_list,
                    edge_importances_list
                )
            }
        )

    # This method loops through all the elements of the dataset which makes it quite computationally
    # expensive, which is why the value will be cached to be more efficient if it is called multiple
    # time redundantly
    def has_importances(self, suffix: t.Union[str, int]) -> bool:
        """
        Returns whether the dataset contains node and edge importances (explanation) annotations which fit
        the given ``suffix``.

        A visual graph dataset may have multiple different importance annotations. The main difference these
        may have is the number of individual importance channels in which the explanations are represented.
        The number of channels is added as a suffix. For example a regression dataset may have the suffix
        "1" which are very simple one-dimensional explanations or the suffix "2" where explanations are
        split into 2 channels where one channel explains the positively influencing input elements and the
        other channel the negatively influencing elements.

        Args:
            suffix: The string suffix of importance annotations to check for.

        Returns:
            boolean
        """
        suffix = str(suffix)

        if suffix not in self._has_importances.keys():
            node_condition = all([f'node_importances_{suffix}' in data['metadata']['graph']
                                  for data in self.index_data_map.values()])

            edge_condition = all([f'edge_importances_{suffix}' in data['metadata']['graph']
                                  for data in self.index_data_map.values()])

            self._has_importances[suffix] = (node_condition and edge_condition)

        return self._has_importances[suffix]

    def get_importances(self,
                        suffix: str,
                        indices: t.Optional[t.List[int]] = None
                        ) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray]]:
        if indices is None:
            indices = list(self.index_data_map.keys())

        node_importances_list: t.List[np.ndarray] = []
        edge_importances_list: t.List[np.ndarray] = []
        for index in indices:
            data = self.index_data_map[index]
            g = data['metadata']['graph']
            node_importances_list.append(g[f'node_importances_{suffix}'])
            edge_importances_list.append(g[f'edge_importances_{suffix}'])

        return node_importances_list, edge_importances_list

    def __hash__(self):
        return hash(self.dataset_name)
