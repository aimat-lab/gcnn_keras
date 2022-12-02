"""
Module for handling Visual Graph Datasets (VGD).
"""
import os
import typing as t

from visual_graph_datasets.config import Config
from visual_graph_datasets.util import get_dataset_path
from visual_graph_datasets.web import PROVIDER_CLASS_MAP, AbstractFileShare
from visual_graph_datasets.data import load_visual_graph_dataset

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.graph.base import GraphDict


class VisualGraphDataset(MemoryGraphDataset):
    
    def __init__(self,
                 name: str):
        super(VisualGraphDataset, self).__init__(
            dataset_name=name,
            file_directory=None,
            file_name=None,
            data_directory=None
        )

        self.vgd_config = Config()
        self.vgd_config.load()

        self.index_data_map: t.Dict[int, dict] = {}

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
        try:
            # This function will try to locate a dataset with the given name inside the system's global
            # default folder where all the visual graph datasets are stored. If it does not find a
            # corresponding dataset folder there, an exception is raised.
            self.data_directory = get_dataset_path(self.dataset_name)
            return
        except FileNotFoundError as e:
            self.logger.info(f'the visual graph dataset "{self.dataset_name}" was not found on the disk. '
                             f'The following exception was raised during lookup:')
            self.logger.info(str(e))

        # At this point we know that the folder does not already exist which means we need to download the
        # dataset.

        # For this we will first check if a dataset with the given name is even available at the remote
        # file share provider.
        file_share_provider: str = self.vgd_config.get_provider()
        file_share_class: type = PROVIDER_CLASS_MAP[file_share_provider]
        file_share: AbstractFileShare = file_share_class(config=self.vgd_config, logger=self.logger)
        file_share.check_dataset(self.dataset_name)

        # If the dataset is available, then we can download it and then finally load the path
        file_share.download_dataset(self.dataset_name, self.vgd_config.get_datasets_path())
        self.data_directory = get_dataset_path(self.dataset_name)
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
            graph_dict = GraphDict(data['graph'])
            self[index] = graph_dict

        self.logger.info(f'loaded dataset as MemoryGraphList')
