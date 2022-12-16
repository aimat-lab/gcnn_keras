import unittest
import tempfile
import os
import random
from pprint import pprint

import numpy as np
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from kgcnn.mol.graph_rdkit import MolecularGraphRDKit
from kgcnn.data.moleculenet import MoleculeNetDataset, map_molecule_callbacks
from kgcnn.mol.encoder import OneHotEncoder

from .utils import ASSETS_PATH


class TestMoleculeNetDataset(unittest.TestCase):

    # -- UNITTESTS --

    def test_construction_basically_works(self):
        """
        If it is possible to construct a new instance of MoleculeNetDataset without an error
        """
        csv_path = os.path.join('simple_molecules.csv')

        with tempfile.TemporaryDirectory() as path:
            molecule_net = MoleculeNetDataset(
                data_directory=path,
                file_name=csv_path,
                dataset_name='test'
            )
            self.assertIsInstance(molecule_net, MoleculeNetDataset)

    def test_basically_works(self):
        """
        If one typical processing pipeline can be executed without errors. This includes constructing a
        new instance and then loading the processed molecules from the csv file into memory as GraphList
        instances.
        """
        csv_path = os.path.join(ASSETS_PATH, 'simple_molecules.csv')

        with tempfile.TemporaryDirectory() as path:
            molecule_net = MoleculeNetDataset(
                data_directory=path,
                file_name=csv_path,
                dataset_name='test'
            )

            molecule_net.prepare_data(
                overwrite=False,
                smiles_column_name='smiles'
            )

            molecule_net.read_in_memory(
                label_column_name='label',
                add_hydrogen=False,
            )

            # We know that this csv file has 4 actual data rows
            self.assertEqual(len(molecule_net), 4)

            for molecule in molecule_net:
                self.assertIsInstance(molecule, dict)

    def test_faulty_smiles_do_not_cause_exception(self):
        """
        Even if the source CSV file contains faulty smiles codes, the full molecule net processing should
        run without an exception interrupting the runtime.
        """
        csv_path = os.path.join(ASSETS_PATH, 'faulty_molecules.csv')

        with tempfile.TemporaryDirectory() as path:
            molecule_net = MoleculeNetDataset(
                data_directory=path,
                file_name=csv_path,
                dataset_name='test'
            )

            molecule_net.prepare_data(
                overwrite=False,
                smiles_column_name='smiles'
            )

            molecule_net.read_in_memory(
                label_column_name='label',
                add_hydrogen=False,
            )

            # At this point the test is successful if there was no error, which is what we wanted to
            # show with this test case: Even if the CSV contains faulty SMILES there will be no error
            # which prevents the processing of the dataset
            self.assertEqual(len(molecule_net), 4)

    def test_pandas_from_csv(self):
        """
        Simply test if pandas data frame indexing works as expected.
        """
        csv_path = os.path.join(ASSETS_PATH, 'simple_molecules.csv')
        data = pd.read_csv(csv_path)

        row_index = 0
        # This line is supposed to return a dict for the values of the row of the CSV file with the specified
        # index
        row_dict = dict(data.loc[row_index])
        self.assertEqual(len(row_dict), 4)
        self.assertIn('index', row_dict.keys())
        self.assertIn('label', row_dict.keys())
        self.assertIn('smiles', row_dict.keys())
        self.assertIn('name', row_dict.keys())

    def test_map_molecule_callbacks_basically_works(self):
        """
        Tests if the new function "map_molecule_callbacks" basically works as intended, which is the
        possibility to define custom callbacks based on the CSV data and the molecule object instance
        which add custom properties to the final GraphList object loaded by the dataest.
        """
        csv_path = os.path.join(ASSETS_PATH, 'simple_molecules.csv')

        with tempfile.TemporaryDirectory() as path:

            molecule_net = MoleculeNetDataset(
                data_directory=path,
                file_name=csv_path,
                dataset_name='test'
            )

            callbacks = {
                'name': lambda mg, dd: np.array(dd['name'], dtype='str'),
                'label': lambda mg, dd: np.array(dd['label'], dtype='str')
            }

            # Before we can call the method, the data needs to be prepared because we need to make sure
            # that the molecule file exists before attempting to read from it.
            molecule_net.prepare_data(
                overwrite=False,
                smiles_column_name='smiles'
            )

            # This method will automatically add properties with the string key names of the "callbacks"
            # dict to the underlying GraphList based on the functionality defined by the callback functions.
            mol_values = map_molecule_callbacks(
                mol_list=molecule_net.get_mol_blocks_from_sdf_file(),
                data=molecule_net.read_in_table_file().data_frame,
                callbacks=callbacks,
                mol_interface_class=MolecularGraphRDKit
            )

            for key, value in mol_values.items():
                molecule_net.assign_property(key, value)

            molecule = molecule_net[1]
            self.assertIn('name', molecule)
            self.assertIn('label', molecule)

            # But the other default attributes are not part of it, since we did not specify those in the
            # callbacks dict
            self.assertNotIn('node_symbol', molecule)

    def test_graph_labels_is_one_dimensional_array(self):
        """
        If the shape of the "graph_labels" property of the loaded molecules behaves as expected, which is
        a 0-dimensional np array if only a single target value column is given and a 1-dimensional array
        if multiple columns are given.
        """
        csv_path = os.path.join(ASSETS_PATH, 'simple_molecules.csv')

        # In this first setting we pass a single string as the label_column_name.
        # The corresponding "graph_labels"
        # attribute of the dataset elements should be a 0-dimensional np array containing the values
        # corresponding to the CSV column of that name
        with tempfile.TemporaryDirectory() as path:
            molecule_net = MoleculeNetDataset(
                data_directory=path,
                file_name=csv_path,
                dataset_name='test'
            )
            molecule_net.prepare_data()
            molecule_net.read_in_memory(label_column_name='label')
            molecule_net.set_attributes()

            molecule = molecule_net[0]
            self.assertIsInstance(molecule['graph_labels'], np.ndarray)
            # It is just a single value, aka 0-dimensional array
            self.assertEqual(molecule['graph_labels'].shape, ())

            # But it should also be possible to pass a list of column names in to get a 1-dimensional
            # array as "graph_labels" instead (for multitask learning for example)
            molecule_net = MoleculeNetDataset(
                data_directory=path,
                file_name=csv_path,
                dataset_name='test'
            )
            molecule_net.prepare_data()
            molecule_net.read_in_memory(label_column_name=['label', 'name'])
            molecule_net.set_attributes()

            molecule = molecule_net[0]
            self.assertIsInstance(molecule['graph_labels'], np.ndarray)
            self.assertEqual(molecule['graph_labels'].shape, (2, ))

