import unittest
import tempfile
import os
import random
from pprint import pprint

import numpy as np
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.mol.encoder import OneHotEncoder


SIMPLE_SMILES_CSV = """
index,name,label,smiles
1,Propanolol,1,[Cl].CC(C)NCC(O)COc1cccc2ccccc12
2,Terbutylchlorambucil,1,C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl
3,40730,1,c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O
4,24,1,C1CCN(CC1)Cc1cccc(c1)OCCCNC(=O)C
"""


FAULTY_SMILES_CSV = """
index,name,label,smiles
57,compound 36,1,CN(C)Cc1ccc(CSCCNC2=C([N+]([O-])=O)C(Cc3ccccc3)=CN2)o1
58,19,0,CNC(=NC#N)Nc1cccc(c1)c1csc(n1)N=C(N)N
59,Y-G 14,1,n(ccc1)c(c1)CCNC
60,15,1,O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3
"""


class TestMoleculeNetDataset(unittest.TestCase):

    # MoleculeNetDataset needs a folder in which it can operate. We create a fixture which will set up a new temp
    # folder for each test case.

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.temp_dir = None
        self.temp_path = None
        self.file_name = 'test.csv'

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.__enter__()

    def tearDown(self):
        self.temp_dir.__exit__(None, None, None)

    # This is a utility method. It will create a CSV file as the source for the dataset in the current temporary folder
    # using the string CSV content
    def write_string(self, string: str):
        with open(os.path.join(self.temp_path, self.file_name), mode='w') as file:
            file.write(string)

    # ~ ACTUAL TEST CASES

    def test_construction_basically_works(self):
        self.write_string(SIMPLE_SMILES_CSV)
        molnet = MoleculeNetDataset(data_directory=self.temp_path, file_name=self.file_name, dataset_name='test')
        self.assertIsInstance(molnet, MoleculeNetDataset)

    def test_basically_works(self):
        self.write_string(SIMPLE_SMILES_CSV)
        molnet = MoleculeNetDataset(data_directory=self.temp_path, file_name=self.file_name, dataset_name='test')

        molnet.prepare_data(
            overwrite=False,
            smiles_column_name='smiles'
        )

        molnet.read_in_memory(
            label_column_name='label',
            add_hydrogen=False,
        )
        # We know that this csv data has 4 entries
        self.assertEqual(len(molnet), 4)

        # Basic tests for one of the entries
        data = molnet[random.randint(0, 3)]
        self.assertIsInstance(data, dict)
        # These are the basic fields which every entry is supposed to have and all of them need to be
        # numpy arrays
        keys = ['graph_size', 'graph_labels', 'node_coordinates', 'node_number',
                'node_symbol', 'edge_indices']
        for key in keys:
            self.assertIn(key, data)
            print(type(data[key]))
            self.assertIsInstance(data[key], np.ndarray)

    def test_setting_attributes_works(self):
        self.write_string(SIMPLE_SMILES_CSV)
        molnet = MoleculeNetDataset(data_directory=self.temp_path, file_name=self.file_name, dataset_name='test')

        # This is list the keys which the dataset as a whole should contain after they have been read
        # to the memory
        default_keys = ['graph_size', 'graph_labels', 'node_coordinates', 'node_number',
                        'node_symbol', 'edge_indices']
        # This is the list of keys, which the data set should contain after the "set_attributes" has been
        # called
        attribute_keys = ['graph_attributes', 'node_attributes', 'edge_attributes']

        molnet.prepare_data(
            overwrite=False,
            smiles_column_name='smiles'
        )

        molnet.read_in_memory(
            label_column_name='label',
            add_hydrogen=False,
        )
        pprint(molnet.__dict__.keys())

        data = molnet[random.randint(0, 3)]
        # At this point, the default keys should be present but not the attribute keys
        for key in default_keys:
            self.assertIn(key, data)

        for key in attribute_keys:
            self.assertNotIn(key, data)

        molnet.set_attributes(
            nodes=['Symbol'],
            encoder_nodes={'Symbol': OneHotEncoder(['C', 'O'], dtype='str', add_unknown=True)},
            edges=['BondType'],
            encoder_edges={'BondType': int}
        )

        # We know that this csv data has 4 entries
        self.assertEqual(len(molnet), 4)

        data = molnet[random.randint(0, 3)]
        # Now also the attribute keys should be present.
        for key in default_keys:
            self.assertIn(key, data)

        for key in attribute_keys:
            self.assertIn(key, data)

    def test_faulty_smiles_do_not_cause_exception(self):
        self.write_string(FAULTY_SMILES_CSV)
        molnet = MoleculeNetDataset(data_directory=self.temp_path, file_name=self.file_name, dataset_name='test')

        molnet.prepare_data(
            overwrite=False,
            smiles_column_name='smiles'
        )

        molnet.read_in_memory(
            label_column_name='label',
            add_hydrogen=False,
        )

        # At this point the test is successful if there was no error...
        self.assertEqual(len(molnet), 4)

    def test_pandas_from_csv(self):
        """
        Simply test if pandas data frame indexing works as expected.
        """
        self.write_string(SIMPLE_SMILES_CSV)
        data = pd.read_csv(os.path.join(self.temp_path, self.file_name))

        row_index = 0
        # This line is supposed to return a dict for the values of the row of the CSV file with the specified
        # index
        row_dict = dict(data.loc[row_index])
        self.assertEqual(len(row_dict), 4)
        self.assertIn('index', row_dict.keys())
        self.assertIn('label', row_dict.keys())
        self.assertIn('smiles', row_dict.keys())
        self.assertIn('name', row_dict.keys())

        self.assertFalse(True)

    def test_map_molecule_attributes_basically_works(self):
        """
        Tests if the new method "._map_molecule_callbacks" works on a simple example
        """
        self.write_string(SIMPLE_SMILES_CSV)
        molnet = MoleculeNetDataset(data_directory=self.temp_path, file_name=self.file_name, dataset_name='test')

        callbacks = {
            'name': lambda mg, dd: np.array(dd['name'], dtype='str'),
            'label': lambda mg, dd: np.array(dd['label'], dtype='str')
        }

        # Before we can call the method, the data needs to be prepared because we need to make sure that the molecule
        # file exists before attempting to read from it.
        molnet.prepare_data(
            overwrite=False,
            smiles_column_name='smiles'
        )
        # This method will automatically add properties with the string key names of the "callbacks" dict to the
        # underlying GraphList based on the transformations defined by the callback functions.
        molnet._map_molecule_callbacks(callbacks)

        molecule = molnet[1]
        self.assertIn('name', molecule)
        self.assertIn('label', molecule)

        # But the other default attributes are not part of it, since we did not specify those in the
        # callbacks dict
        self.assertNotIn('node_symbol', molecule)

    def test_graph_labels_is_one_dimensional_array(self):
        self.write_string(SIMPLE_SMILES_CSV)

        # In this first setting we pass a single string as the label name. The corresponding "graph_labels"
        # attribute of the dataset elements should be a one-dim np array regardless (and not zero dim)
        molnet = MoleculeNetDataset(data_directory=self.temp_path, file_name=self.file_name, dataset_name='test')
        molnet.prepare_data()
        molnet.read_in_memory(label_column_name='label')
        molnet.set_attributes()

        molecule = molnet[0]
        self.assertIsInstance(molecule['graph_labels'], np.ndarray)
        self.assertEqual(len(molecule['graph_labels'].shape), 1)

        # But it should also be possible to pass a list of column names in to get a true np array of
        # multiple labels
        molnet = MoleculeNetDataset(data_directory=self.temp_path, file_name=self.file_name, dataset_name='test')
        molnet.prepare_data()
        molnet.read_in_memory(label_column_name=['label', 'name'])
        molnet.set_attributes()

        molecule = molnet[0]
        self.assertIsInstance(molecule['graph_labels'], np.ndarray)
        self.assertEqual(len(molecule['graph_labels'].shape), 1)
        self.assertEqual(len(molecule['graph_labels']), 2)
        # Testing if it is the correct value as well
        self.assertEqual(molecule['graph_labels'][1], 'Propanolol')
