import os
import unittest

from kgcnn.data.visual_graph import VisualGraphDataset


class TestVisualGraphDataset(unittest.TestCase):

    def test_basically_works(self):
        # The dataset "mock" is a small dataset which should always be available for testing purposes
        vgd = VisualGraphDataset('mock')

        vgd.ensure()
        self.assertTrue(os.path.exists(vgd.data_directory))

        vgd.read_in_memory()
        self.assertNotEqual(0, len(vgd))
        self.assertEqual(100, len(vgd))
