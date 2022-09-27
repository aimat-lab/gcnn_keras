import unittest

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.literature.MEGAN import MEGAN


class TestMegan(unittest.TestCase):

    def test_construction_basically_works(self):
        model = MEGAN(units=[1])
        self.assertIsInstance(model, MEGAN)
        self.assertIsInstance(model, ks.models.Model)