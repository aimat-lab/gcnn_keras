import unittest

from kgcnn.layers.disjoint.topk import PoolingTopK

class TestStringMethods(unittest.TestCase):

    def test_pool_multiple_times(self):
        self.assertEqual(1,1)

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)



if __name__ == '__main__':
    unittest.main()