import unittest
import numpy as np

from kgcnn.graph.geom import range_neighbour_lattice
# from kgcnn.data.datasets.MatProjectEFormDataset import MatProjectEFormDataset


class TestRangePeriodic(unittest.TestCase):
    # Handmade test case.
    artificial_lattice = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    artificial_atoms = np.array([[0.1, 0.0, 0.0], [0.5, 0.5, 0.5]])

    # Example from dataset, where lattice and atom coordinates do not really align.
    real_lattice = np.array([[-8.71172704, -0., -5.02971843],
                             [-10.97279872, -0.01635133, 8.94600922],
                             [-6.5538005, 12.48246168, 1.29207947]])
    real_atoms = np.array([[-24.14652308, 12.46611035, 6.41607351],
                           [-2.09180318, 0., -1.20770325],
                           [0., 0., 0.],
                           [-4.35586352, 0., -2.51485921]])

    def test_nn_range(self):

        def _test(atom, lattice, max_r):
            indices_max, images_max, dist_max = range_neighbour_lattice(atom,
                                                                        lattice, max_distance=max_r,
                                                                        max_neighbours=None)
            # For simplicity pick atom 0 to check NN.
            dist_max_0 = dist_max[indices_max[:, 0] == 0]

            test_x_results = []
            for x in [5, 10, 50, 100, 500, 1000]:
                if len(dist_max_0) < x:
                    print("Too small R for NN. Please increase for test.")
                indices, images, dist = range_neighbour_lattice(atom, lattice, max_distance=None, max_neighbours=x)
                test_num = len(indices) == len(atom) * x
                dist_0 = dist[indices[:, 0] == 0]
                test_correct_nn = np.amax(np.abs(dist_max_0[: len(dist_0)] - dist_0)) < 1e-6
                if not test_correct_nn:
                    print(np.abs(dist_max_0[: len(dist_0)] - dist_0))
                test_x_results.append(test_num and test_correct_nn)
            return test_x_results

        self.assertTrue(all(_test(self.artificial_atoms, self.artificial_lattice, 5.0)))
        self.assertTrue(all(_test(self.real_atoms, self.real_lattice, 100.0)))

    def test_dist_range(self):

        test_distance = [3.0, 4.0, 10.0, 20.0]

        def _test(atoms, lattice, ref_num_dist):
            # ref_num_dist = [len(self.compare_reference(atoms, lattice, x)[0])
            #                 for x in test_distance]
            num_dist = [len(range_neighbour_lattice(atoms,
                                                    lattice, max_distance=x,
                                                    max_neighbours=None)[0])
                        for x in test_distance]
            test_results = []
            for x, y in zip(num_dist, ref_num_dist):
                # print(x, y)
                test_results.append(x == y)
            return test_results

        self.assertTrue(all(_test(self.artificial_atoms, self.artificial_lattice, [484, 1072, 16736, 133816])))
        self.assertTrue(all(_test(self.real_atoms, self.real_lattice, [8, 8, 24, 320])))

    # def test_dist_all_correct(self):
    #     indices, images, dist = range_neighbour_lattice(self.artificial_atoms,
    #                                                     self.artificial_lattice, max_distance=5.0,
    #                                                     max_neighbours=None)
    #     indices, images, dist = self.full_sort(indices, images, dist)
    #
    #     ref_indices, ref_images, ref_dist = self.compare_reference(self.artificial_atoms,
    #     self.artificial_lattice, 5.0)
    #     ref_indices, ref_images, ref_dist = self.full_sort(ref_indices, ref_images, ref_dist)
    #     self.assertTrue(np.amax(np.abs(dist - ref_dist)) < 1e-6)
    #     self.assertTrue(np.amax(np.abs(ref_images - images)) < 1e-6)
    #     self.assertTrue(np.amax(np.abs(ref_indices - indices)) < 1e-6)
    #
    #     indices, images, dist = range_neighbour_lattice(self.real_atoms,
    #                                                     self.real_lattice, max_distance=5.0,
    #                                                     max_neighbours=None)
    #     if len(indices) <= 0:
    #         return
    #     indices, images, dist = self.full_sort(indices, images, dist)
    #
    #     ref_indices, ref_images, ref_dist = self.compare_reference(self.real_atoms, self.real_lattice, 5.0)
    #     ref_indices, ref_images, ref_dist = self.full_sort(ref_indices, ref_images, ref_dist)
    #     self.assertTrue(np.amax(np.abs(dist - ref_dist)) < 1e-6)
    #     self.assertTrue(np.amax(np.abs(ref_images - images)) < 1e-6)
    #     self.assertTrue(np.amax(np.abs(ref_indices - indices)) < 1e-6)

    def set_real_lattice_from_data(self, data, i):
        self.real_lattice = data[i]["graph_lattice"]
        self.real_atoms = data[i]["node_coordinates"]

    @staticmethod
    def full_sort(indices, images, dist):
        def reorder(order, *args):
            return [x[order] for x in args]

        s = np.argsort(dist, kind="stable")
        indices, images, dist = reorder(s, indices, images, dist)
        s = np.argsort(images[:, 2], kind="stable")
        indices, images, dist = reorder(s, indices, images, dist)
        s = np.argsort(images[:, 1], kind="stable")
        indices, images, dist = reorder(s, indices, images, dist)
        s = np.argsort(images[:, 0], kind="stable")
        indices, images, dist = reorder(s, indices, images, dist)
        s = np.argsort(indices[:, 1], kind="stable")
        indices, images, dist = reorder(s, indices, images, dist)
        s = np.argsort(indices[:, 0], kind="stable")
        indices, images, dist = reorder(s, indices, images, dist)
        return indices, images, dist

    # @staticmethod
    # def compare_reference(coordinates, lattice, max_distance):
    #     from pymatgen.core.structure import Structure
    #     py_struct = Structure(lattice, species=["C"] * len(coordinates),
    #                           coords=np.dot(coordinates, np.linalg.inv(lattice)))
    #     all_nbrs = py_struct.get_all_neighbors_py(max_distance, include_index=True)
    #
    #     all_edge_distance = []
    #     all_edge_indices = []
    #     all_edge_image = []
    #     for i, start_site in enumerate(all_nbrs):
    #         edge_distance = []
    #         edge_indices = []
    #         edge_image = []
    #         for j, stop_site in enumerate(start_site):
    #             edge_distance.append(stop_site.nn_distance)
    #             edge_indices.append([i, stop_site.index])
    #             edge_image.append(stop_site.image)
    #         # Sort after distance
    #         edge_distance = np.array(edge_distance)
    #         order_dist = np.argsort(edge_distance)
    #         edge_distance = edge_distance[order_dist]
    #         edge_indices = np.array(edge_indices, dtype="int")[order_dist]
    #         edge_image = np.array(edge_image, dtype="int")[order_dist]
    #         # Append to index list
    #         if len(edge_distance) > 0:
    #             all_edge_distance.append(edge_distance)
    #             all_edge_indices.append(edge_indices)
    #             all_edge_image.append(edge_image)
    #
    #     if len(all_edge_distance) > 0:
    #         all_edge_distance = np.concatenate(all_edge_distance, axis=0)
    #     if len(all_edge_indices) > 0:
    #         all_edge_indices = np.concatenate(all_edge_indices, axis=0)
    #     if len(all_edge_image) > 0:
    #         all_edge_image = np.concatenate(all_edge_image, axis=0)
    #     return all_edge_indices, all_edge_image, all_edge_distance


# dataset = MatProjectEFormDataset()

if __name__ == '__main__':
    # test = TestRangePeriodic()
    # test.test_dist_range()
    # test.test_nn_range()
    # for i in range(20000, len(dataset)):
    #     if i % 1000 == 0:
    #         print("..%s" % i)
    #     test.set_real_lattice_from_data(dataset, i)
    #     test.test_nn_range()
    unittest.main()
