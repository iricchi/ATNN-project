from unittest import TestCase
from src.consensus_clustering import pairwise_accord_naive, are_equal_up_to_bijection
import numpy as np


class PairWiseTest(TestCase):
    def test_pairwise_accord_returns_correct_dimensions(self):
        vector = np.zeros((14,))
        accord = pairwise_accord_naive(vector)
        self.assertEqual(accord.shape[0], 14)
        self.assertEqual(accord.shape[1], 14)

    def test_pairwise_accord_in_simple_case_returns_expected_matrix(self):
        vector = np.asarray([1, 2, 1, 4, 3, 1, 2])
        expected_accord = np.asarray([[1, 0, 1, 0, 0, 1, 0],
                                      [0, 1, 0, 0, 0, 0, 1],
                                      [1, 0, 1, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0],
                                      [1, 0, 1, 0, 0, 1, 0],
                                      [0, 1, 0, 0, 0, 0, 1]]) - np.eye(7)
        self.assertTrue(np.all(expected_accord == pairwise_accord_naive(vector)))

    def test_pairwise_accord_in_trivial_case_returns_expected_matrix(self):
        vector = np.asarray([1, 1, 1, 1])
        expected_accord = np.asarray([[1, 1, 1, 1],
                                     [1, 1, 1, 1],
                                     [1, 1, 1, 1],
                                     [1, 1, 1, 1]]) - np.eye(4)
        self.assertTrue(np.all(expected_accord == pairwise_accord_naive(vector)))

    def test_pairwise_accord_in_two_classes_returns_expected_matrix(self):
        vector = np.asarray([1, 2, 2, 1, 1])
        expected_accord = np.asarray([[1, 0, 0, 1, 1],
                                      [0, 1, 1, 0, 0],
                                      [0, 1, 1, 0, 0],
                                      [1, 0, 0, 1, 1],
                                      [1, 0, 0, 1, 1]]) - np.eye(5)
        self.assertTrue(np.all(expected_accord == pairwise_accord_naive(vector)))



class BijectionEquality(TestCase):
    def test_non_equal_vectors_are_not_equal(self):
        self.assertFalse(are_equal_up_to_bijection(np.asarray([0, 0, 0]), np.asarray([1, 2, 3])))

    def test_vectors_different_length_not_equal(self):
        self.assertFalse(are_equal_up_to_bijection(np.asarray([0, 0, 0]), np.asarray([0, 0])))

    def test_vector_equal_to_itself(self):
        v1 = np.zeros((10,))
        v2 = np.ones((10,))
        v3 = np.asarray([1, 2, 3, 4])
        v4 = np.asarray([5, -5.5, 10.7, 9.6])
        self.assertTrue(are_equal_up_to_bijection(v1, v1))
        self.assertTrue(are_equal_up_to_bijection(v2, v2))
        self.assertTrue(are_equal_up_to_bijection(v3, v3))
        self.assertTrue(are_equal_up_to_bijection(v4, v4))

    def test_vectors_are_equal_when_permuting_values(self):
        v1 = np.asarray([5, -5.5, -5.5, 5])
        v2 = np.asarray([-5.5, 5, 5, -5.5])
        self.assertTrue(are_equal_up_to_bijection(v1, v2))

    def test_vectors_are_equal_when_one_increasing_other_decreasing(self):
        v1 = np.asarray([1, 2, 3, 4, 5])
        v2 = np.asarray([5, 4, 3, 2, 1])
        self.assertTrue(are_equal_up_to_bijection(v1, v2))

    def test_vectors_are_equal_when_arbitrary_permutation_order(self):
        v1 = np.asarray([1, 2, 3, 4, 5])
        v2 = np.asarray([2, 3, 1, 5, 4])
        self.assertTrue(are_equal_up_to_bijection(v1, v2))

    def test_vectors_are_equal_when_bijection_exists(self):
        v1 = np.asarray([1, 2, 3, 4, 5])
        v2 = np.asarray([20, 30, 10, 50, 4])
        self.assertTrue(are_equal_up_to_bijection(v1, v2))

    def test_vectors_not_equal_when_function_is_not_injective(self):
        v1 = np.asarray([1, 2, 3, 4, 5])
        v2 = np.asarray([20, 20, 10, 50, 4])
        self.assertFalse(are_equal_up_to_bijection(v1, v2))

    def test_vectors_not_equal_when_function_is_not_surjective(self):
        v1 = np.asarray([1, 1, 3, 4, 5])
        v2 = np.asarray([20, 30, 10, 50, 4])
        self.assertFalse(are_equal_up_to_bijection(v1, v2))

