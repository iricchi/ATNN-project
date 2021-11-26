from unittest import TestCase
import networkx as nx
import numpy as np
from src.graph_measures import create_communities_from_partition, compute_degree, compute_participation_coefficient, \
    compute_system_segregation, compute_within_degree


class CreateCommunity(TestCase):
    def check_graphs_equal(self, g1, g2):
        # One way to check that the graphs are equal: both should have the same edge set
        self.assertListEqual(list(g1.edges), list(g2.edges))

        # Furthermore, all edges should have the same weights
        for e in list(g1.edges):
            n1, n2 = e[0], e[1]
            self.assertEqual(g1[n1][n2]['weight'], g2[n1][n2]['weight'])

    def test_trivial_partition_should_return_original_graph(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 1, 1, 1])
        subgraphs = create_communities_from_partition(G, partition)
        # Should be exactly one subgraph
        self.assertEqual(len(subgraphs), 1)
        # That subgraph should have all nodes of G
        self.check_graphs_equal(subgraphs[0], G)

    def test_excluding_nodes_from_partition_yields_exception(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 1, 1])
        self.assertRaises(AssertionError, create_communities_from_partition, G, partition)

    def test_partition_values_can_be_negative(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([-1, -1, -1, -1])
        subgraphs = create_communities_from_partition(G, partition)
        self.check_graphs_equal(subgraphs[0], G)

    def test_bipartite_partition_creates_two_subgraphs(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 2, 1, 2])
        subgraphs = create_communities_from_partition(G, partition)
        self.assertEqual(len(subgraphs), 2)

    def test_bipartite_partition_creates_correct_subgraphs_unweighted(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 2, 1, 2])
        expected_subgraph_1 = nx.from_numpy_array(np.asarray([[0, 0, 1, 0],
                                                              [0, 0, 0, 0],
                                                              [1, 0, 1, 0],
                                                              [0, 0, 0, 0]]))
        expected_subgraph_2 = nx.from_numpy_array(np.asarray([[0, 0, 0, 0],
                                                              [0, 0, 0, 1],
                                                              [0, 0, 0, 0],
                                                              [0, 1, 0, 0]]))
        subgraphs = create_communities_from_partition(G, partition)

        self.check_graphs_equal(subgraphs[0], expected_subgraph_1)
        self.check_graphs_equal(subgraphs[1], expected_subgraph_2)

    def test_partition_values_negative_and_positive_returns_first_the_community_with_negative_partition_value(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, -1, 1, -1])
        expected_subgraph_1 = nx.from_numpy_array(np.asarray([[0, 0, 1, 0],
                                                              [0, 0, 0, 0],
                                                              [1, 0, 1, 0],
                                                              [0, 0, 0, 0]]))
        expected_subgraph_2 = nx.from_numpy_array(np.asarray([[0, 0, 0, 0],
                                                              [0, 0, 0, 1],
                                                              [0, 0, 0, 0],
                                                              [0, 1, 0, 0]]))
        subgraphs = create_communities_from_partition(G, partition)

        self.check_graphs_equal(subgraphs[0], expected_subgraph_2)
        self.check_graphs_equal(subgraphs[1], expected_subgraph_1)

    def test_threepartite_partition_creates_exactly_three_subgraphs(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 2, 1, 3])
        subgraphs = create_communities_from_partition(G, partition)
        self.assertEqual(len(subgraphs), 3)

    def test_threepartie_partition_creates_correct_subgraphs_weighted(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 1, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 2, 1, 3])
        expected_subgraph_1 = nx.from_numpy_array(np.asarray([[0, 0, 1, 0],
                                                              [0, 0, 0, 0],
                                                              [1, 0, 1, 0],
                                                              [0, 0, 0, 0]]))
        expected_subgraph_2 = nx.from_numpy_array(np.asarray([[0, 0, 0, 0],
                                                              [0, 1, 0, 0],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0]]))

        expected_subgraph_3 = nx.from_numpy_array(np.asarray([[0, 0, 0, 0],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0]]))

        subgraphs = create_communities_from_partition(G, partition)

        self.check_graphs_equal(subgraphs[0], expected_subgraph_1)
        self.check_graphs_equal(subgraphs[1], expected_subgraph_2)
        self.check_graphs_equal(subgraphs[2], expected_subgraph_3)


class DegreeComputationTest(TestCase):
    def test_compute_degree_returns_nodes_of_graph_correctly_without_stand_without_weight(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=False, weighted=False)
        self.assertListEqual(list(nodes), list(G.nodes))

    def test_compute_degree_returns_nodes_of_graph_correctly_without_stand_with_weight(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=False, weighted=True)
        self.assertListEqual(list(nodes), list(G.nodes))

    def test_compute_degree_returns_nodes_of_graph_correctly_with_stand_without_weight(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=True, weighted=False)
        self.assertListEqual(list(nodes), list(G.nodes))

    def test_compute_degree_returns_nodes_of_graph_correctly_with_stand_with_weight(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=True, weighted=True)
        self.assertListEqual(list(nodes), list(G.nodes))

    def test_self_loop_means_degree_of_two(self):
        adjacency_matrix = np.asarray([[1]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=False, weighted=False)
        expected_degrees = [2]
        self.assertListEqual(list(degrees), expected_degrees)

    def test_compute_degree_returns_degrees_without_stand_without_weight(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=False, weighted=False)
        expected_degrees = [2, 3, 4, 1]
        self.assertListEqual(list(degrees), expected_degrees)

    def test_compute_degree_returns_weighted_degrees_without_stand_with_weight(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=False, weighted=True)
        expected_degrees = [2, 3, 4, 1]
        self.assertListEqual(list(degrees), expected_degrees)

    def test_compute_degree_returns_weighted_degrees_weighted_adjacency_matrix_without_stand(self):
        adjacency_matrix = np.asarray([[0, 10, 20.0, 0],
                                       [10, 0, -4.3, 7.8],
                                       [20, -4.3, 21.6, 0],
                                       [0, 7.8, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=False, weighted=True)
        expected_degrees = [30, 13.5, 58.9, 7.8]
        self.assertListEqual(list(degrees), expected_degrees)

    def test_compute_degree_standardized_returns_expected_results(self):
        adjacency_matrix = np.asarray([[0, 10, 20.0, 0],
                                       [10, 0, -4.3, 7.8],
                                       [20, -4.3, 21.6, 0],
                                       [0, 0, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        nodes, degrees = compute_degree(G, standardize=True, weighted=True)
        expected_degrees = np.asarray([30, 13.5, 58.9, 7.8])
        expected_degrees = (expected_degrees - expected_degrees.mean()) / expected_degrees.std()
        self.assertListEqual(list(degrees), list(expected_degrees))


class ParticipationCoefficientTest(TestCase):
    def test_compute_participation_coefficient(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 2, 1, 2])
        # We know the closed form formula of participation coefficient:
        expected_pc = [1.0 - (1 / 2.) ** 2 - (1 / 2.) ** 2, 1.0 - (1 / 3.) ** 2 - (2 / 3.) ** 2,
                       1.0 - (1 / 4.) ** 2 - (3 / 4.) ** 2, 0.0]
        pc = compute_participation_coefficient(G, False, partition_values=partition)
        self.assertListEqual(list(pc), expected_pc)

    def test_participation_coefficient_identical_weighted_unweighted_when_adjacency_is_binary(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 0],
                                       [1, 0, 1, 1],
                                       [1, 1, 1, 0],
                                       [0, 1, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 2, 1, 2])
        # We know the closed form formula of participation coefficient:
        pc = compute_participation_coefficient(G, False, partition_values=partition)
        pc_2 = compute_participation_coefficient(G, True, partition_values=partition)
        self.assertListEqual(list(pc), list(pc_2))

    def test_participation_coefficient_correct_with_self_loops_weighted_adjacency(self):
        adjacency_matrix = np.asarray([[0, 10, -4, 0],
                                       [10, 0, 31, -4],
                                       [-4, 31, 3, 0],
                                       [0, -4, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 2, 1, 2])
        # We know the closed form formula of participation coefficient:
        expected_pc = np.asarray(
            [1. - (10. / 6) ** 2 - (-4 / 6.) ** 2, 1. - (41 / (10 + 31 - 4)) ** 2 - (-4 / (10. + 31 - 4)) ** 2,
             1. - (2. / 33) ** 2 - (31 / 33.) ** 2, 0.0])
        pc = compute_participation_coefficient(G, True, partition_values=partition)

        # Of course we're dealing with floating point numbers, so we can't test exact equality
        self.assertTrue(np.all(np.abs(pc - expected_pc) < 10e-8))

    def test_participation_coefficient_with_three_communities_no_self_loop(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 1, 0, 0],
                                       [1, 0, 1, 0, 0, 0],
                                       [1, 1, 0, 1, 0, 0],
                                       [1, 0, 1, 0, 1, 1],
                                       [0, 0, 0, 1, 0, 1],
                                       [0, 0, 0, 1, 1, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 1, 1, 2, 2, 3])
        # We know the closed form formula of participation coefficient:
        expected_pc = np.asarray(
            [1. - (2. / 3) ** 2 - (1 / 3.) ** 2, 0, 1. - (2 / 3) ** 2 - (1 / 3) ** 2,
             1. - (1 / 4) ** 2 - (2 / 4) ** 2 - (1 / 4) ** 2, 1. - (1 / 2) ** 2 - (1 / 2) ** 2, 0])
        pc = compute_participation_coefficient(G, True, partition_values=partition)

        # Of course we're dealing with floating point numbers, so we can't test exact equality
        self.assertTrue(np.all(np.abs(pc - expected_pc) < 10e-8))

    def test_participation_coefficient_of_disconnected_nodes_is_zero(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 1, 0, 0],
                                       [1, 0, 1, 0, 0, 0],
                                       [1, 1, 0, 1, 0, 0],
                                       [1, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 1, 1, 2, 2, 3])
        # We know the closed form formula of participation coefficient:
        expected_pc = np.asarray(
            [1. - (2. / 3) ** 2 - (1 / 3.) ** 2, 0, 1. - (2 / 3) ** 2 - (1 / 3) ** 2,
             0, 0, 0])
        pc = compute_participation_coefficient(G, True, partition_values=partition)

        # Of course we're dealing with floating point numbers, so we can't test exact equality
        self.assertTrue(np.all(np.abs(pc - expected_pc) < 10e-8))


class SystemSegregationTest(TestCase):
    def test_compute_system_segregation(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 1, 0, 0],
                                       [1, 0, 1, 0, 0, 0],
                                       [1, 1, 0, 1, 0, 0],
                                       [1, 0, 1, 0, 1, 1],
                                       [0, 0, 0, 1, 0, 1],
                                       [0, 0, 0, 1, 1, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 1, 1, 2, 2, 3])

        # System segregation is a single value.
        # It is equal to mean strength of edges between nodes of same network minus mean strength of edges spanning
        # two networks, divided by the former.
        mean_strength_within_net = 1.0  # All weights are 1 here
        mean_strength_across_net = 1.0  # All weights are 1 here
        expected_seg = (mean_strength_within_net - mean_strength_across_net) / mean_strength_within_net
        seg = compute_system_segregation(G, partition)
        self.assertEqual(expected_seg, seg)

    def test_compute_system_segregation_weighted(self):
        adjacency_matrix = np.asarray([[0, 0.5, 3, 4, 0, 0],
                                       [0.5, 0, -2, 0, 0, 0],
                                       [3, -2, 0, 1, 0, 0],
                                       [4, 0, 1, 0, 21, -5],
                                       [0, 0, 0, 21, 0, 1],
                                       [0, 0, 0, -5, 1, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 1, 1, 2, 2, 3])

        # System segregation is a single value.
        # It is equal to mean strength of edges between nodes of same network minus mean strength of edges spanning
        # two networks, divided by the former.
        mean_strength_within_net = (0.5 + 3 - 2 + 21) / 4
        mean_strength_across_net = (4 + 1 + 1 - 5) / 4.
        expected_seg = (mean_strength_within_net - mean_strength_across_net) / mean_strength_within_net
        seg = compute_system_segregation(G, partition)
        self.assertEqual(expected_seg, seg)


class WithinDegreeTest(TestCase):
    def standardize(self, x):
        return (x - x.mean())/x.std()

    def test_compute_within_degree(self):
        adjacency_matrix = np.asarray([[0, 1, 1, 1, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 1, 0, 0],
                                       [1, 0, 1, 0, 1, 1],
                                       [0, 0, 0, 1, 0, 1],
                                       [0, 0, 0, 1, 1, 0]])
        G = nx.from_numpy_array(adjacency_matrix)
        partition = np.asarray([1, 1, 1, 2, 2, 3])
        modules = create_communities_from_partition(G, partition)

        # The expected values can be classified in two types.
        # For the first community, we will perform normal standardization
        # For the other two, since they have the same degrees, the standard deviation is undefined.
        # Our convention in this case is that within-module degree is zero, since it measures deviation from mean.
        community_1_degrees = np.asarray([2, 1, 1])
        community_1_within = self.standardize(community_1_degrees)

        expected_within_degrees = np.hstack((community_1_within, [0,0], [0]))
        within_degrees = compute_within_degree(modules)
        self.assertListEqual(list(expected_within_degrees), list(within_degrees))
