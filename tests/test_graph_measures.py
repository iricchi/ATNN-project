from unittest import TestCase
from src.graph_measures import create_communities_from_partition, compute_degree
import networkx as nx
import numpy as np


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


class Test(TestCase):
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