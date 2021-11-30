import numpy as np
import community as community_louvain
import networkx as nx



def pairwise_accord_naive(category_vector):
    """
    Given an input vector, check naively entries i,j, and construct matrix A, where Aij = 1 iff v(i) = v(j)
    :param category_vector: The input vector
    :return: Accordance matrix A
    """
    n = category_vector.size
    accordance_matrix = np.zeros((n, n))

    for i in range(0, category_vector.size):
        for j in range(i + 1, category_vector.size):
            if category_vector[i] == category_vector[j]:
                accordance_matrix[i, j] = 1
                accordance_matrix[j, i] = 1
    return accordance_matrix


def are_equal_up_to_bijection(v1, v2):
    """
    Given two vectors, check if there exists a function f, such that f(v1) = v2.
    Returns true iff this function exists and is bijective.

    :param v1: First vector
    :param v2: Second vector

    :return: True iff the two vectors are equal up to bijective transformation
    """
    corresp = {}
    equal = v1.size == v2.size
    if equal:
        for i, e in enumerate(v1):
            if (e in corresp and corresp[e] != v2[i]) or (v2[i] in corresp.values() and e not in corresp):
                equal = False
                break
            else:
                corresp[e] = v2[i]
    return equal

def compute_consensus_from_graph(graph, number_partitions, threshold, resolution):
    """
    Applies Louvain algorithm number_partitions times, with provided resolution.
    For each partition, compute the pairwise accordance (ie: if two nodes are grouped in same community or not), and constitute consensus matrix as the sum of these accordance matrices
    The consensus matrix is then normalized by the total number of partitions and thresholded by the provided threshold value.
    
    :param graph: The graph on which to apply the algorithm
    :param number_partitions: Number of different partitions to compute
    :param threshold: Value with which to threshold the consensus matrix, setting all entries below threshold to zero
    :param resolution: Resolution to use in the Louvain algorithm
    
    :return consensus_matrix, the consensus matrix obtained by the procedure
    :return partitions, the number_partitions partitions obtained by the procedure
    """
    n_nodes = len(list(graph.nodes))
    consensus_matrix = np.zeros((n_nodes, n_nodes))
    partitions = np.zeros((number_partitions, n_nodes))
    # First get consensus matrix
    for i in range(0, number_partitions):
        partition = community_louvain.best_partition(graph, weight='weight', resolution=resolution)
        partitions[i, :] =  np.asarray(list(partition.values()))
        consensus_matrix += pairwise_accord_naive(partitions[i, :])
    consensus_matrix /= number_partitions
    # Threshold
    consensus_matrix[consensus_matrix < threshold] = 0.0
    return consensus_matrix, partitions

def consensus_clustering(graph, number_partitions, threshold, n_steps, resolution):
    """
    Overall consensus clustering algorithm described in https://www.nature.com/articles/srep00336#rightslink
    A first consensus matrix is computed on the original graph.
    Then this consensus matrix is treated as an adjacency matrix itself, on which we apply the consensus procedure until either
    convergence (meaning all partitions are the same) or maximum number of steps are reached.

    :param graph: The original graph on which to perform consensus clustering
    :param number_partitions: Number of partitions at each iteration of the clustering algorithm
    :param threshold: The threshold used in consensus matrix computation
    :param n_steps: The maximum number of steps before termination of the algorithm, should it fail to reach convergence
    :param resolution: The resolution for Louvain's algorithm

    :return partitions: The last number_partitions derived by the algorithm. If algorithm is converged, they are all equal up to bijection.
    :return i: The iteration at which the algorithm finished. Useful to assess how quickly it converged or if it even converged at all.
    """
    consensus_matrix, partitions = compute_consensus_from_graph(graph, number_partitions, threshold, resolution)

    # Until convergence or number of steps exceeded
    for i in range(0, n_steps):
        # Convert consensus matrix to graph
        G = nx.convert_matrix.from_numpy_matrix(consensus_matrix)
        # Get new consensus matrix
        consensus_matrix, partitions = compute_consensus_from_graph(G, number_partitions, threshold, resolution)

        # Now we must ask if all category vectors are the same or not
        should_stop = True
        for vi in range(0, number_partitions - 1):
            should_stop = are_equal_up_to_bijection(partitions[vi], partitions[vi + 1])
            if not should_stop:
                break
        if should_stop:
            break
    return partitions, i