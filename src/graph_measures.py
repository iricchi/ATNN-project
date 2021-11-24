import numpy as np
import networkx as nx
from functools import reduce


def create_communities_from_partition(G, partition):
    tmp_G = G.copy()
    if len(list(G.nodes)) != partition.size:
        raise AssertionError("Partition should have exactly one value per node of input graph.")
    unique= np.unique(partition)
    for i in range(0, partition.size):
        tmp_G.add_nodes_from([i], partition=partition[i])
    subgraphs = [tmp_G.subgraph((node for node, data in tmp_G.nodes(data=True) if data.get("partition") == e)) for i, e in enumerate(unique)]
    if not nx.community.is_partition(tmp_G, subgraphs):
        raise AssertionError('The provided partition is not a proper partition for this graph')
    return subgraphs


def compute_degree(G, standardize=True, weighted=True):
    """
    Function to compute degree of all nodes within the graph of interest.
    If weighted degree is requested, the degree will then be defined as the sum of weighted edges, for each node.
    If z-scoring is requested, the function will standardize the returned degrees.

    :param G: input graph. Degree will be computed for each node within this graph
    :param standardize: Whether degrees should be standardized before being returned. (Default = True)
    :param weighted: Computes weighted degree instead of regular degree. If the graph is unweighted, this parameter is ignored. (Default = True)

    :return nodes: Nodes of the graph (for debug purposes)
    :return degrees: Degrees computed according to the chosen method.
    """
    if weighted and nx.is_weighted(G):
        nodes = list(G.nodes)
        degrees = [0] * len(nodes)
        i = 0
        for n in nodes:
            for neighbor in G[n]:
                degrees[i] += G[n][neighbor]['weight']
                # A self loop means we must add the degree a second time
                if neighbor == n:
                    degrees[i] += G[neighbor][n]['weight']
            i += 1
    else:
        nodes, degrees = zip(*list(G.degree))

    degrees = np.asarray(degrees)
    if standardize:
        degrees = (degrees - degrees.mean()) / degrees.std()
    return nodes, degrees


def compute_participation_coefficient(G, weighted, partition_values):
    """
    Computes participation coefficient as 1 - sum((k_is / k_i)**2), where k_is is the degree of node i in community s
    and k_i is the degree of node i in the graph (thus, sum of k_is).

    :param G: graph on which to compute participation coefficient node-wise.
    :param weighted: Whether to compute weighted degree or not.
    :param partition_values: Vector indicating which community which node belongs to.

    :return participation coefficient vector of shape n_nodes x 1
    """
    nodes = list(G.nodes)
    n_nodes = len(nodes)
    partition_categories = np.unique(partition_values)
    degrees = np.zeros((n_nodes, partition_categories.size))
    for n_i in range(0, n_nodes):
        for c_i, u in enumerate(partition_categories):
            neighbours = G[n_i]
            for neighbour in neighbours:
                if partition_values[neighbour] == u:
                    if weighted:
                        degrees[n_i, c_i] += neighbours[neighbour]['weight']
                    else:
                        degrees[n_i, c_i] += 1
    norm_factor = 1. / degrees.sum(axis=1)
    s = (norm_factor.reshape((-1, 1)) * degrees) ** 2
    return 1 - s.sum(axis=1)
    # return degrees


def compute_system_segregation(G, partition_values):
    """
    System segregation is defined as:
    (mean(connections within same community) - mean(connections not part of same community))/mean(connection within same community)
    """
    # First, compute all communities
    subgraphs = create_communities_from_partition(G, partition_values)

    # Next for all communities, compute the within weighted degree values
    degree_values = [compute_degree(subgraph, standardize=False, weighted=True)[1] for subgraph in subgraphs]

    # Now we must fuse all these together
    degree_values = reduce(lambda x, y: np.hstack((x, y)), degree_values)

    mean_same_community_weight = degree_values.mean()

    # Now, we can worry about node pairs, ie: edges!
    edges = list(G.edges)
    count = 0
    mean_diff_community_weight = 0
    for e in edges:
        n1, n2 = e[0], e[1]
        if partition_values[n1] != partition_values[n2]:
            count += 1
            mean_diff_community_weight += G[n1][n2]['weight']
    mean_diff_community_weight /= count
    # return mean_same_community_weight
    return (mean_same_community_weight - mean_diff_community_weight) / mean_same_community_weight