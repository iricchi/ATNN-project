import numpy as np
import networkx as nx
from functools import reduce
from scipy.stats import rankdata


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

    Self loops are counted twice in degree computation.

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
        std = degrees.std()
        if std > 10e-8:
            degrees = (degrees - degrees.mean()) / degrees.std()
        else:
            degrees = (degrees - degrees.mean())
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
                    # This handles the self looping case, which counts as twice in the degrees
                    factor= 2.0 if n_i == neighbour else 1.0
                    if weighted:
                        degrees[n_i, c_i] += (neighbours[neighbour]['weight']*factor)
                    else:
                        degrees[n_i, c_i] += 1*factor
    total_degrees= degrees.sum(axis=1)
    disconnected_nodes = total_degrees == 0
    total_degrees[disconnected_nodes] = 1
    norm_factor = 1. / total_degrees
    s = (norm_factor.reshape((-1, 1)) * degrees) ** 2
    # Places where the degrees where zero will be mapped to zero
    pc = 1. - s.sum(axis=1)
    pc[disconnected_nodes] = 0
    return pc

def compute_system_segregation(G, partition_values):
    """
    System segregation is defined as:
    (mean(connections within same community) - mean(connections not part of same community))/mean(connection within same community)
    """
    within_net_count = 0
    within_net_strength = 0
    between_net_count = 0
    between_net_strength = 0

    for e in list(G.edges):
        n1, n2 = e[0], e[1]
        s = G[n1][n2]['weight']
        if n1 == n2:
            within_net_count += 2
            within_net_strength += 2*s
        else:
            if partition_values[n1] == partition_values[n2]:
                within_net_count += 1
                within_net_strength += s
            else:
                between_net_count += 1
                between_net_strength += s
    between_net_strength /= between_net_count
    within_net_strength /= within_net_count
    return (within_net_strength - between_net_strength) / within_net_strength


def compute_within_degree(modules):
    # First, compute the within degree for each node (standardized and weighted)
    module_degrees = [compute_degree(module, standardize=True, weighted=True) for module in modules]

    # Then, flatten the results as a single array
    nodes_tpm, within_degrees_tmp = zip(*module_degrees)
    nodes = []
    for e in nodes_tpm:
        if len(e)>1:
            nodes.extend(e)
        else:
            nodes.append(e[0])
    within_degrees = reduce(lambda x, y: np.hstack((x, y)), within_degrees_tmp)

    # Simply put back the degrees in order of nodes, so that within_degree[i] corresponds to ith node in the graph
    within_degrees = within_degrees[np.argsort(nodes)]

    return within_degrees


def get_ranking(v):
    ordered_indices = np.argsort(np.argsort(v))
    c=np.bincount(v)
    return (np.cumsum(np.concatenate(([0.0], c))))[ordered_indices]


def compute_hub_score(integration_node_list, segregation_node_list):
    # high integration should be first rank (0) so we need reverse order
    integration_ranking = rankdata(-integration_node_list, method='min') - 1
    # low segregation should be first rank (0) so this order is correct
    segregation_ranking = rankdata(segregation_node_list, method='min') - 1

    rank_val = integration_ranking + segregation_ranking
    # Now scale the score between 0 (highest rank) and 1 (lowest rank):
    n_nodes = integration_node_list.shape[0]
    max_rank = (n_nodes-1) * 2
    rank_val = (max_rank - rank_val) / max_rank

    # Finally, we want the highest value to be highest rank and lowest value to be lowest rank
    return rank_val