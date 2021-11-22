import numpy as np


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

