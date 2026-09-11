import numpy as np

from dolcestat.clustering.input_validation import validate_linkage


def lance_williams_coefficients(linkage, size_i, size_j, size_k):
    """Returns the Lance-Williams coefficients for the requested linkage.

    The four coefficients (alpha_i, alpha_j, beta, gamma) are the ones plugged
    into the recurrence below. For "single", "complete" and "average" they do
    not depend on the cluster being measured against, so they come back as
    scalars; for "ward" they do too, but their value depends on ``size_k``.

    Parameters
    ----------
    linkage : str
        One of "single", "complete", "average" or "ward".
    size_i, size_j : int
        Number of observations in each of the two clusters being merged.
    size_k : int
        Number of observations in the other cluster k, i.e. the cluster the
        merged one is being measured against.

    Returns
    -------
    tuple
        (alpha_i, alpha_j, beta, gamma), each a float.
    """
    match linkage:
        case "single":
            # min(d_ik, d_jk), written as the average minus half the spread
            return 0.5, 0.5, 0.0, -0.5
        case "complete":
            # max(d_ik, d_jk), same trick with the sign of gamma flipped
            return 0.5, 0.5, 0.0, 0.5
        case "average":
            # Mean over all pairs, so each side weighs as much as its size
            size_ij = size_i + size_j
            return size_i / size_ij, size_j / size_ij, 0.0, 0.0
        case "ward":
            # Increase in within-cluster variance: the sizes of all three
            # clusters take part
            total = size_i + size_j + size_k
            return (
                (size_i + size_k) / total,
                (size_j + size_k) / total,
                -size_k / total,
                0.0,
            )


def apply_lance_williams_formula(
    dist_i, dist_j, dist_ij, linkage, size_i, size_j, size_k
):
    """Returns the distance from a freshly merged cluster to one other cluster.

    The Lance-Williams recurrence gives the distance between the union of two
    clusters i and j and any other cluster k out of the distances that were
    already known, so the original observations never have to be revisited:

        d(i+j, k) = alpha_i * d(i,k) + alpha_j * d(j,k)
                    + beta * d(i,j) + gamma * |d(i,k) - d(j,k)|

    Every linkage in this module is a different choice of the four
    coefficients, so a single line covers all of them -- except "ward", whose
    coefficients recombine *squared* distances (it is defined in terms of
    within-cluster variance), so the inputs are squared before applying the
    recurrence and the result is square-rooted back on the way out.

    Parameters
    ----------
    dist_i, dist_j : float
        Distance from cluster i (respectively j) to the other cluster k.
    dist_ij : float
        Distance between the two clusters being merged, i.e. the height at
        which the merge happens.
    linkage : str
        One of "single", "complete", "average" or "ward".
    size_i, size_j : int
        Number of observations in each of the two clusters being merged.
    size_k : int
        Number of observations in the other cluster k. Ignored by every
        linkage but "ward".

    Returns
    -------
    float
        Distance from the merged cluster to cluster k.
    """
    validate_linkage(linkage)

    dist_i = float(dist_i)
    dist_j = float(dist_j)

    # Ward's recurrence is only linear in squared distances: it tracks the
    # increase in within-cluster variance, which is a sum of squares.
    if linkage == "ward":
        dist_i, dist_j, dist_ij = dist_i**2, dist_j**2, dist_ij**2

    alpha_i, alpha_j, beta, gamma = lance_williams_coefficients(
        linkage, size_i, size_j, size_k
    )

    merged = (
        alpha_i * dist_i
        + alpha_j * dist_j
        + beta * dist_ij
        + gamma * abs(dist_i - dist_j)
    )

    if linkage == "ward":
        merged = merged**0.5

    return merged
