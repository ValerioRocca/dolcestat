"""K-means and k-medians clustering."""

import numpy as np

from dolcestat.clustering.centroids import CentroidsInitializer
from dolcestat.clustering.input_validation import (
    validate_flavor,
    validate_init_strategy,
    validate_max_iters,
    validate_X,
)
from dolcestat.neighbors.distances import compute_distances_matrix
from dolcestat.neighbors.input_validation import validate_k


class KMeans:
    """K-means (or k-medians) clustering.

    Built from its settings alone; ``fit(data)`` clusters ``data.X`` (the
    target, if any, is ignored) and returns the model. After fitting:
    ``centroids``, ``labels`` (cluster index of each row), ``n_iters``,
    ``wcss`` and ``silhouette``.

    Parameters
    ----------
    k : int
        Number of clusters.
    init_strategy : {"kmeans++", "random"}, default "kmeans++"
        How the first centroids are chosen. Both draw from NumPy's global
        random generator, so seed it (``np.random.seed``) to reproduce a fit.
    flavor : {"kmeans", "kmedians"}, default "kmeans"
        "kmeans" assigns by Euclidean distance and moves each centroid to the
        mean of its points; "kmedians" uses Manhattan distance and the
        coordinate-wise median, which resists outliers.
    max_iters : int, default 100
    track_history : bool, default False
        Record the centroids and assignments of every iteration, in
        ``centroids_history`` and ``assignments_history``.
    """

    def __init__(
        self,
        k,
        *,
        init_strategy="kmeans++",
        flavor="kmeans",
        max_iters=100,
        track_history=False,
    ):
        validate_k(k)
        validate_init_strategy(init_strategy)
        validate_flavor(flavor)
        validate_max_iters(max_iters)
        self.k = k
        self.init_strategy = init_strategy
        self.flavor = flavor
        self.max_iters = max_iters
        self.track_history = track_history

        self.is_fitted = False
        self.centroids = None
        self.labels = None
        self.n_iters = None
        self.wcss = None
        self.centroids_history = None
        self.assignments_history = None
        self._X = None
        self._silhouette = None

    @property
    def silhouette(self):
        """Mean silhouette score of the fitted clustering (computed on first use).

        For each point, ``a`` is its mean distance to the other points of its
        cluster and ``b`` its mean distance to the nearest other cluster; the
        score is ``(b - a) / max(a, b)``. NaN when there is a single cluster.
        """
        if not self.is_fitted:
            raise ValueError("Cannot compute the silhouette: call fit(data) first.")
        if self._silhouette is None:
            self._silhouette = self._compute_silhouette_score(self._X, self.labels)
        return self._silhouette

    def _compute_silhouette_score(self, X, labels):

        clusters = np.unique(labels)

        # Silhouette is undefined for a single cluster.
        if clusters.size < 2:
            return np.nan

        # 1. Compute distance between every pair of points
        dist_matrix = compute_distances_matrix(X, X, metric="euclidean")
        n = len(labels)

        # 2. Total distance from each point to each cluster. A point's distance
        #    to itself is 0, so it does not disturb its own cluster's total.
        one_hot = (labels[:, np.newaxis] == clusters[np.newaxis, :]).astype(float)
        totals = dist_matrix @ one_hot
        sizes = one_hot.sum(axis=0)
        own = np.searchsorted(clusters, labels)
        rows = np.arange(n)

        # 3. a = mean distance to the other points of its own cluster
        a = totals[rows, own] / np.maximum(sizes[own] - 1, 1)

        # 4. b = mean distance to the nearest *other* cluster
        mean_distances = totals / sizes
        mean_distances[rows, own] = np.inf
        b = mean_distances.min(axis=1)

        # 5. Silhouette per point; a point alone in its cluster scores 0
        denominator = np.maximum(a, b)
        s = np.divide(b - a, denominator, out=np.zeros(n), where=denominator > 0)
        s[sizes[own] == 1] = 0.0

        return float(np.mean(s))

    def fit(self, data):
        """Clusters ``data.X`` and returns the fitted model."""

        X = data.X
        validate_X(X)
        if self.k > X.shape[0]:
            raise ValueError(
                f"k ({self.k}) cannot be greater than the number of rows ({X.shape[0]})."
            )

        if self.track_history:
            self.centroids_history = []
            self.assignments_history = []
        else:
            self.centroids_history = None
            self.assignments_history = None

        k = self.k
        flavor = self.flavor

        # 1. Distinguish K-means and K-medians cases
        if flavor == "kmeans":
            metric = "euclidean"
        else:
            metric = "manhattan"

        # 2. Initialize centroids based on strategy
        if self.init_strategy == "random":
            centroids = CentroidsInitializer().random(X, k)
        else:
            centroids = CentroidsInitializer().kmeans_plus_plus(X, k, metric)

        for i in range(self.max_iters):

            # 2. Assign each data point to the nearest centroid
            dist_matrix = compute_distances_matrix(X, centroids, metric)
            min_dist_idxs = dist_matrix.argmin(axis=1)

            # 3. If a cluster is empty, raise error
            if np.unique(min_dist_idxs).size < k:
                raise ValueError(f"At iteration {i}, one or more clusters are empty.")

            # 4. Compute new centroids as the mean/median of the assigned data points
            if flavor == "kmeans":
                new_centroids = np.array(
                    [X[min_dist_idxs == j].mean(axis=0) for j in range(k)]
                )
            else:
                new_centroids = np.array(
                    [np.median(X[min_dist_idxs == j], axis=0) for j in range(k)]
                )

            # 5. Save centroids and assignments history if requested
            if self.track_history:
                self.centroids_history.append(new_centroids.copy())
                self.assignments_history.append(min_dist_idxs.copy())

            # 6. If centroids do not change, break the loop
            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break

            centroids = new_centroids

        # Recompute the assignment against the final centroids so that WCSS,
        # the labels, and self.centroids are all mutually consistent even when
        # the loop stops at max_iters without fully converging.
        dist_matrix = compute_distances_matrix(X, new_centroids, metric)
        min_dist_idxs = dist_matrix.argmin(axis=1)

        self.n_iters = i + 1
        self.centroids = new_centroids
        self.labels = min_dist_idxs
        # WCSS is the within-cluster sum of *squared* distances to the centroid
        # (matches sklearn's inertia_). For kmedians the metric is Manhattan, so
        # this is the sum of squared L1 distances.
        self.wcss = np.sum(np.min(dist_matrix, axis=1) ** 2)
        self._X = X
        self._silhouette = None
        self.is_fitted = True
        return self
