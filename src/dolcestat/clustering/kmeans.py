import numpy as np

from dolcestat.clustering.centroids import CentroidsInitializer
from dolcestat.clustering.input_validation import (
    validate_flavor,
    validate_init_strategy,
    validate_max_iters,
)
from dolcestat.neighbors.distances import compute_distances_matrix
from dolcestat.neighbors.input_validation import validate_k
from dolcestat.preprocessing import DolceSet


class KMeans:

    def __init__(self):
        pass

    def _compute_silhouette_score(self, X, labels):

        clusters = np.unique(labels)

        # Silhouette is undefined for a single cluster.
        if clusters.size < 2:
            return np.nan

        # 1. Compute distance between every pair of points
        dist_matrix = compute_distances_matrix(X, X, metric="euclidean")

        s = []

        for idx, obs_distances in enumerate(dist_matrix):

            own_label = labels[idx]

            # 2. Strip out the point itself from both the distances and the
            #    labels.
            other_distances = np.delete(obs_distances, idx)
            other_labels = np.delete(labels, idx)

            # 3. Mean distance from this point to every cluster
            mean_distances = {
                cluster: other_distances[other_labels == cluster].mean()
                for cluster in clusters
                if np.any(other_labels == cluster)
            }

            # 4. a = mean distance to the other points of its own cluster.
            #    If it is the only point in the cluster, set it to 0.
            if own_label not in mean_distances:
                s.append(0.0)
                continue
            a = mean_distances[own_label]

            # 5. b = mean distance to the nearest *other* cluster
            b = min(
                mean for cluster, mean in mean_distances.items() if cluster != own_label
            )

            # 6. Silhouette for this point
            s.append((b - a) / max(a, b))

        return np.mean(s)

    def fit(
        self,
        data: DolceSet,
        k: int,
        init_strategy: str = "kmeans++",
        flavor: str = "kmeans",
        max_iters: int = 100,
        track_history: bool = False,
    ):

        if track_history:
            self.centroids_history = []
            self.assignments_history = []

        X = data.X
        validate_k(k)
        validate_init_strategy(init_strategy)
        validate_flavor(flavor)
        validate_max_iters(max_iters)

        # 1. Distinguish K-means and K-medians cases
        if flavor == "kmeans":
            metric = "euclidean"
        elif flavor == "kmedians":
            metric = "manhattan"

        # 2. Initialize centroids based on strategy
        if init_strategy == "random":
            centroids = CentroidsInitializer().random(X, k)
        elif init_strategy == "kmeans++":
            centroids = CentroidsInitializer().kmeans_plus_plus(X, k, metric)

        for i in range(max_iters):

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
            elif flavor == "kmedians":
                new_centroids = np.array(
                    [np.median(X[min_dist_idxs == j], axis=0) for j in range(k)]
                )

            # 5. Save centroids and assignments history if requested
            if track_history:
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
        self.silhouette = self._compute_silhouette_score(X, min_dist_idxs)
