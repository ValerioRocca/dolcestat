import numpy as np

from dolcestat.neighbors.distances import compute_distances_matrix


class CentroidsInitializer:
    """Strategies for initializing cluster centroids."""

    @classmethod
    def random(cls, X, k):
        """Selects k random data points from X as initial centroids"""

        # Select randomly k data points from X as initial centroids
        n_samples = X.shape[0]
        random_idxs = np.random.choice(n_samples, size=k, replace=False)
        return X[random_idxs]

    @classmethod
    def kmeans_plus_plus(cls, X, k, metric):
        """Selects k initial centroids using the k-means++ strategy"""

        # 1. Initialize a random centroid
        centroids = cls.random(X, 1)

        # For each centroids from 2nd to k-th...
        for i in range(1, k):

            # 2. Compute the distance matrix between each data point
            #    and the current centroids
            dist_matrix = compute_distances_matrix(X, centroids, metric)

            # 3. For each datapoint, extract the smallest euclidean
            #    distance from any centroid
            min_distances = np.min(dist_matrix, axis=1) ** 2

            # 4. Select the next centroid with probability directly
            #    proportional to the minimum distance. If every remaining
            #    point coincides with an already-chosen centroid, fall back
            #    to a uniform choice to avoid a division by zero.
            total_distance = np.sum(min_distances)
            if total_distance == 0:
                probs = np.full(X.shape[0], 1 / X.shape[0])
            else:
                probs = min_distances / total_distance
            next_centroid_idx = np.random.choice(X.shape[0], p=probs)
            centroids = np.vstack([centroids, X[next_centroid_idx]])

        return centroids
