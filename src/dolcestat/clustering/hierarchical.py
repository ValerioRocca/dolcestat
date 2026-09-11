import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dolcestat.clustering.input_validation import validate_linkage
from dolcestat.clustering.linkage import apply_lance_williams_formula
from dolcestat.neighbors.distances import compute_distances_matrix
from dolcestat.preprocessing import DolceSet


class Cluster:
    """A node of the dendrogram: either a single point or a merge of two clusters.

    Leaves are created one per observation with height 0. Every later merge
    creates a new Cluster whose two children are flagged ``has_parent=True``,
    so the still-mergeable clusters are exactly those with ``has_parent=False``.

    Parameters
    ----------
    members_ids : list
        IDs of the observations belonging to the cluster. A leaf holds a
        single-element list with the index of its observation; a merged
        cluster holds the members of the two clusters it was built from.
    dist_matrix_idx : int or None
        Row/column of the current distance matrix representing this cluster.
        Because rows are deleted as clusters merge, this index is rewritten at
        every iteration, and set to None once the cluster acquires a parent.
    height : float
        Distance between the two clusters that were merged to create this one,
        i.e. the height at which it appears in the dendrogram. Leaves sit at 0.

    Attributes
    ----------
    has_parent : bool
        Whether the cluster has already been merged into a larger one. Merged
        clusters keep their place in the list but are skipped by later merges.
    """

    def __init__(self, members_ids, dist_matrix_idx, height):
        self.has_parent = False
        self.members_ids = members_ids
        self.dist_matrix_idx = dist_matrix_idx
        self.height = height


class HierarchicalClustering:
    """Agglomerative (bottom-up) hierarchical clustering.

    Every observation starts in its own cluster; at each step the two closest
    clusters are merged, until a single cluster containing all the observations
    is left. The sequence of merges and the distance at which each happened
    form the dendrogram, which can be cut at any height to obtain a flat
    partition -- unlike KMeans, the number of clusters need not be chosen in
    advance.
    """

    def __init__(self):
        self.is_fitted = False

    def fit(
        self,
        data: DolceSet,
        linkage: str = "ward",
        track_history: bool = False,
    ):
        """Build the dendrogram by repeatedly merging the two closest clusters.

        Parameters
        ----------
        data : DolceSet
            Dataset to cluster. Only ``data.X`` is used; the target, if any,
            is ignored.
        linkage : str, default "ward"
            How the distance between two clusters is derived from the
            distances between their members. One of "single" (closest pair),
            "complete" (farthest pair), "average" (mean over all pairs) or
            "ward" (merge that increases the within-cluster variance least).
        track_history : bool, default False
            Whether to record the intermediate state of each merge.

        Returns
        -------
        list of Cluster
            The nodes of the dendrogram, leaves first and in merge order, so
            the last element is the root spanning every observation.
        """

        # WHAT SHOULD HAPPEN IF TWO POINTS ARE EQUAL?
        # RENAME DISTANCE MATRIX TO LINKAGE MATRIX? (and comments accordingly)

        X = data.X
        clusters = []
        validate_linkage(linkage)

        # 1. Compute distance matrix
        dist_matrix = compute_distances_matrix(X, X, metric="euclidean")
        np.fill_diagonal(dist_matrix, np.inf)  # add inf to avoid self-merging

        # 2. Assign each point to its own cluster
        clusters += [Cluster([i], i, 0.0) for i in range(dist_matrix.shape[0])]

        # Until there is only one cluster left...
        for _ in range(X.shape[0] - 1):

            # 3. Find the two clusters with the smallest distance
            dist_matrix_idxs_to_be_merged = np.array(
                np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            )

            # 4. Create new cluster's distance row/column via Lance-Williams
            #    formula
            # 4a. Extract linkages
            new_cluster_linkage_vector = []
            linkage_ij = dist_matrix[
                dist_matrix_idxs_to_be_merged[0], dist_matrix_idxs_to_be_merged[1]
            ]
            new_cluster_height = linkage_ij
            # 4b. Extract list of left and right members to be merged
            members_ids_to_be_merged = [
                x.members_ids
                for i in [0, 1]
                for x in clusters
                if x.has_parent == False
                and x.dist_matrix_idx == dist_matrix_idxs_to_be_merged[i]
            ]
            # 4c. Extract length for other clusters (excluding the two being
            #     merged)
            other_clusters_length = [
                len(x.members_ids)
                for x in clusters
                if x.has_parent == False
                and x.dist_matrix_idx not in dist_matrix_idxs_to_be_merged
            ]
            other_dist_columns = np.delete(
                dist_matrix[dist_matrix_idxs_to_be_merged, :],
                dist_matrix_idxs_to_be_merged,
                axis=1,
            )
            for (linkage_ik, linkage_jk), size_k in zip(
                other_dist_columns.T, other_clusters_length
            ):
                linkage_ijk = apply_lance_williams_formula(
                    dist_i=linkage_ik,
                    dist_j=linkage_jk,
                    dist_ij=linkage_ij,
                    linkage=linkage,
                    size_i=len(members_ids_to_be_merged[0]),
                    size_j=len(members_ids_to_be_merged[1]),
                    size_k=size_k,
                )
                new_cluster_linkage_vector.append(linkage_ijk)

            # 5. Update the distance matrix
            # 5a. Remove rows and columns corresponding to the merged units
            dist_matrix = np.delete(dist_matrix, dist_matrix_idxs_to_be_merged, axis=0)
            dist_matrix = np.delete(dist_matrix, dist_matrix_idxs_to_be_merged, axis=1)
            # 5b. Append the new cluster's row and column, with inf on its
            #     own diagonal entry to avoid self-merging
            dist_matrix = np.vstack([dist_matrix, new_cluster_linkage_vector])
            new_column = np.append(new_cluster_linkage_vector, np.inf)
            dist_matrix = np.column_stack([dist_matrix, new_column])

            # 5. Updates the other clusters
            for cluster in clusters:

                # Clusters already merged are not touched
                if cluster.has_parent == True:
                    continue

                # Clusters not merged and not to be merged: update their dist_matrix_idx
                # after deletion at step 4a.
                if cluster.dist_matrix_idx not in dist_matrix_idxs_to_be_merged:
                    cluster.dist_matrix_idx -= np.sum(
                        dist_matrix_idxs_to_be_merged < cluster.dist_matrix_idx
                    )

                # Clusters to be merged: update has_parent and dist_matrix_idx
                # parameters
                else:
                    cluster.has_parent = True
                    cluster.dist_matrix_idx = None

            # 6. Create and append the new cluster
            new_cluster = Cluster(
                members_ids=members_ids_to_be_merged[0] + members_ids_to_be_merged[1],
                dist_matrix_idx=dist_matrix.shape[0] - 1,
                height=new_cluster_height,
            )
            clusters.append(new_cluster)

        self.clusters = clusters
        self.is_fitted = True

    def plot_dendrogram(self, ax=None):
        """Draw the dendrogram with seaborn/matplotlib.

        Leaves are placed left to right in the order they first appear as a
        member of a merge (equivalent to scipy's default leaf ordering), and
        every merge is drawn as a bracket connecting its two children at the
        midpoint of their x-positions, rising to the merge's height.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if not given.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the dendrogram was drawn on.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before plot_dendrogram().")

        clusters = self.clusters
        n_leaves = sum(1 for c in clusters if len(c.members_ids) == 1)

        # 1. Assign each leaf an x-position by first appearance in the merge
        #    order, then work out every cluster's x as the midpoint of its
        #    two children (leaves have no children and keep their own x).
        leaf_order = [c.members_ids[0] for c in clusters[:n_leaves]]
        x_of_member = {member_id: x for x, member_id in enumerate(leaf_order)}

        # Each merge in clusters[n_leaves:] consumes exactly the two most
        # recent still-unconsumed clusters that its members_ids contains, in
        # the same order the fit() loop built them -- so replaying merges
        # against a "roots so far" pool recovers each cluster's children in
        # O(1) amortized, no subset search needed.
        roots_by_member = {
            cluster.members_ids[0]: cluster for cluster in clusters[:n_leaves]
        }

        x_position = {id(c): x_of_member[c.members_ids[0]] for c in clusters[:n_leaves]}
        segments = []
        for cluster in clusters[n_leaves:]:
            left = roots_by_member[cluster.members_ids[0]]
            right = roots_by_member[cluster.members_ids[-1]]

            x_left, x_right = x_position[id(left)], x_position[id(right)]
            x_mid = (x_left + x_right) / 2
            x_position[id(cluster)] = x_mid

            segments.append(([x_left, x_left], [left.height, cluster.height]))
            segments.append(([x_right, x_right], [right.height, cluster.height]))
            segments.append(([x_left, x_right], [cluster.height, cluster.height]))

            for member_id in cluster.members_ids:
                roots_by_member[member_id] = cluster

        # 2. Draw
        if ax is None:
            with sns.axes_style("whitegrid"):
                _, ax = plt.subplots()

        for x, y in segments:
            ax.plot(x, y, color="steelblue")

        ax.set_xticks(list(x_of_member.values()))
        ax.set_xticklabels(list(x_of_member.keys()))
        ax.set_xlabel("Observation")
        ax.set_ylabel("Distance")
        sns.despine(ax=ax)

        return ax
