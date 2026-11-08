import matplotlib.pyplot as plt
import numpy as np
import pytest

from dolcestat.clustering import HierarchicalClustering, KMeans
from dolcestat.preprocessing import DolceSet

sk_cluster = pytest.importorskip("sklearn.cluster")
sk_metrics = pytest.importorskip("sklearn.metrics")
hierarchy = pytest.importorskip("scipy.cluster.hierarchy")


def _blobs(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [c + rng.normal(0, 0.5, (n_per, 2)) for c in ([0, 0], [5, 5], [0, 5])]
    )
    return DolceSet.from_arrays(X)


def test_kmeans_matches_sklearn_on_separated_blobs():
    data = _blobs()
    km = KMeans(3).fit(data)
    ref = sk_cluster.KMeans(n_clusters=3, n_init=10, random_state=0).fit(data.X)
    assert km.wcss == pytest.approx(ref.inertia_)
    assert km.silhouette == pytest.approx(
        sk_metrics.silhouette_score(data.X, km.labels)
    )
    assert sorted(map(tuple, np.round(km.centroids, 6))) == sorted(
        map(tuple, np.round(ref.cluster_centers_, 6))
    )


def test_kmeans_settings_and_fit_return():
    with pytest.raises(TypeError):
        KMeans()  # k is required
    km = KMeans(3, init_strategy="random", max_iters=50)
    assert km.fit(_blobs()) is km and km.is_fitted and 1 <= km.n_iters <= 50
    assert km.labels.shape == (120,) and km.centroids.shape == (3, 2)


def test_kmeans_history_is_recorded_only_on_request_and_reset_on_refit():
    data = _blobs()
    km = KMeans(3, track_history=True).fit(data)
    assert len(km.centroids_history) == km.n_iters == len(km.assignments_history)
    assert km.centroids_history[0].shape == (3, 2)
    assert KMeans(3).fit(data).centroids_history is None


def test_kmedians_resists_outliers():
    rng = np.random.default_rng(3)
    core = np.vstack(
        [c + rng.normal(0, 0.5, (60, 2)) for c in ([0.0, 0.0], [10.0, 0.0])]
    )
    outliers = np.column_stack([rng.uniform(22, 30, 6), rng.uniform(-1, 1, 6)])
    data = DolceSet.from_arrays(np.vstack([core, outliers]))
    np.random.seed(2)
    means = KMeans(2).fit(data).centroids
    np.random.seed(2)
    medians = KMeans(2, flavor="kmedians").fit(data).centroids

    def drift(cents):
        return np.min(np.linalg.norm(cents - np.array([10.0, 0.0]), axis=1))

    assert drift(medians) < drift(means)


def test_silhouette_is_lazy_nan_for_one_cluster_and_zero_for_singletons():
    data = _blobs()
    km = KMeans(3).fit(data)
    assert km._silhouette is None  # not computed by fit
    assert not np.isnan(km.silhouette) and km._silhouette is not None
    assert np.isnan(KMeans(1).fit(data).silhouette)
    with pytest.raises(ValueError, match="fit"):
        KMeans(2).silhouette
    # a point alone in its cluster scores 0, as in scikit-learn
    X = np.array([[0.0], [0.1], [0.2], [10.0]])
    labels = np.array([0, 0, 0, 1])
    ours = KMeans(2)._compute_silhouette_score(X, labels)
    assert ours == pytest.approx(sk_metrics.silhouette_score(X, labels))


def test_kmeans_validates_its_input():
    with pytest.raises(ValueError, match="init_strategy"):
        KMeans(2, init_strategy="x")
    with pytest.raises(ValueError, match="max_iters"):
        KMeans(2, max_iters=0)
    with pytest.raises(ValueError, match="k must be"):
        KMeans(0)
    with pytest.raises(ValueError, match="greater than the number of rows"):
        KMeans(5).fit(DolceSet.from_arrays(np.zeros((3, 2))))


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_hierarchical_heights_match_scipy(linkage):
    data = _blobs(n_per=12)
    model = HierarchicalClustering(linkage=linkage).fit(data)
    ours = np.sort([c.height for c in model.clusters[len(data.X) :]])
    theirs = np.sort(hierarchy.linkage(data.X, method=linkage)[:, 2])
    assert np.allclose(ours, theirs)


def test_hierarchical_fit_returns_the_model_and_has_no_dead_options():
    model = HierarchicalClustering("single")
    assert model.fit(_blobs(n_per=5)) is model and model.is_fitted
    assert len(model.clusters) == 2 * 15 - 1
    with pytest.raises(TypeError):
        model.fit(_blobs(n_per=5), "single")
    with pytest.raises(ValueError, match="linkage"):
        HierarchicalClustering("nope")
    with pytest.raises(RuntimeError, match="fit"):
        HierarchicalClustering().plot_dendrogram()


@pytest.mark.parametrize("linkage", ["single", "ward"])
def test_dendrogram_leaves_are_ordered_so_that_no_branch_crosses(linkage):
    rng = np.random.default_rng(5)
    data = DolceSet.from_arrays(rng.normal(size=(30, 2)))
    model = HierarchicalClustering(linkage).fit(data)
    fig, ax = plt.subplots()
    model.plot_dendrogram(ax=ax)
    order = [int(t.get_text()) for t in ax.get_xticklabels()]
    assert sorted(order) == list(range(30))
    position = {leaf: i for i, leaf in enumerate(order)}
    for cluster in model.clusters[30:]:
        spots = sorted(position[m] for m in cluster.members_ids)
        assert spots[-1] - spots[0] + 1 == len(spots)  # each cluster is contiguous
    plt.close(fig)
