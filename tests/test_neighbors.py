import numpy as np
import pytest

from dolcestat.metrics import (
    KNNClassificationAnalyzer,
    KNNRegressionAnalyzer,
    MulticlassAnalyzer,
)
from dolcestat.neighbors import KNNClassifier, KNNRegressor
from dolcestat.neighbors.distances import compute_distances_matrix
from dolcestat.preprocessing import DolceSet
from helpers import make_binary, make_blobs, make_regression, to_sets

sk_neighbors = pytest.importorskip("sklearn.neighbors")
spatial = pytest.importorskip("scipy.spatial.distance")


@pytest.mark.parametrize(
    "metric, kwargs, scipy_args",
    [
        ("euclidean", {}, ("euclidean",)),
        ("manhattan", {}, ("cityblock",)),
        ("minkowski", {"minkowski_p": 3}, ("minkowski", 3)),
    ],
)
def test_distance_matrix_matches_scipy(metric, kwargs, scipy_args):
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(30, 4)), rng.normal(size=(20, 4))
    ours = compute_distances_matrix(A, B, metric, **kwargs)
    expected = spatial.cdist(
        A, B, *scipy_args[:1], **({"p": scipy_args[1]} if len(scipy_args) > 1 else {})
    )
    assert ours.shape == (30, 20) and np.allclose(ours, expected)


def test_distance_matrix_blocks_give_the_same_answer(monkeypatch):
    from dolcestat.neighbors import distances

    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(25, 3)), rng.normal(size=(17, 3))
    whole = compute_distances_matrix(A, B)
    monkeypatch.setattr(distances, "_BLOCK_VALUES", 50)  # a few rows at a time
    assert np.array_equal(distances.compute_distances_matrix(A, B), whole)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_matches_sklearn(weights):
    X, y = make_binary(220)
    train, test = to_sets(X, y, 160)
    ours = KNNClassifier(k=7, weights=weights).fit(train).predict(test)
    ref = sk_neighbors.KNeighborsClassifier(n_neighbors=7, weights=weights).fit(
        train.X, train.y
    )
    assert isinstance(ours, KNNClassificationAnalyzer)
    assert np.array_equal(ours.y_fit, ref.predict(test.X))
    assert np.allclose(ours.proba, ref.predict_proba(test.X))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regressor_matches_sklearn(weights):
    X, y = make_regression(120)
    train, test = to_sets(X, y, 90)
    ours = KNNRegressor(k=5, weights=weights).fit(train).predict(test)
    ref = sk_neighbors.KNeighborsRegressor(n_neighbors=5, weights=weights).fit(
        train.X, train.y
    )
    assert isinstance(ours, KNNRegressionAnalyzer)
    assert np.allclose(ours.y_fit, ref.predict(test.X))
    assert ours.weight_matrix.shape == (30, 5)


@pytest.mark.parametrize("weights", ["distance", "squared_distance"])
def test_a_query_that_equals_a_training_point_is_not_nan(weights):
    # The weight 1/0 used to turn into inf and the prediction into nan
    X, y = make_regression(60)
    train = DolceSet.from_arrays(X, y)
    result = KNNRegressor(k=3, weights=weights).fit(train).predict(train)
    assert np.allclose(result.y_fit, y)
    clf = KNNClassifier(k=3, weights=weights).fit(
        DolceSet.from_arrays(*make_binary(60))
    )
    proba = clf.predict(DolceSet.from_arrays(*make_binary(60))).proba
    assert not np.isnan(proba).any() and np.allclose(proba.sum(axis=1), 1)


def test_gaussian_weights_do_not_depend_on_the_other_queries():
    X, y = make_regression(80)
    train, test = to_sets(X, y, 60)
    knn = KNNRegressor(k=6, weights="gaussian").fit(train)
    together = knn.predict(test).y_fit
    alone = np.array(
        [knn.predict(DolceSet.from_arrays(row[None, :])).y_fit[0] for row in test.X]
    )
    assert np.allclose(together, alone)


def test_settings_live_in_the_constructor_and_fit_returns_the_estimator():
    train, test = to_sets(*make_binary(120), 90)
    with pytest.raises(ValueError, match="k must be an integer"):
        KNNClassifier(train)  # data where the first setting, k, goes
    with pytest.raises(TypeError):
        KNNClassifier(5, "manhattan")  # the other settings are keyword-only
    knn = KNNClassifier(k=3, metric="manhattan")
    assert knn.fit(train) is knn
    assert KNNClassifier(7).k == 7  # k may also be given positionally
    assert len(knn.predict(test).y_fit) == 30


def test_predict_needs_a_fit_and_a_valid_k():
    train, test = to_sets(*make_binary(60), 40)
    with pytest.raises(ValueError, match="not fitted"):
        KNNClassifier().predict(test)
    with pytest.raises(ValueError, match="cannot be greater"):
        KNNClassifier(k=50).fit(train).predict(test)
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="k must be"):
            KNNClassifier(k=bad)


def test_predicting_without_a_target():
    train, test = to_sets(*make_binary(100), 80)
    result = KNNClassifier(k=5).fit(train).predict(DolceSet.from_arrays(test.X))
    assert result.y_true is None and len(result.y_fit) == 20
    assert (
        KNNRegressor()
        .fit(DolceSet.from_arrays(*make_regression(30)))
        .predict(DolceSet.from_arrays(np.zeros((2, 3))))
        .y_true
        is None
    )


def test_multiclass_knn_returns_the_multiclass_analyzer():
    X, y = make_blobs(40)
    train, test = to_sets(X, y, 90)
    result = KNNClassifier(k=7).fit(train).predict(test)
    assert isinstance(result, MulticlassAnalyzer)
    assert result.accuracy() == pytest.approx(np.mean(result.y_fit == test.y))
    assert np.allclose(result.proba.sum(axis=1), 1) and result.proba.shape == (30, 3)
    ref = sk_neighbors.KNeighborsClassifier(n_neighbors=7).fit(train.X, train.y)
    assert np.array_equal(result.y_fit, ref.predict(test.X))
    assert result.auc() > 0.7


def test_knn_class_labels_must_start_at_zero():
    X, y = make_blobs(20)
    with pytest.raises(ValueError, match="encode_target"):
        KNNClassifier().fit(DolceSet.from_arrays(X, y + 1))
    with pytest.raises(ValueError, match="numeric"):
        KNNClassifier().fit(DolceSet.from_arrays(X, np.array(["a", "b", "c"])[y]))


def test_metric_and_weights_are_validated():
    with pytest.raises(ValueError, match="minkowski_p"):
        KNNClassifier(metric="minkowski")
    with pytest.raises(ValueError, match="weights"):
        KNNClassifier(weights="nope")
