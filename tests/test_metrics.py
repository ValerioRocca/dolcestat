import numpy as np
import pytest

from dolcestat.metrics import (
    ClassificationAnalyzer,
    GaussianRegressionAnalyzer,
    KNNClassificationAnalyzer,
    MulticlassAnalyzer,
    RegressionAnalyzer,
    classification_analyzer,
)

sk_metrics = pytest.importorskip("sklearn.metrics")


# --- binary classification ----------------------------------------------------


def _scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, 0.5 * rng.random(n) + 0.5 * y * rng.random(n)


def test_auc_matches_sklearn():
    y, score = _scores()
    assert np.isclose(
        ClassificationAnalyzer(y, score).auc(), sk_metrics.roc_auc_score(y, score)
    )


def test_auc_counts_ties_and_hard_labels():
    # Hard 0/1 predictions are all ties within each label: sklearn says 0.5 here
    y = np.array([0, 0, 1, 1])
    hard = np.array([1.0, 0.0, 1.0, 0.0])
    assert ClassificationAnalyzer(y, hard).auc() == pytest.approx(0.5)


def test_auc_of_a_very_confident_model():
    # Probabilities packed against 1.0: a fixed grid of thresholds would miss them
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    score = np.clip(0.9995 + 0.0004 * (y + rng.normal(0, 0.5, 500)), 0, 1)
    assert np.isclose(
        ClassificationAnalyzer(y, score).auc(), sk_metrics.roc_auc_score(y, score)
    )


def test_auc_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        ClassificationAnalyzer(np.zeros(5), np.linspace(0, 1, 5)).auc()


def test_auc_runs_on_a_numpy_without_trapz():
    # np.trapz was removed in NumPy 2.x: the analyzer must not depend on it
    y, score = _scores()
    assert 0.5 < ClassificationAnalyzer(y, score).auc() <= 1.0


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_threshold_metrics_match_sklearn(threshold):
    y, score = _scores()
    pred = (score >= threshold).astype(int)
    a = ClassificationAnalyzer(y, score)
    assert a.accuracy(threshold) == pytest.approx(sk_metrics.accuracy_score(y, pred))
    assert a.precision(threshold) == pytest.approx(sk_metrics.precision_score(y, pred))
    assert a.recall(threshold) == pytest.approx(sk_metrics.recall_score(y, pred))
    assert a.f1(threshold) == pytest.approx(sk_metrics.f1_score(y, pred))


def test_threshold_defaults_to_one_half():
    y, score = _scores()
    a = ClassificationAnalyzer(y, score)
    assert a.accuracy() == a.accuracy(0.5)
    assert a.confusion_matrix() == a.confusion_matrix(0.5)


def test_confusion_matrix_counts():
    a = ClassificationAnalyzer([1, 1, 0, 0, 1], [0.9, 0.2, 0.8, 0.1, 0.6])
    assert a.confusion_matrix(0.5) == (2, 1, 1, 1)  # tp, tn, fp, fn
    assert a.fp_rate(0.5) == pytest.approx(0.5)
    assert a.fn_rate(0.5) == pytest.approx(1 / 3)


def test_bce_matches_sklearn_log_loss():
    y, score = _scores()
    score = np.clip(score, 0.01, 0.99)
    assert ClassificationAnalyzer(y, score).bce() == pytest.approx(
        sk_metrics.log_loss(y, score)
    )


def test_roc_curve_is_drawn():
    y, score = _scores()
    ax = ClassificationAnalyzer(y, score).roc_curve()
    assert ax.get_xlabel() == "False positive rate"


def test_proba_feeds_the_probability_metrics_but_not_the_labels():
    y = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    a = KNNClassificationAnalyzer(y, labels, proba=proba)
    assert a.auc() == 1.0  # the scores separate the classes perfectly
    assert a.accuracy() == 0.75  # the labels do not


def test_binary_analyzer_refuses_other_labels():
    # Labels 1/2 used to be read as "all positive" and score a perfect accuracy
    a = ClassificationAnalyzer([1, 2, 1, 2], [1, 2, 1, 2])
    with pytest.raises(ValueError, match="MulticlassAnalyzer"):
        a.accuracy()


def test_predictions_without_target_are_still_available():
    a = ClassificationAnalyzer(None, [0.2, 0.9])
    assert a.n == 2 and list(a.y_fit) == [0.2, 0.9]
    with pytest.raises(ValueError, match="true values"):
        a.accuracy()


def test_lengths_must_match():
    with pytest.raises(ValueError, match="y_true has 3"):
        ClassificationAnalyzer([0, 1, 0], [0.1, 0.9])


# --- regression ----------------------------------------------------------------


def test_regression_metrics():
    y, fit = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 3.0])
    a = RegressionAnalyzer(y, fit)
    assert a.mse() == pytest.approx(2 / 3)
    assert a.rmse() == pytest.approx((2 / 3) ** 0.5)
    assert a.mape() == pytest.approx(100 * (0 + 0.5 + 0.25) / 3)


def test_residuals_are_observed_minus_fitted():
    a = RegressionAnalyzer([5.0, 1.0], [4.0, 3.0])
    assert list(a.residuals()) == [1.0, -2.0]


def test_regression_without_target():
    a = RegressionAnalyzer(None, [1.0, 2.0])
    assert a.n == 2
    with pytest.raises(ValueError, match="true values"):
        a.rmse()


def test_gaussian_analyzer():
    y = np.array([0.0, 0.0, 3.0, 0.0])
    a = GaussianRegressionAnalyzer(y, np.zeros(4), np.ones(4))
    assert a.coverage(1.0) == 0.75  # the 3.0 is three sigmas out
    assert list(a.within_interval(1.0)) == [True, True, False, True]
    expected = np.mean(0.5 * np.log(2 * np.pi) + y**2 / 2)
    assert a.nll() == pytest.approx(expected)
    assert a.rmse() == pytest.approx(1.5)  # sqrt(9 / 4)


# --- multiclass ----------------------------------------------------------------


def _multiclass(n=240, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, k, n))
    proba = rng.dirichlet(np.ones(k), n) + 3 * np.eye(k)[pred]
    return y, pred, proba / proba.sum(axis=1, keepdims=True)


def test_multiclass_matches_sklearn():
    y, pred, proba = _multiclass()
    a = MulticlassAnalyzer(y, pred, proba=proba)
    assert a.accuracy() == pytest.approx(sk_metrics.accuracy_score(y, pred))
    assert np.array_equal(a.confusion_matrix(), sk_metrics.confusion_matrix(y, pred))
    for average in ("macro", "weighted", "micro"):
        p, r, f, _ = sk_metrics.precision_recall_fscore_support(
            y, pred, average=average
        )
        assert a.precision(average) == pytest.approx(p)
        assert a.recall(average) == pytest.approx(r)
        assert a.f1(average) == pytest.approx(f)
    assert a.log_loss() == pytest.approx(sk_metrics.log_loss(y, proba))
    assert a.auc() == pytest.approx(
        sk_metrics.roc_auc_score(y, proba, multi_class="ovr")
    )


def test_multiclass_per_class_values():
    y, pred, _ = _multiclass()
    a = MulticlassAnalyzer(y, pred)
    p, r, f, _ = sk_metrics.precision_recall_fscore_support(y, pred, average=None)
    assert np.allclose(a.precision(None), p)
    assert np.allclose(a.recall(None), r)
    assert np.allclose(a.f1(None), f)


def test_multiclass_micro_average_is_the_accuracy():
    y, pred, _ = _multiclass()
    a = MulticlassAnalyzer(y, pred)
    assert (
        a.precision("micro")
        == a.recall("micro")
        == a.f1("micro")
        == pytest.approx(a.accuracy())
    )


def test_multiclass_accuracy_is_not_fooled_by_the_label_values():
    # The binary analyzer read every non-zero label as "positive"
    y = np.array([1, 2, 3, 1, 2, 3])
    pred = np.array([2, 3, 1, 1, 2, 3])
    assert MulticlassAnalyzer(y, pred).accuracy() == pytest.approx(0.5)


def test_confusion_matrix_keeps_every_class():
    a = MulticlassAnalyzer([0, 0, 1], [0, 1, 1], classes=np.arange(3))
    assert a.confusion_matrix().shape == (3, 3)
    # a class that is never predicted has precision 0, not an error
    assert a.precision(None)[2] == 0.0


def test_multiclass_probability_metrics_need_probabilities():
    a = MulticlassAnalyzer([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="probabilities"):
        a.log_loss()
    with pytest.raises(ValueError, match="probabilities"):
        a.auc()


def test_multiclass_rejects_unknown_labels():
    a = MulticlassAnalyzer([0, 1, 5], [0, 1, 2], classes=np.arange(3))
    with pytest.raises(ValueError, match="not among the known classes"):
        a.confusion_matrix()


def test_multiclass_checks_the_probability_shape():
    with pytest.raises(ValueError, match="shape"):
        MulticlassAnalyzer([0, 1, 2], [0, 1, 2], proba=np.ones((3, 2)))


def test_multiclass_without_target():
    a = MulticlassAnalyzer(None, [0, 2, 1])
    assert a.n == 3
    with pytest.raises(ValueError, match="true values"):
        a.accuracy()


def test_classification_analyzer_picks_binary_or_multiclass():
    assert type(classification_analyzer([0, 1], [0, 1], 2)) is ClassificationAnalyzer
    assert type(classification_analyzer([0, 1, 2], [0, 1, 2], 3)) is MulticlassAnalyzer
