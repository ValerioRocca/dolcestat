import numpy as np
import pytest

from dolcestat.metrics import (
    ClassificationAnalyzer,
    GaussianRegressionAnalyzer,
    RegressionAnalyzer,
)
from dolcestat.neural_networks import (
    LU,
    BinaryCrossEntropy,
    DenseLayer,
    GaussianNegativeLogLikelihood,
    GradientDescent,
    MeanSquaredError,
    MultiHead,
    ReLU,
    Sequential,
    Sigmoid,
    Softplus,
)
from dolcestat.preprocessing import DolceSet
from helpers import make_blobs, make_regression


def _gradient_error(net, loss, X, y):
    # Largest gap between backprop's gradient and a finite-difference estimate
    for p in net.get_parameters():
        p.zero_grad()
    loss.forward(net.forward(X), y)
    net.backward(loss.backward())
    worst, h = 0.0, 1e-6
    for p in net.get_parameters():
        analytic = p.grad.copy()
        for idx in np.ndindex(p.value.shape):
            old = p.value[idx]
            p.value[idx] = old + h
            up = loss.forward(net.forward(X, training=False), y, training=False)
            p.value[idx] = old - h
            down = loss.forward(net.forward(X, training=False), y, training=False)
            p.value[idx] = old
            worst = max(worst, abs((up - down) / (2 * h) - analytic[idx]))
    return worst


def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 3))
    binary = rng.integers(0, 2, (25, 1)).astype(float)
    continuous = rng.normal(size=(25, 1))
    nets = [
        (
            Sequential(DenseLayer(ReLU(), 3, 5), DenseLayer(Sigmoid(), 5, 1)),
            BinaryCrossEntropy(),
            binary,
        ),
        (
            Sequential(DenseLayer(ReLU(), 3, 5), DenseLayer(LU(), 5, 1)),
            MeanSquaredError(),
            continuous,
        ),
        (
            MultiHead(
                Sequential(DenseLayer(ReLU(), 3, 5)),
                DenseLayer(LU(), 5, 1),
                DenseLayer(Softplus(), 5, 1),
            ),
            GaussianNegativeLogLikelihood(),
            continuous,
        ),
    ]
    for net, loss, y in nets:
        assert _gradient_error(net, loss, X, y) < 1e-6


def test_a_network_learns_xor_and_predict_returns_an_analyzer():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.uniform(0.4, 3, (60, 2)) * s for s in ([1, 1], [-1, -1], [1, -1], [-1, 1])]
    )
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    data = DolceSet.from_arrays(X, y)
    np.random.seed(0)
    net = Sequential(
        DenseLayer(ReLU(), 2, 8), DenseLayer(ReLU(), 8, 8), DenseLayer(Sigmoid(), 8, 1)
    )
    assert (
        net.fit(
            data,
            loss=BinaryCrossEntropy(),
            n_epochs=300,
            optimizer=GradientDescent(alpha=0.1),
            tol=None,
        )
        is net
    )
    result = net.predict(data)
    assert isinstance(result, ClassificationAnalyzer)
    assert (
        result.y_fit.shape == (240,) and ((result.y_fit > 0) & (result.y_fit < 1)).all()
    )
    assert result.accuracy() > 0.95


def test_predict_uses_the_analyzer_of_the_loss_it_was_fitted_with():
    X, y = make_regression(80, n_features=2)
    data = DolceSet.from_arrays(X, y)
    np.random.seed(0)
    net = Sequential(DenseLayer(ReLU(), 2, 6), DenseLayer(LU(), 6, 1))
    net.fit(data, loss=MeanSquaredError(), n_epochs=30, tol=None)
    result = net.predict(data)
    assert isinstance(result, RegressionAnalyzer) and result.y_fit.shape == (80,)
    assert result.rmse() == pytest.approx(
        np.sqrt(np.mean((net.forward(X, training=False).ravel() - y) ** 2))
    )


def test_a_two_headed_network_returns_a_gaussian_analyzer():
    X, y = make_regression(80, n_features=2)
    data = DolceSet.from_arrays(X, y)
    np.random.seed(0)
    net = MultiHead(
        Sequential(DenseLayer(ReLU(), 2, 6)),
        DenseLayer(LU(), 6, 1),
        DenseLayer(Softplus(eps=1e-3), 6, 1),
    )
    net.fit(data, loss=GaussianNegativeLogLikelihood(), n_epochs=20, tol=None)
    result = net.predict(data)
    assert isinstance(result, GaussianRegressionAnalyzer)
    assert (
        result.y_fit.shape == result.variance.shape == (80,)
        and (result.sigma > 0).all()
    )
    assert 0 <= result.coverage() <= 1


def test_predict_needs_a_fit_a_matching_width_and_no_target():
    X, y = make_regression(40, n_features=2)
    data = DolceSet.from_arrays(X, y)
    net = Sequential(DenseLayer(LU(), 2, 1))
    with pytest.raises(ValueError, match="not fitted"):
        net.predict(data)
    net.fit(data, loss=MeanSquaredError(), n_epochs=5, tol=None)
    with pytest.raises(ValueError, match="features"):
        net.predict(DolceSet.from_arrays(np.zeros((3, 4))))
    assert net.predict(DolceSet.from_arrays(X)).y_true is None


def test_a_loss_that_names_no_analyzer_says_so():
    from dolcestat.neural_networks import Loss

    class Custom(MeanSquaredError):
        analyzer = Loss.analyzer

    X, y = make_regression(20, n_features=2)
    net = Sequential(DenseLayer(LU(), 2, 1))
    net.fit(DolceSet.from_arrays(X, y), loss=Custom(), n_epochs=2, tol=None)
    with pytest.raises(NotImplementedError, match="forward"):
        net.predict(DolceSet.from_arrays(X, y))


def test_binary_cross_entropy_refuses_targets_that_are_not_binary():
    X, y = make_blobs(20)
    net = Sequential(DenseLayer(Sigmoid(), 2, 1))
    with pytest.raises(ValueError, match="binary target"):
        net.fit(DolceSet.from_arrays(X, y), loss=BinaryCrossEntropy(), n_epochs=2)


def test_fit_validates_its_data():
    net = Sequential(DenseLayer(LU(), 2, 1))
    with pytest.raises(ValueError, match="no target"):
        net.fit(DolceSet.from_arrays(np.zeros((4, 2))), loss=MeanSquaredError())
    with pytest.raises(ValueError, match="missing values"):
        net.fit(
            DolceSet.from_arrays(np.array([[np.nan, 1.0]] * 4), np.zeros(4)),
            loss=MeanSquaredError(),
        )


def test_history_is_only_kept_on_request():
    X, y = make_regression(40, n_features=2)
    data = DolceSet.from_arrays(X, y)
    net = Sequential(DenseLayer(LU(), 2, 1))
    net.fit(data, loss=MeanSquaredError(), n_epochs=5, tol=None, track_history=True)
    assert len(net.get_loss()) == 5
