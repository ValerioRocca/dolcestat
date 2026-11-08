import numpy as np
import pytest

from dolcestat.metrics import ClassificationAnalyzer
from dolcestat.neural_networks import RosenblattPerceptron
from dolcestat.optimization import GradientDescent, NewtonMethod, OptimizerAnalyzer
from dolcestat.preprocessing import DolceSet
from helpers import make_binary, make_blobs, make_regression


def _data():
    return DolceSet.from_arrays(*make_regression())


def test_constructors_take_settings_only_and_fit_returns_the_optimizer():
    with pytest.raises(TypeError):
        GradientDescent(_data())
    gd = GradientDescent(alpha=0.3, n_iters=50)
    assert gd.fit(_data()) is gd and gd.is_fitted
    newton = NewtonMethod()
    assert newton.fit(_data()) is newton


def test_fitting_again_starts_from_scratch():
    gd = GradientDescent(flavor="batch", alpha=0.3, n_iters=40, tol=1e-12)
    gd.fit(_data())
    first = gd.get_loss().copy()
    gd.fit(_data())
    assert np.array_equal(first, gd.get_loss())


def test_predict_returns_a_plain_array_and_needs_a_fit():
    gd = GradientDescent(flavor="batch", alpha=0.3, n_iters=300)
    with pytest.raises(ValueError, match="not fitted"):
        gd.predict(_data())
    out = gd.fit(_data()).predict(_data())
    assert isinstance(out, np.ndarray) and out.shape == (150,)


def test_newton_solves_a_quadratic_in_a_few_steps():
    X, y = make_regression()
    newton = NewtonMethod(tol=1e-12).fit(DolceSet.from_arrays(X, y))
    design = np.column_stack((X, np.ones(len(y))))
    assert np.allclose(
        newton.get_weights(-1), np.linalg.lstsq(design, y, rcond=None)[0]
    )
    assert len(newton.get_loss()) <= 4


def test_the_tracked_loss_includes_the_penalty():
    X, y = make_regression()
    data = DolceSet.from_arrays(X, y)
    gd = GradientDescent(
        flavor="batch", alpha=0.05, n_iters=20, tol=1e-15, l1=0.3, l2=1.0
    )
    gd.fit(data)
    design = np.column_stack((X, np.ones(len(y))))
    for i in (0, 5, 12):
        w = gd.weights[i]  # the loss logged at iteration i belongs to these weights
        mse = np.mean((y - design @ w) ** 2)
        penalty = 0.3 * np.abs(w[:-1]).sum() + 0.5 * 1.0 * (w[:-1] ** 2).sum()
        assert gd.get_loss(i) == pytest.approx(mse + penalty)


def test_the_plot_title_says_penalty_only_when_there_is_one():
    plain = GradientDescent(alpha=0.1, n_iters=5).fit(_data())
    penalised = GradientDescent(alpha=0.1, n_iters=5, l2=0.1).fit(_data())
    assert OptimizerAnalyzer(plain).plot_loss().get_title() == "Loss (MSE)"
    assert (
        OptimizerAnalyzer(penalised).plot_loss().get_title() == "Loss (MSE + penalty)"
    )


def test_l2_shrinks_the_weights():
    data = _data()
    free = GradientDescent(flavor="batch", alpha=0.1, n_iters=500).fit(data)
    ridge = GradientDescent(flavor="batch", alpha=0.1, n_iters=500, l2=2.0).fit(data)
    assert (
        np.abs(ridge.get_weights(-1)[:-1]).sum()
        < np.abs(free.get_weights(-1)[:-1]).sum()
    )
    assert ridge.get_weights(-1)[-1] == pytest.approx(free.get_weights(-1)[-1], abs=0.3)


@pytest.mark.parametrize("momentum", [None, "polyak", "nesterov"])
@pytest.mark.parametrize("flavor", ["batch", "mini_batch", "sgd"])
def test_every_flavor_and_momentum_reduces_the_loss(flavor, momentum):
    # Gentle settings: single-row steps with heavy momentum can overshoot
    gd = GradientDescent(
        flavor=flavor,
        alpha=0.01,
        n_iters=300,
        momentum_type=momentum,
        momentum_rate=0.5,
    )
    loss = gd.fit(_data()).get_loss()
    assert loss[-1] < loss[0]


def test_the_loss_is_inferred_from_the_target():
    assert GradientDescent().fit(_data()).loss_function == "mse"
    assert (
        GradientDescent().fit(DolceSet.from_arrays(*make_binary())).loss_function
        == "bce"
    )


def test_hyperparameters_are_validated():
    for bad in (
        {"alpha": 1.5},
        {"flavor": "x"},
        {"momentum_type": "x"},
        {"l2": -1},
        {"n_iters": 0},
    ):
        with pytest.raises(ValueError):
            GradientDescent(**bad)


def test_the_perceptron_is_a_classifier_returning_an_analyzer():
    X, y = make_blobs(40, centres=((-3, -3), (3, 3)))
    data = DolceSet.from_arrays(X, y)
    p = RosenblattPerceptron(alpha=0.1, n_iters=400)
    assert p.fit(data) is p
    result = p.predict(data)
    assert isinstance(result, ClassificationAnalyzer)
    assert result.accuracy() == 1.0  # separable data: Novikoff's theorem
    assert p.get_loss()[-1] == 0.0


def test_the_perceptron_plot_is_labelled_as_an_error_rate():
    X, y = make_blobs(20, centres=((-3, -3), (3, 3)))
    p = RosenblattPerceptron(n_iters=20).fit(DolceSet.from_arrays(X, y))
    assert OptimizerAnalyzer(p).plot_loss().get_title() == "Loss (error rate)"


def test_the_perceptron_refuses_non_binary_targets():
    X, y = make_blobs(20)
    with pytest.raises(ValueError, match="binary target"):
        RosenblattPerceptron().fit(DolceSet.from_arrays(X, y))
    with pytest.raises(ValueError, match="binary target"):
        RosenblattPerceptron().fit(DolceSet.from_arrays(X[:20], y[:20] + 1))
