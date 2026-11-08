import copy

import numpy as np
import pytest

from dolcestat.linear_models import LinearRegression, LogisticRegression
from dolcestat.optimization import GradientDescent, NewtonMethod
from dolcestat.preprocessing import DolceSet
from helpers import make_binary, make_blobs, make_regression, to_sets

linear_model = pytest.importorskip("sklearn.linear_model")


def test_ols_matches_sklearn():
    X, y = make_regression()
    data = DolceSet.from_arrays(X, y)
    model = LinearRegression().fit(data)
    ref = linear_model.LinearRegression().fit(X, y)
    assert np.allclose(model.weights[:-1], ref.coef_)
    assert model.weights[-1] == pytest.approx(ref.intercept_)
    assert model.predict(data).r2() == pytest.approx(ref.score(X, y))


def test_gradient_descent_and_newton_reach_the_ols_solution():
    X, y = make_regression()
    data = DolceSet.from_arrays(X, y)
    exact = LinearRegression().fit(data).weights
    newton = LinearRegression("newton").fit(data).weights
    assert np.allclose(newton, exact, atol=1e-6)
    gd = LinearRegression(
        GradientDescent(flavor="batch", alpha=0.3, n_iters=2000, tol=1e-12)
    )
    assert np.allclose(gd.fit(data).weights, exact, atol=1e-3)


def test_logistic_newton_matches_sklearn():
    X, y = make_binary(300)
    model = LogisticRegression("newton").fit(DolceSet.from_arrays(X, y))
    ref = linear_model.LogisticRegression(C=1e12, tol=1e-10, max_iter=10000).fit(X, y)
    assert np.allclose(model.weights[:-1], ref.coef_[0], atol=1e-5)
    assert model.weights[-1] == pytest.approx(ref.intercept_[0], abs=1e-5)


def test_logistic_default_optimizer_learns_the_signal():
    X, y = make_binary(300)
    data = DolceSet.from_arrays(X, y)
    assert LogisticRegression().fit(data).predict(data).accuracy() > 0.7


def test_fit_returns_the_model_and_keeps_training_metrics():
    data = DolceSet.from_arrays(*make_regression())
    model = LinearRegression()
    assert model.fit(data) is model
    assert model.training_metrics.r2() > 0.9


def test_predict_before_fit_and_with_the_wrong_width():
    train, test = to_sets(*make_regression(), 100)
    with pytest.raises(ValueError, match="not fitted"):
        LinearRegression().predict(test)
    model = LinearRegression().fit(train)
    with pytest.raises(ValueError, match="features"):
        model.predict(DolceSet.from_arrays(np.zeros((3, 2))))


def test_predict_without_target():
    train, test = to_sets(*make_regression(), 100)
    unlabelled = DolceSet.from_arrays(test.X)
    model = LinearRegression().fit(train)
    assert np.allclose(model.predict(unlabelled).y_fit, model.predict(test).y_fit)
    assert model.predict(unlabelled).y_true is None
    logit = LogisticRegression("newton").fit(DolceSet.from_arrays(*make_binary()))
    assert len(logit.predict(DolceSet.from_arrays(np.zeros((4, 2)))).y_fit) == 4


def test_an_optimizer_instance_is_a_template_and_refits_work():
    data = DolceSet.from_arrays(*make_regression())
    template = GradientDescent(flavor="batch", alpha=0.3, n_iters=50)
    before = copy.deepcopy(template.__dict__)
    model = LinearRegression(template)
    first = model.fit(data).weights.copy()
    second = model.fit(data).weights  # used to raise "already has training data"
    assert np.allclose(first, second)
    assert not template.is_fitted and template.__dict__.keys() == before.keys()
    assert model.optimization.gd is not template


def test_an_optimizer_without_a_loss_takes_the_models_loss():
    # A 0.0/1.0 float target made the optimizer infer "mse" and fit the wrong thing
    X, y = make_binary()
    data = DolceSet.from_arrays(X, y.astype(float))
    model = LogisticRegression(
        GradientDescent(flavor="batch", alpha=0.5, n_iters=200)
    ).fit(data)
    assert model.optimizer_obj.loss_function == "bce"


def test_an_optimizer_with_the_wrong_loss_is_refused():
    data = DolceSet.from_arrays(*make_binary())
    with pytest.raises(ValueError, match="needs 'bce'"):
        LogisticRegression(GradientDescent(loss_function="mse")).fit(data)


def test_optimizer_choice_is_validated_up_front():
    with pytest.raises(ValueError, match="closed form"):
        LogisticRegression("closed form")
    with pytest.raises(ValueError, match="optimizer must be"):
        LinearRegression(3)


def test_logistic_regression_refuses_targets_that_are_not_binary():
    X, y = make_blobs(40)
    with pytest.raises(ValueError, match="binary target"):
        LogisticRegression("newton").fit(DolceSet.from_arrays(X, y))
    with pytest.raises(ValueError, match="binary target"):
        LogisticRegression().fit(DolceSet.from_arrays(X[:20], y[:20] + 1))


def test_collinear_features_give_a_clear_error():
    X = np.column_stack([np.arange(10.0), 2 * np.arange(10.0)])
    data = DolceSet.from_arrays(X, np.arange(10.0))
    with pytest.raises(ValueError, match="collinear"):
        LinearRegression().fit(data)
    with pytest.raises(ValueError, match="singular"):
        LinearRegression("newton").fit(data)


def test_the_optimization_property_needs_an_iterative_fit():
    data = DolceSet.from_arrays(*make_regression())
    with pytest.raises(ValueError, match="closed-form"):
        LinearRegression().fit(data).optimization
    with pytest.raises(ValueError, match="fit"):
        LinearRegression().optimization
    assert isinstance(
        LinearRegression(NewtonMethod()).fit(data).optimizer_obj, NewtonMethod
    )
