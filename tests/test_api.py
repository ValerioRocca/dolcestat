"""Every model follows the same rhythm: build from settings, fit(data), predict(data)."""

import numpy as np
import pytest

from dolcestat.clustering import HierarchicalClustering, KMeans
from dolcestat.linear_models import LinearRegression, LogisticRegression
from dolcestat.metrics import PredictionAnalyzer
from dolcestat.neighbors import KNNClassifier, KNNRegressor
from dolcestat.neural_networks import (
    LU,
    BinaryCrossEntropy,
    DenseLayer,
    MeanSquaredError,
    RosenblattPerceptron,
    Sequential,
    Sigmoid,
)
from dolcestat.optimization import GradientDescent, NewtonMethod
from dolcestat.preprocessing import DolceSet
from dolcestat.trees import DecisionTree, GradientBoostingTree, RandomForest
from helpers import make_binary, make_regression, to_sets

REGRESSORS = {
    "linear": lambda: LinearRegression(),
    "linear-newton": lambda: LinearRegression("newton"),
    "tree": lambda: DecisionTree(max_depth=3, impurity_measure="mse"),
    "forest": lambda: RandomForest(
        n_trees=4, n_feat=2, max_depth=3, impurity_measure="mse"
    ),
    "boosting": lambda: GradientBoostingTree(
        max_depth=2, max_forest_size=4, impurity_measure="mse"
    ),
    "knn": lambda: KNNRegressor(k=3),
}
CLASSIFIERS = {
    "logistic": lambda: LogisticRegression("newton"),
    "tree": lambda: DecisionTree(max_depth=3),
    "forest": lambda: RandomForest(n_trees=4, n_feat=2, max_depth=3),
    "boosting": lambda: GradientBoostingTree(max_depth=2, max_forest_size=4),
    "knn": lambda: KNNClassifier(k=3),
    "perceptron": lambda: RosenblattPerceptron(n_iters=50),
}


def _check_rhythm(make_model, train, test):
    model = make_model()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(test)
    assert model.fit(train) is model  # fit returns the model, so calls chain
    result = model.predict(test)
    assert (
        isinstance(result, PredictionAnalyzer) and len(result.y_fit) == test.X.shape[0]
    )
    # no target is needed to predict
    blind = model.predict(DolceSet.from_arrays(test.X))
    assert blind.y_true is None and np.allclose(blind.y_fit, result.y_fit)
    # the model refuses data of the wrong width
    with pytest.raises(ValueError, match="features"):
        model.predict(DolceSet.from_arrays(np.zeros((2, test.X.shape[1] + 1))))
    # and a set with no target cannot be fitted
    with pytest.raises(ValueError, match="no target"):
        make_model().fit(DolceSet.from_arrays(train.X))


@pytest.mark.parametrize("name", REGRESSORS)
def test_regressors_share_the_rhythm(name):
    train, test = to_sets(*make_regression(80), 60)
    _check_rhythm(REGRESSORS[name], train, test)


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_classifiers_share_the_rhythm(name):
    train, test = to_sets(*make_binary(120), 90)
    _check_rhythm(CLASSIFIERS[name], train, test)


def test_networks_share_the_rhythm_too():
    train, test = to_sets(*make_regression(60, n_features=2), 45)
    net = Sequential(DenseLayer(LU(), 2, 1))
    with pytest.raises(ValueError, match="not fitted"):
        net.predict(test)
    assert net.fit(train, loss=MeanSquaredError(), n_epochs=3, tol=None) is net
    assert len(net.predict(test).y_fit) == 15
    train, test = to_sets(*make_binary(60), 45)
    clf = Sequential(DenseLayer(Sigmoid(), 2, 1))
    assert clf.fit(train, loss=BinaryCrossEntropy(), n_epochs=3, tol=None) is clf
    assert isinstance(clf.predict(test), PredictionAnalyzer)


def test_clustering_and_optimizers_fit_and_return_themselves():
    data = DolceSet.from_arrays(make_regression(40)[0])
    for model in (KMeans(2), HierarchicalClustering()):
        assert model.fit(data) is model
    train = DolceSet.from_arrays(*make_regression(40))
    for optimizer in (GradientDescent(n_iters=5), NewtonMethod(n_iters=5)):
        assert optimizer.fit(train) is optimizer


@pytest.mark.parametrize(
    "build",
    [
        DecisionTree,
        RandomForest,
        GradientBoostingTree,
        GradientDescent,
        NewtonMethod,
        RosenblattPerceptron,
    ],
)
def test_constructors_take_settings_not_data(build):
    with pytest.raises(TypeError):
        build(DolceSet.from_arrays(*make_binary(20)))
