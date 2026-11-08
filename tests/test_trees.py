import numpy as np
import pytest

from dolcestat.metrics import (
    ClassificationAnalyzer,
    MulticlassAnalyzer,
    RegressionAnalyzer,
)
from dolcestat.preprocessing import DolceSet
from dolcestat.trees import DecisionTree, GradientBoostingTree, RandomForest
from helpers import make_binary, make_blobs, make_regression, to_sets

sk_tree = pytest.importorskip("sklearn.tree")
sk_ensemble = pytest.importorskip("sklearn.ensemble")


# --- DecisionTree ----------------------------------------------------------------


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classification_tree_matches_sklearn(criterion):
    X, y = make_binary(250)
    tree = DecisionTree(max_depth=4, impurity_measure=criterion).fit(
        DolceSet.from_arrays(X, y)
    )
    ref = sk_tree.DecisionTreeClassifier(
        max_depth=4, criterion=criterion, random_state=0
    ).fit(X, y)
    assert np.mean(tree._predict_values(X) == ref.predict(X)) > 0.97


def test_regression_tree_matches_sklearn():
    X, y = make_regression(200)
    tree = DecisionTree(max_depth=4, impurity_measure="mse").fit(
        DolceSet.from_arrays(X, y)
    )
    ref = sk_tree.DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    assert np.allclose(tree._predict_values(X), ref.predict(X))


def test_constructor_takes_settings_only_and_fit_returns_the_tree():
    train, test = to_sets(*make_binary(), 150)
    with pytest.raises(TypeError):
        DecisionTree(train)
    tree = DecisionTree(max_depth=3)
    assert tree.fit(train) is tree
    assert isinstance(tree.predict(test), ClassificationAnalyzer)
    assert tree.tree.n_samples == 150 and not hasattr(tree.tree, "X")


def test_categorical_split_is_an_equality_test():
    rng = np.random.default_rng(0)
    category = rng.integers(0, 4, 200).astype(float)
    # class 1 exactly for categories 0 and 2: no single threshold separates them
    y = np.isin(category, [0, 2]).astype(int)
    data = DolceSet.from_arrays(np.column_stack([category, rng.normal(size=200)]), y)
    tree = DecisionTree(max_depth=2, categorical_features=[0]).fit(data)
    assert tree.tree.is_categorical_split
    assert tree.predict(data).accuracy() == 1.0
    numeric = DecisionTree(max_depth=1).fit(data)
    assert numeric.predict(data).accuracy() < 1.0


def test_categorical_features_are_checked_against_the_width_of_x():
    data = DolceSet.from_arrays(*make_binary(60))
    with pytest.raises(ValueError, match="out of bounds"):
        DecisionTree(categorical_features=[5]).fit(data)
    with pytest.raises(ValueError, match="integer"):
        DecisionTree(categorical_features=["a"])


def test_predict_needs_a_fit_and_the_right_width():
    train, test = to_sets(*make_binary(), 150)
    with pytest.raises(ValueError, match="not fitted"):
        DecisionTree().predict(test)
    tree = DecisionTree(max_depth=2).fit(train)
    with pytest.raises(ValueError, match="features"):
        tree.predict(DolceSet.from_arrays(np.zeros((3, 5))))


def test_predicting_without_a_target():
    train, test = to_sets(*make_binary(), 150)
    result = DecisionTree(max_depth=3).fit(train).predict(DolceSet.from_arrays(test.X))
    assert len(result.y_fit) == 50 and result.y_true is None


def test_integer_valued_float_labels_are_accepted():
    X, y = make_binary(120)
    tree = DecisionTree(max_depth=3).fit(DolceSet.from_arrays(X, y.astype(float)))
    assert tree.predict(DolceSet.from_arrays(X, y)).accuracy() > 0.7


@pytest.mark.parametrize("shift", [1, 5])
def test_labels_that_do_not_start_at_zero_are_refused(shift):
    X, y = make_blobs(20)
    with pytest.raises(ValueError, match="encode_target"):
        DecisionTree().fit(DolceSet.from_arrays(X, y + shift))


def test_text_and_fractional_labels_are_refused_with_a_clear_message():
    X, y = make_blobs(20)
    labels = np.array(["a", "b", "c"])[y]
    with pytest.raises(ValueError, match="numeric"):
        DecisionTree().fit(DolceSet.from_arrays(X, labels))
    with pytest.raises(ValueError, match="integers"):
        DecisionTree().fit(DolceSet.from_arrays(X, y + 0.5))


def test_encode_target_makes_text_labels_usable():
    X, y = make_blobs(30)
    train = DolceSet.from_arrays(X, np.array(["a", "b", "c"])[y])
    train.encode_target()
    result = DecisionTree(max_depth=4).fit(train).predict(train)
    assert isinstance(result, MulticlassAnalyzer) and result.accuracy() > 0.8
    assert set(train.decode_target(result.y_fit)) <= {"a", "b", "c"}


# --- multiclass trees ---------------------------------------------------------------


def test_multiclass_tree_metrics_are_computed_from_the_labels():
    X, y = make_blobs(40)
    train, test = to_sets(X, y, 90)
    result = DecisionTree(max_depth=4).fit(train).predict(test)
    assert isinstance(result, MulticlassAnalyzer)
    assert result.accuracy() == pytest.approx(np.mean(result.y_fit == test.y))
    assert result.confusion_matrix().sum() == 30


def test_the_analyzer_keeps_every_class_even_if_the_test_set_misses_one():
    X, y = make_blobs(40)
    tree = DecisionTree(max_depth=3).fit(DolceSet.from_arrays(X, y))
    only_two = DolceSet.from_arrays(X[y < 2][:20], y[y < 2][:20])
    assert tree.predict(only_two).confusion_matrix().shape == (3, 3)


# --- RandomForest ----------------------------------------------------------------


def test_forest_votes_and_exposes_vote_shares():
    X, y = make_binary(200)
    train, test = to_sets(X, y, 150)
    forest = RandomForest(n_trees=15, n_feat=2, max_depth=4).fit(train)
    result = forest.predict(test)
    assert isinstance(result, ClassificationAnalyzer)
    assert result.proba.shape == (50, 2) and np.allclose(result.proba.sum(axis=1), 1)
    assert result.accuracy() > 0.65
    assert 0.5 < result.auc() <= 1.0


def test_forest_with_categorical_features_renumbers_the_columns():
    # Column 7 is categorical; with 3 columns per tree its position inside a
    # tree is 0-2. The old code kept "7" and crashed or split the wrong column.
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=(120, 7)), rng.integers(0, 3, 120)]).astype(
        float
    )
    y = (X[:, 7] == 1).astype(int)
    data = DolceSet.from_arrays(X, y)
    forest = RandomForest(
        n_trees=25, n_feat=3, max_depth=3, categorical_features=[7]
    ).fit(data)
    used_categorical = [
        t.tree.is_categorical_split
        for t in forest.forest
        if 7 in t.feature_idxs.tolist() and not t.tree.is_leaf
    ]
    assert used_categorical and all(used_categorical)
    assert forest.predict(data).accuracy() > 0.85


def test_forest_default_n_feat_adapts_to_the_data():
    X, y = make_binary(80)  # two columns: the old default of 5 raised an error
    forest = RandomForest(n_trees=3, max_depth=2).fit(DolceSet.from_arrays(X, y))
    assert all(len(t.feature_idxs) == 1 for t in forest.forest)
    with pytest.raises(ValueError, match="cannot be larger"):
        RandomForest(n_trees=2, n_feat=5).fit(DolceSet.from_arrays(X, y))


def test_regression_forest_averages_the_trees():
    X, y = make_regression(150)
    train, test = to_sets(X, y, 100)
    forest = RandomForest(
        n_trees=15, n_feat=3, max_depth=4, impurity_measure="mse"
    ).fit(train)
    result = forest.predict(test)
    assert isinstance(result, RegressionAnalyzer)
    assert result.rmse() < np.std(test.y)


def test_multiclass_forest():
    X, y = make_blobs(40)
    train, test = to_sets(X, y, 90)
    result = RandomForest(n_trees=15, n_feat=2, max_depth=4).fit(train).predict(test)
    assert isinstance(result, MulticlassAnalyzer)
    assert result.accuracy() == pytest.approx(np.mean(result.y_fit == test.y))
    assert result.log_loss() > 0 and 0.5 < result.auc() <= 1.0


def test_forest_predicts_without_a_target():
    train, test = to_sets(*make_binary(), 150)
    forest = RandomForest(n_trees=5, n_feat=2, max_depth=3).fit(train)
    assert len(forest.predict(DolceSet.from_arrays(test.X)).y_fit) == 50


# --- GradientBoostingTree -----------------------------------------------------------


def test_boosting_regression_is_close_to_sklearn():
    X, y = make_regression(200)
    train, test = to_sets(X, y, 150)
    gb = GradientBoostingTree(
        alpha=0.1, max_depth=3, max_forest_size=40, impurity_measure="mse"
    ).fit(train)
    ref = sk_ensemble.GradientBoostingRegressor(
        n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0
    ).fit(train.X, train.y)
    mine = gb.predict(test).rmse()
    theirs = np.sqrt(np.mean((ref.predict(test.X) - test.y) ** 2))
    assert mine == pytest.approx(theirs, rel=0.25)


def test_boosting_is_deterministic_and_fitting_again_starts_over():
    train, test = to_sets(*make_regression(100), 80)
    gb = GradientBoostingTree(max_depth=2, max_forest_size=8, impurity_measure="mse")
    first = gb.fit(train).predict(test).y_fit.copy()
    assert len(gb.forest) == 8
    second = gb.fit(train).predict(test).y_fit  # used to stack 8 more trees on top
    assert len(gb.forest) == 8
    assert np.allclose(first, second)


def test_boosting_defaults_fit_a_simple_signal():
    X = np.linspace(0, 1, 60).reshape(-1, 1)
    y = 4 * X[:, 0] + 1
    data = DolceSet.from_arrays(X, y)
    result = GradientBoostingTree(impurity_measure="mse").fit(data).predict(data)
    assert result.rmse() < 0.15  # the old defaults (alpha=0.01) stopped far short


def test_binary_boosting_returns_probabilities():
    train, test = to_sets(*make_binary(250), 180)
    gb = GradientBoostingTree(max_depth=2, max_forest_size=20).fit(train)
    result = gb.predict(test)
    assert isinstance(result, ClassificationAnalyzer)
    assert ((result.y_fit > 0) & (result.y_fit < 1)).all()
    assert result.accuracy() > 0.65 and result.auc() > 0.7
    assert gb.classes_.tolist() == [0, 1] and len(gb.forest) == 2


def test_multiclass_boosting_predicts():
    X, y = make_blobs(40)
    train, test = to_sets(X, y, 90)
    gb = GradientBoostingTree(alpha=0.3, max_depth=2, max_forest_size=15).fit(train)
    result = gb.predict(test)  # used to raise: "predict() cannot wrap multiclass"
    assert isinstance(result, MulticlassAnalyzer)
    assert np.allclose(result.proba.sum(axis=1), 1)
    assert result.accuracy() > 0.6  # the blobs overlap, and chance is 0.33
    ref = sk_ensemble.GradientBoostingClassifier(
        n_estimators=15, learning_rate=0.3, max_depth=2, random_state=0
    ).fit(train.X, train.y)
    assert result.accuracy() == pytest.approx(ref.score(test.X, test.y), abs=0.12)


def test_boosting_labels_must_start_at_zero():
    X, y = make_binary(60)
    with pytest.raises(ValueError, match="encode_target"):
        GradientBoostingTree().fit(DolceSet.from_arrays(X, y + 1))


def test_boosting_predicts_without_a_target():
    train, test = to_sets(*make_binary(150), 100)
    gb = GradientBoostingTree(max_depth=2, max_forest_size=5).fit(train)
    assert len(gb.predict(DolceSet.from_arrays(test.X)).y_fit) == 50
    with pytest.raises(ValueError, match="not fitted"):
        GradientBoostingTree().predict(test)


def test_boosting_subtrees_always_split_on_mse():
    train, _ = to_sets(*make_binary(60), 60)
    gb = GradientBoostingTree(
        max_depth=2, max_forest_size=2, impurity_measure="entropy"
    ).fit(train)
    assert {t.impurity_measure for trees in gb.forest for t in trees} == {"mse"}


def test_hyperparameters_are_validated():
    for cls, bad in [
        (DecisionTree, {"max_depth": 0}),
        (DecisionTree, {"impurity_measure": "x"}),
        (RandomForest, {"n_trees": 0}),
        (RandomForest, {"n_feat": 0}),
        (GradientBoostingTree, {"alpha": -1}),
        (GradientBoostingTree, {"max_forest_size": 0}),
    ]:
        with pytest.raises(ValueError):
            cls(**bad)
