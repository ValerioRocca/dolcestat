import numpy as np
import polars as pl
import pytest

from dolcestat.linear_models import LinearRegression
from dolcestat.preprocessing import DolceSet


def _set(**columns):
    data = DolceSet()
    data.load_from_dict(columns, target_col="y")
    return data


def test_from_arrays():
    data = DolceSet.from_arrays(
        [[1.0, 2.0], [3.0, 4.0]], [0, 1], feature_names=["a", "b"]
    )
    assert data.features == ["a", "b"] and data.target == "target"
    assert data.can_train and data.X.shape == (2, 2) and list(data.y) == [0, 1]


def test_from_arrays_without_target_cannot_train():
    data = DolceSet.from_arrays(np.zeros((3, 2)))
    assert data.can_train is False and data.y is None
    assert data.features == ["x0", "x1"]


def test_from_arrays_checks_shapes():
    with pytest.raises(ValueError, match="2D"):
        DolceSet.from_arrays([1.0, 2.0])
    with pytest.raises(ValueError, match="one value per row"):
        DolceSet.from_arrays(np.zeros((3, 2)), [0, 1])


def test_filter_features_keeps_the_target_and_updates_the_feature_list():
    data = _set(
        a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0], c=[7.0, 8.0, 9.0], y=[1.0, 2.0, 3.0]
    )
    data.filter_features(["a", "b"])
    assert data.get_features() == ["a", "b"]
    assert data.get_colnames() == ["a", "b", "y"]
    assert data.X.shape == (3, 2) and list(data.y) == [1.0, 2.0, 3.0]
    data.scale("standardize")  # used to crash on the dropped column
    assert np.allclose(data.X.mean(axis=0), 0)


def test_columns_that_are_not_features_are_refused():
    data = _set(a=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not a feature"):
        data.filter_features(["nope"])
    with pytest.raises(ValueError, match="not a feature"):
        data.one_hot_encode("y")
    with pytest.raises(ValueError, match="not a feature"):
        data.scale("standardize", "y")


def test_one_hot_encode_drops_first_and_keeps_the_target_out():
    data = _set(
        size=[1.0, 2.0, 3.0, 4.0], kind=["a", "b", "c", "a"], y=[1.0, 2.0, 3.0, 4.0]
    )
    data.one_hot_encode("kind")
    assert data.get_features() == ["size", "kind_b", "kind_c"]
    assert data.X.shape == (4, 3)


def test_scaling_standardizes_and_records_the_statistics():
    data = _set(a=[1.0, 2.0, 3.0, 4.0], y=[0, 0, 1, 1])
    data.scale("standardize")
    assert np.isclose(data.X.mean(), 0) and np.isclose(data.X.std(ddof=1), 1)
    stats = data.scaling_info["details"][0]["stats"]
    test = _set(a=[5.0, 6.0], y=[1, 1])
    test.scale("standardize", mean=stats["mean"], std=stats["std"])
    assert test.X[0, 0] == pytest.approx((5.0 - 2.5) / np.std([1, 2, 3, 4], ddof=1))


def test_min_max_scaling():
    data = _set(a=[10.0, 20.0, 30.0], y=[0, 1, 0])
    data.scale("min-max")
    assert list(data.X[:, 0]) == [0.0, 0.5, 1.0]


def test_scaling_a_constant_column_is_an_error_not_nan():
    data = _set(a=[2.0, 2.0, 2.0], b=[1.0, 2.0, 3.0], y=[0, 1, 0])
    with pytest.raises(ValueError, match="'a'.*standard deviation"):
        data.scale("standardize")
    with pytest.raises(ValueError, match="'a'.*constant"):
        data.scale("min-max")


def test_scaling_rejects_unknown_parameters_and_text_columns():
    data = _set(a=[1.0, 2.0, 3.0], t=["x", "y", "z"], y=[0, 1, 0])
    with pytest.raises(ValueError, match="Unexpected"):
        data.scale("standardize", "a", minimum=1)
    with pytest.raises(TypeError, match="float or integer"):
        data.scale("standardize", "t")


def test_encode_target_numbers_the_labels_and_remembers_them():
    data = _set(a=[1.0, 2.0, 3.0, 4.0], y=["cat", "dog", "cat", "bird"])
    data.encode_target()
    assert list(data.y) == [1, 2, 1, 0]
    assert list(data.classes_) == ["bird", "cat", "dog"]
    assert list(data.decode_target([0, 2])) == ["bird", "dog"]
    assert data.get_column("y").tolist() == [1, 2, 1, 0]  # the DataFrame follows


def test_encode_target_reuses_the_training_classes_for_a_test_set():
    train = _set(a=[1.0, 2.0, 3.0], y=[5, 7, 9])
    train.encode_target()
    test = _set(a=[4.0, 5.0], y=[9, 5])
    test.encode_target(classes=train.classes_)
    assert list(test.y) == [2, 0]
    unseen = _set(a=[4.0], y=[6])
    with pytest.raises(ValueError, match="not in `classes`"):
        unseen.encode_target(classes=train.classes_)


def test_decode_target_needs_an_encoded_target():
    data = _set(a=[1.0, 2.0], y=[0, 1])
    with pytest.raises(ValueError, match="not encoded"):
        data.decode_target([0])


def test_models_refuse_missing_values_and_text_features():
    with_nan = DolceSet.from_arrays(
        np.array([[1.0], [np.nan], [3.0], [4.0]]), [1.0, 2.0, 3.0, 4.0]
    )
    with pytest.raises(ValueError, match="missing values"):
        LinearRegression().fit(with_nan)
    text = DolceSet()
    text.load_from_polars_dataframe(
        pl.DataFrame({"a": ["x", "y", "z"], "y": [1.0, 2.0, 3.0]}), target_col="y"
    )
    with pytest.raises(ValueError, match="must be numeric"):
        LinearRegression().fit(text)


def test_fitting_without_a_target_is_refused():
    with pytest.raises(ValueError, match="no target"):
        LinearRegression().fit(DolceSet.from_arrays(np.zeros((3, 1))))
