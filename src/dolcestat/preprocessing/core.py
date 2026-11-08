"""The DolceSet data container."""

import numpy as np
import polars as pl

from dolcestat.preprocessing.scalers import (
    is_valid_scaling_strategy,
    min_max_scale_dataframe,
    standardize_dataframe,
)


def validate_polars_dataframe(df):
    """Raises a ValueError unless ``df`` is a Polars DataFrame."""
    if not isinstance(df, pl.DataFrame):
        raise ValueError(f"Expected a Polars DataFrame, but got {type(df)}.")


def is_column_in_dataframe(colname, df):
    """Raises a ValueError unless ``colname`` is a column of ``df``."""
    if colname not in df.columns:
        raise ValueError(
            f"Column '{colname}' not found in DataFrame columns: {df.columns}."
        )


class DolceSet:
    """The container every model consumes: features ``X``, target ``y`` and a
    record of the transformations applied to them.

    Build one with ``load_from_polars_dataframe``, ``load_from_dict`` or
    ``from_arrays``. A set built without a target can be used to predict but
    not to fit.
    """

    def __init__(self):
        self.df_as_polars = None
        self.X = None
        self.y = None
        self.features = None
        self.target = None
        self._scaling_applied = []
        self.can_train = None
        # Original labels of an encoded target: classes_[k] is the label that
        # was turned into the integer k. None until encode_target is called.
        self.classes_ = None

    @classmethod
    def from_arrays(cls, X, y=None, feature_names=None, target_name="target"):
        """Builds a DolceSet straight from NumPy arrays.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,), optional
            Leave it out for data you only want to predict on.
        feature_names : list of str, optional
            Defaults to ``x0, x1, ...``.
        target_name : str, default "target"
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
        feature_names = list(feature_names)
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Got {len(feature_names)} feature names for {X.shape[1]} columns."
            )
        columns = {name: X[:, i] for i, name in enumerate(feature_names)}
        if y is not None:
            y = np.asarray(y)
            if y.ndim != 1 or y.shape[0] != X.shape[0]:
                raise ValueError("y must be a 1D array with one value per row of X.")
            if target_name in columns:
                raise ValueError(f"target_name '{target_name}' is also a feature.")
            columns[target_name] = y
        data = cls()
        data.load_from_polars_dataframe(
            pl.DataFrame(columns), target_col=target_name if y is not None else None
        )
        return data

    def _require_loaded(self):
        if self.df_as_polars is None:
            raise ValueError("No data loaded: call load_from_polars_dataframe first.")

    def _require_feature(self, column_name):
        self._require_loaded()
        if column_name not in self.features:
            raise ValueError(
                f"Column '{column_name}' is not a feature of this DolceSet "
                f"(features: {self.features})."
            )

    def _validate_int_or_float_column(self, column_name):
        col_dtype = self.df_as_polars.select(pl.col(column_name)).dtypes[0]
        if not (col_dtype.is_float() or col_dtype.is_integer()):
            raise TypeError(f"Column '{column_name}' must be a float or integer type.")

    def load_from_polars_dataframe(self, df, target_col=None):
        """Loads a Polars DataFrame; ``target_col`` names the target column.

        Without a target the set can be used to predict but not to fit. Any
        scaling or target encoding recorded before is forgotten.
        """
        validate_polars_dataframe(df)
        self._scaling_applied = []
        self.classes_ = None

        if target_col is not None:
            is_column_in_dataframe(target_col, df)
            self.X = df.drop(target_col).to_numpy()
            self.y = df.select(pl.col(target_col)).to_numpy().flatten()
            self.features = df.drop(target_col).columns
            self.can_train = True

        else:
            self.X = df.to_numpy()
            self.y = None
            self.features = df.columns
            self.can_train = False

        self.df_as_polars = df
        self.target = target_col

    def load_from_dict(self, data, target_col=None):
        """Loads a ``{column name: values}`` dict; ``target_col`` names the target."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected a dict, but got {type(data)}.")
        df = pl.DataFrame(data)
        self.load_from_polars_dataframe(df, target_col=target_col)

    @property
    def scaling_info(self):
        """The scalings applied so far, to repeat them on a test set.

        A dict ``{"scaled": bool, "details": [...]}`` with one entry per call to
        ``scale``, holding its strategy, columns and the statistics used.
        """
        if not self._scaling_applied:
            return {"scaled": False, "details": []}
        return {
            "scaled": True,
            "details": [
                {"strategy": s, "columns": list(cols), "stats": stats}
                for s, cols, stats in self._scaling_applied
            ],
        }

    def get_column(self, column_name):
        """The values of a column (feature or target) as a NumPy array."""
        self._require_loaded()
        is_column_in_dataframe(column_name, self.df_as_polars)
        return self.df_as_polars.select(pl.col(column_name)).to_numpy().flatten()

    def get_colnames(self):
        """Names of all the columns, target included."""
        self._require_loaded()
        return self.df_as_polars.columns

    def get_features(self):
        """Names of the feature columns."""
        return self.features

    def get_target(self):
        """Name of the target column, or None if the set has no target."""
        return self.target

    def filter_features(self, features_to_keep):
        """Keeps only the given features (the target, if any, is kept too)."""
        features_to_keep = list(features_to_keep)
        for feature in features_to_keep:
            self._require_feature(feature)
        columns = features_to_keep + ([self.target] if self.target is not None else [])
        self.df_as_polars = self.df_as_polars.select(columns)
        self.features = features_to_keep
        self.X = self.df_as_polars.select(pl.col(self.features)).to_numpy()

    def one_hot_encode(self, column_name):
        """Replaces a categorical feature with 0/1 indicator columns.

        One category is dropped (the first), since the remaining indicators
        already determine it; keeping all of them would make the columns
        collinear. ``features`` and ``X`` are refreshed.
        """
        # 1. Check existence
        self._require_feature(column_name)
        # 2. One-hot-encode
        self.df_as_polars = self.df_as_polars.to_dummies(column_name, drop_first=True)
        # 3. Refresh features/X from the encoded columns, keeping the target out
        #    of the feature matrix (the same invariant scale() maintains).
        self.features = [c for c in self.df_as_polars.columns if c != self.target]
        self.X = self.df_as_polars.select(pl.col(self.features)).to_numpy()

    def encode_target(self, classes=None):
        """Turns the target into the integer class labels 0, 1, ..., K-1.

        Classifiers need labels in that form. By default the distinct values of
        the target are sorted and numbered; the original labels are kept in
        ``classes_`` (``decode_target`` maps integers back to them).

        To encode a test set the same way as the training set, pass the
        training set's ``classes_``. A label the training set never saw is an
        error.
        """
        if self.target is None:
            raise ValueError("This DolceSet has no target to encode.")
        if classes is None:
            classes, codes = np.unique(self.y, return_inverse=True)
        else:
            classes = np.asarray(classes)
            position = {label: i for i, label in enumerate(classes.tolist())}
            unseen = sorted({v for v in self.y.tolist() if v not in position}, key=str)
            if unseen:
                raise ValueError(
                    f"The target holds labels not in `classes`: {unseen[:8]}."
                )
            codes = np.array([position[v] for v in self.y.tolist()])
        self.y = codes.astype(int)
        self.df_as_polars = self.df_as_polars.with_columns(
            pl.Series(self.target, self.y)
        )
        self.classes_ = classes

    def decode_target(self, labels):
        """Maps integer class labels back to the original target labels."""
        if self.classes_ is None:
            raise ValueError("The target was not encoded: call encode_target first.")
        return self.classes_[np.asarray(labels, dtype=int)]

    def scale(self, scaling_strategy, column_name=None, **kwargs):
        """Scale one or more columns.

        Statistics are computed from the data by default. They can also be
        supplied manually via keyword arguments, either as a single scalar
        (applied to every scaled column) or as a {column: value} mapping:

        - "standardize": ``mean``, ``std``
        - "min-max": ``min``, ``max``
        """
        self._require_loaded()
        is_valid_scaling_strategy(scaling_strategy)
        # 1. If column_name is None, scale every feature
        if isinstance(column_name, str):
            columns_to_scale = [column_name]
        elif column_name is None:
            columns_to_scale = list(self.features)
        else:
            columns_to_scale = list(column_name)
        # 2. Check the columns are features that store int or float values
        for col in columns_to_scale:
            self._require_feature(col)
            self._validate_int_or_float_column(col)
        # 3. Validate the manual parameters allowed for this strategy
        allowed_kwargs = {
            "standardize": {"mean", "std"},
            "min-max": {"min", "max"},
        }[scaling_strategy]
        unexpected = set(kwargs) - allowed_kwargs
        if unexpected:
            raise ValueError(
                f"Unexpected scaling parameter(s) {sorted(unexpected)} for "
                f"strategy '{scaling_strategy}'. Allowed: {sorted(allowed_kwargs)}."
            )
        # 4. Scale depending on strategy
        df_to_scale = self.df_as_polars.select(pl.col(columns_to_scale))
        if scaling_strategy == "standardize":
            scaled_df, stats = standardize_dataframe(df_to_scale, **kwargs)
        else:
            scaled_df, stats = min_max_scale_dataframe(df_to_scale, **kwargs)
        # 5. Update df_as_polars in place (preserves column order, keeps target)
        self.df_as_polars = self.df_as_polars.with_columns(scaled_df)
        # 6. Refresh X/features from the feature columns only, so the target
        #    never leaks into the feature matrix.
        self.X = self.df_as_polars.select(pl.col(self.features)).to_numpy()
        self._scaling_applied.append((scaling_strategy, columns_to_scale, stats))
