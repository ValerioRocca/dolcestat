import polars as pl

from dolcestat.preprocessing.scalers import (
    is_valid_scaling_strategy,
    min_max_scale_dataframe,
    standardize_dataframe,
)


def validate_polars_dataframe(df):
    if not isinstance(df, pl.DataFrame):
        raise ValueError(f"Expected a Polars DataFrame, but got {type(df)}.")


def is_column_in_dataframe(colname, df):
    if colname not in df.columns:
        raise ValueError(
            f"Column '{colname}' not found in DataFrame columns: {df.columns}."
        )


class DolceSet:

    def __init__(self):
        self.df_as_polars = None
        self.X = None
        self.y = None
        self.features = None
        self.target = None
        self._scaling_applied = []
        self.can_train = None

    def _is_column_in_features(self, column_name):
        if column_name not in self.features:
            print(
                f"Column '{column_name}' not found in features. DolceSet object cannot be used for training."
            )

    def _validate_int_or_float_column(self, column_name):
        col_dtype = self.df_as_polars.select(pl.col(column_name)).dtypes[0]
        if not (col_dtype.is_float() or col_dtype.is_integer()):
            raise TypeError(f"Column '{column_name}' must be a float or integer type.")

    def load_from_polars_dataframe(self, df, target_col=None):
        validate_polars_dataframe(df)
        self._scaling_applied = []

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
        if not isinstance(data, dict):
            raise ValueError(f"Expected a dict, but got {type(data)}.")
        df = pl.DataFrame(data)
        self.load_from_polars_dataframe(df, target_col=target_col)

    @property
    def scaling_info(self):
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
        is_column_in_dataframe(column_name, self.df_as_polars)
        return self.df_as_polars.select(pl.col(column_name)).to_numpy().flatten()

    def get_colnames(self):
        return self.df_as_polars.columns

    def get_features(self):
        return self.features

    def get_target(self):
        return self.target

    def filter_features(self, features_to_keep):
        for feature in features_to_keep:
            self._is_column_in_features(feature)
        self.df_as_polars = self.df_as_polars.select(pl.col(features_to_keep))
        self.X = self.df_as_polars.to_numpy()

    def one_hot_encode(self, column_name):
        # 1. Check existence
        self._is_column_in_features(column_name)
        # 2. One-hot-encode
        self.df_as_polars = self.df_as_polars.to_dummies(column_name, drop_first=True)
        # 3. Refresh features/X from the encoded columns, keeping the target out
        #    of the feature matrix (the same invariant scale() maintains).
        self.features = [c for c in self.df_as_polars.columns if c != self.target]
        self.X = self.df_as_polars.select(pl.col(self.features)).to_numpy()

    def scale(self, scaling_strategy, column_name=None, **kwargs):
        """Scale one or more columns.

        Statistics are computed from the data by default. They can also be
        supplied manually via keyword arguments, either as a single scalar
        (applied to every scaled column) or as a {column: value} mapping:

        - "standardize": ``mean``, ``std``
        - "min-max": ``min``, ``max``
        """
        # 1. Check existence
        if column_name is not None:
            self._is_column_in_features(column_name)
        is_valid_scaling_strategy(scaling_strategy)
        # 2. If column_name is None, standardize all dataset
        if isinstance(column_name, str):
            columns_to_scale = [column_name]
        elif column_name is None:
            columns_to_scale = self.features
        else:
            columns_to_scale = column_name
        # 2. Check if columns store int or float values
        for col in columns_to_scale:
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
        elif scaling_strategy == "min-max":
            scaled_df, stats = min_max_scale_dataframe(df_to_scale, **kwargs)
        else:
            raise NotImplementedError(f"This should not have happened.")
        # 5. Update df_as_polars in place (preserves column order, keeps target)
        self.df_as_polars = self.df_as_polars.with_columns(scaled_df)
        # 6. Refresh X/features from the feature columns only, so the target
        #    never leaks into the feature matrix.
        self.X = self.df_as_polars.select(pl.col(self.features)).to_numpy()
        self._scaling_applied.append((scaling_strategy, columns_to_scale, stats))
