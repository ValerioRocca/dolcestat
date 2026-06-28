import polars as pl


def is_valid_scaling_strategy(strategy):
    valid_strategies = ["min-max", "standardize"]
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid scaling strategy '{strategy}'. Valid options are: {valid_strategies}."
        )


def _resolve_per_column_param(name, value, columns):
    """Normalize a manual scaling parameter into a {column: value} dict.

    Accepts either a single scalar (applied to every column) or a mapping
    from column name to value. Returns None if no manual value was provided.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        missing = [col for col in columns if col not in value]
        if missing:
            raise ValueError(
                f"Manual '{name}' is missing values for columns: {missing}."
            )
        return {col: value[col] for col in columns}
    return {col: value for col in columns}


def standardize_dataframe(df, mean=None, std=None):
    """Standardize ``df``.

    Returns ``(scaled_df, stats)`` where ``stats`` is a dict with the per-column
    ``mean`` and ``std`` actually used (whether auto-computed or supplied).
    """
    means_dict = _resolve_per_column_param("mean", mean, df.columns)
    stds_dict = _resolve_per_column_param("std", std, df.columns)
    if means_dict is None:
        means_dict = df.mean().row(0, named=True)
    if stds_dict is None:
        stds_dict = df.std().row(0, named=True)
    scaled_df = df.with_columns(
        (pl.col(col) - means_dict[col]) / stds_dict[col] for col in df.columns
    )
    return scaled_df, {"mean": means_dict, "std": stds_dict}


def min_max_scale_dataframe(df, min=None, max=None):
    """Min-max scale ``df``.

    Returns ``(scaled_df, stats)`` where ``stats`` is a dict with the per-column
    ``min`` and ``max`` actually used (whether auto-computed or supplied).
    """
    min_dict = _resolve_per_column_param("min", min, df.columns)
    max_dict = _resolve_per_column_param("max", max, df.columns)
    if min_dict is None:
        min_dict = df.min().row(0, named=True)
    if max_dict is None:
        max_dict = df.max().row(0, named=True)
    scaled_df = df.with_columns(
        (pl.col(col) - min_dict[col]) / (max_dict[col] - min_dict[col])
        for col in df.columns
    )
    return scaled_df, {"min": min_dict, "max": max_dict}
