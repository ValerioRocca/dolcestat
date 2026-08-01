"""Stores the prediction analyzers that expose evaluation metrics."""

import numpy as np
import seaborn as sns

from dolcestat.core.losses import BinaryCrossEntropy


class PredictionAnalyzer:
    """Base for on-demand evaluation metrics over a prediction.

    Built from the ground-truth and fitted vectors — ``y_true`` and ``y_fit`` —
    rather than from a model, so any estimator family (the GLMs in
    ``dolcestat.linear_models``, the estimators in ``dolcestat.neighbors``) can
    reuse it. Each metric is computed lazily from these two attributes when its
    method is called.

    Metrics are task-specific and therefore live on the subclasses:
    :class:`RegressionAnalyzer` owns regression metrics, while
    :class:`ClassificationAnalyzer` owns classification metrics. The KNN
    subclasses additionally surface estimator-specific extras (the weighting
    matrix / class probabilities).

    Scaffold only — the metric bodies are not implemented yet.
    """

    def __init__(self, y_true, y_fit):
        # Ground-truth targets and the model's fitted/predicted values.
        self.y_true = y_true
        self.y_fit = y_fit
        self.n = len(y_true)


class RegressionAnalyzer(PredictionAnalyzer):
    """Regression metrics computed from ``y_true``/``y_fit``."""

    # --- Overall metrics ---

    def mse(self):
        return (1 / self.n) * np.sum((self.y_fit - self.y_true) ** 2)

    def rmse(self):
        return self.mse() ** 0.5

    def mape(self, metric="mean"):
        ape = self.absolute_percentage_errors()

        if metric == "mean":
            return (1 / self.n) * np.sum(ape) * 100
        elif metric == "median":
            return np.median(ape) * 100
        else:
            raise ValueError("metric must be one of 'mean', 'median'")

    # --- Single-units metrics ---

    def squared_errors(self):
        return (self.y_fit - self.y_true) ** 2

    def residuals(self):
        return self.y_fit - self.y_true

    def absolute_percentage_errors(self):
        y_true = np.asarray(self.y_true, dtype=float)
        if np.any(y_true == 0):
            raise ValueError(
                "Cannot compute (absolute) percentage errors: y_true contains "
                "zeros, which would divide by zero."
            )
        return np.abs((y_true - self.y_fit) / y_true)

    # --- Plots ---

    def get_errors_violinplot(self, metric):
        match metric:
            case "residuals":
                errors = self.residuals()
                xlabel = "Residual"
            case "squared errors":
                errors = self.squared_errors()
                xlabel = "Squared error"
            case "ape":
                errors = self.absolute_percentage_errors()
                xlabel = "Absolute percentage error"
            case _:
                raise ValueError(
                    "metric must be one of 'residuals', 'squared errors', 'ape'"
                )

        sns.set_theme()
        ax = sns.violinplot(x=errors)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        return ax


class LinearRegressionAnalyzer(RegressionAnalyzer):
    """Regression analyzer for linear models, aware of the design width.

    Carries ``n_features`` — the number of columns of the input matrix ``X``
    (the predictors, excluding the intercept) — which the ordinary
    :class:`RegressionAnalyzer` does not need. With it, :meth:`r2` reports the
    plain coefficient of determination and :meth:`adjusted_r2` penalises it for
    model complexity.
    """

    def __init__(self, y_true, y_fit, n_features):
        super().__init__(y_true, y_fit)
        # Number of columns of the input matrix X (predictors, no intercept).
        self.n_features = n_features

    def r2(self):
        y_mean = np.mean(self.y_true)
        ss_res = np.sum((self.y_true - self.y_fit) ** 2)
        ss_tot = np.sum((self.y_true - y_mean) ** 2)
        return 1 - ss_res / ss_tot

    def adjusted_r2(self):
        return 1 - (1 - self.r2()) * (self.n - 1) / (self.n - self.n_features - 1)


class KNNRegressionAnalyzer(RegressionAnalyzer):
    """Regression analyzer for KNN, carrying the neighbour weight matrix."""

    def __init__(self, y_true, y_fit, weight_matrix=None):
        super().__init__(y_true, y_fit)
        # Per-prediction neighbour weights used to form each fitted value.
        self.weight_matrix = weight_matrix


class ClassificationAnalyzer(PredictionAnalyzer):
    """Classification metrics computed from ``y_true``/``y_fit``.

    ``y_fit`` holds class-1 probabilities (e.g. logistic regression's sigmoid
    output). Label-based metrics take an explicit ``threshold`` and derive their
    confusion-matrix counts (``tp``/``tn``/``fp``/``fn``) from it on each call,
    while probability-based metrics (BCE/ROC/AUC) use the raw scores directly.
    """

    # --- Metrics ---

    @staticmethod
    def _safe_divide(numerator, denominator):
        """Divide, returning 0.0 when the denominator is 0 (empty count)."""
        return numerator / denominator if denominator != 0 else 0.0

    def confusion_matrix(self, threshold):
        pred_pos = np.asarray(self.y_fit) >= threshold
        actual_pos = np.asarray(self.y_true).astype(bool)
        tp = int(np.sum(pred_pos & actual_pos))
        tn = int(np.sum(~pred_pos & ~actual_pos))
        fp = int(np.sum(pred_pos & ~actual_pos))
        fn = int(np.sum(~pred_pos & actual_pos))
        return tp, tn, fp, fn

    def fp_rate(self, threshold):
        _, tn, fp, __ = self.confusion_matrix(threshold)
        return self._safe_divide(fp, fp + tn)

    def fn_rate(self, threshold):
        tp, _, __, fn = self.confusion_matrix(threshold)
        return self._safe_divide(fn, fn + tp)

    def accuracy(self, threshold):
        tp, tn, _, __ = self.confusion_matrix(threshold)
        return (tp + tn) / self.n

    def precision(self, threshold):
        tp, _, fp, __ = self.confusion_matrix(threshold)
        return self._safe_divide(tp, tp + fp)

    def recall(self, threshold):
        tp, _, __, fn = self.confusion_matrix(threshold)
        return self._safe_divide(tp, fn + tp)

    def f1(self, threshold):
        p = self.precision(threshold)
        r = self.recall(threshold)
        return self._safe_divide(2 * p * r, p + r)

    def bce(self):
        # training=False: this is a pure metric, so the loss must not leave a
        # backward-pass cache behind.
        return BinaryCrossEntropy().forward(
            np.asarray(self.y_fit, dtype=float),
            np.asarray(self.y_true, dtype=float),
            training=False,
        )

    def _roc_points(self):
        # (FPR, TPR) pairs swept over thresholds, shared by the ROC plot and AUC.
        thresholds = np.arange(0, 1, step=0.001)
        fp_rates = [self.fp_rate(x) for x in thresholds]
        recalls = [self.recall(x) for x in thresholds]
        return fp_rates, recalls

    # --- Plots ---

    def roc_curve(self):
        fp_rates, recalls = self._roc_points()

        sns.set_theme()
        ax = sns.lineplot(x=fp_rates, y=recalls)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        return ax

    def auc(self):
        fp_rates, recalls = self._roc_points()

        # Thresholds sweep 0 -> 1, so FPR comes out monotonically decreasing.
        # np.trapz integrates left to right, so reverse to ascending FPR.
        fpr = np.asarray(fp_rates)[::-1]
        tpr = np.asarray(recalls)[::-1]
        return np.trapz(tpr, fpr)


class KNNClassificationAnalyzer(ClassificationAnalyzer):
    """Classification analyzer for KNN, carrying per-class probabilities."""

    def __init__(self, y_true, y_fit, proba=None):
        super().__init__(y_true, y_fit)
        # Per-prediction class probabilities (columns follow sorted classes).
        self.proba = proba
