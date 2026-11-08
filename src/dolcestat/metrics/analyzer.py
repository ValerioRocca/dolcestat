"""Evaluation metrics for regression and classification predictions."""

import numpy as np
import seaborn as sns

from dolcestat.optimization.loss_and_activation import compute_bce_loss


def _trapezoid(y, x):
    """Area under the piecewise-linear curve through the points (x, y).

    Written out rather than calling ``np.trapz``/``np.trapezoid``: NumPy
    renamed the function in 2.0 and removed the old name.
    """
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


class PredictionAnalyzer:
    """Base for on-demand evaluation metrics over a prediction.

    Built from the ground-truth and fitted vectors — ``y_true`` and ``y_fit`` —
    rather than from a model, so any estimator family (the GLMs in
    ``dolcestat.linear_models``, the estimators in ``dolcestat.neighbors``, the
    trees, the networks) can reuse it. Each metric is computed lazily from these
    two attributes when its method is called.

    ``y_true`` may be None, for a prediction made on data with no target:
    ``y_fit`` is still available, and any metric that needs the truth says so.

    Metrics are task-specific and therefore live on the subclasses:
    :class:`RegressionAnalyzer` owns regression metrics, while
    :class:`ClassificationAnalyzer` (two classes) and
    :class:`MulticlassAnalyzer` (more) own classification metrics. The KNN
    subclasses additionally surface estimator-specific extras (the weighting
    matrix / class probabilities).
    """

    def __init__(self, y_true, y_fit):
        # Ground-truth targets and the model's fitted/predicted values.
        self.y_true = None if y_true is None else np.asarray(y_true)
        self.y_fit = np.asarray(y_fit)
        self.n = len(self.y_fit)
        if self.y_true is not None and len(self.y_true) != self.n:
            raise ValueError(
                f"y_true has {len(self.y_true)} values but y_fit has {self.n}."
            )

    def _truth(self):
        if self.y_true is None:
            raise ValueError(
                "This metric needs the true values, but this prediction was "
                "made on data without a target."
            )
        return self.y_true


class RegressionAnalyzer(PredictionAnalyzer):
    """Regression metrics computed from ``y_true``/``y_fit``."""

    # --- Overall metrics ---

    def mse(self):
        """Mean squared error: the average of the squared errors."""
        return (1 / self.n) * np.sum((self.y_fit - self._truth()) ** 2)

    def rmse(self):
        """Root mean squared error: the square root of :meth:`mse`, in units of y."""
        return self.mse() ** 0.5

    def mape(self, metric="mean"):
        """Mean (or median) absolute percentage error, as a percentage.

        ``metric`` is "mean" or "median". Raises a ValueError when ``y_true``
        contains zeros, since the error is relative to the true value.
        """
        ape = self.absolute_percentage_errors()

        if metric == "mean":
            return (1 / self.n) * np.sum(ape) * 100
        elif metric == "median":
            return np.median(ape) * 100
        else:
            raise ValueError("metric must be one of 'mean', 'median'")

    # --- Single-units metrics ---

    def squared_errors(self):
        """Squared error of each row: ``(y_fit - y_true) ** 2``."""
        return (self.y_fit - self._truth()) ** 2

    def residuals(self):
        """Observed minus fitted: positive where the model under-predicts."""
        return self._truth() - self.y_fit

    def absolute_percentage_errors(self):
        """Absolute percentage error of each row, as a fraction of ``|y_true|``."""
        y_true = np.asarray(self._truth(), dtype=float)
        if np.any(y_true == 0):
            raise ValueError(
                "Cannot compute (absolute) percentage errors: y_true contains "
                "zeros, which would divide by zero."
            )
        return np.abs((y_true - self.y_fit) / y_true)

    # --- Plots ---

    def get_errors_violinplot(self, metric):
        """Violin plot of the distribution of the errors; returns the axes.

        ``metric`` is "residuals", "squared errors" or "ape" (absolute
        percentage errors).
        """
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
        """Coefficient of determination: the share of the variance of y explained.

        1 is a perfect fit and 0 is no better than predicting the mean of y.
        It is not defined when y is constant.
        """
        y_true = self._truth()
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - self.y_fit) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        return 1 - ss_res / ss_tot

    def adjusted_r2(self):
        """R² corrected for the number of predictors, so extra features must pay off."""
        return 1 - (1 - self.r2()) * (self.n - 1) / (self.n - self.n_features - 1)


class KNNRegressionAnalyzer(RegressionAnalyzer):
    """Regression analyzer for KNN, carrying the neighbour weight matrix."""

    def __init__(self, y_true, y_fit, weight_matrix=None):
        super().__init__(y_true, y_fit)
        # Per-prediction neighbour weights used to form each fitted value.
        self.weight_matrix = weight_matrix


class GaussianRegressionAnalyzer(RegressionAnalyzer):
    """Regression analyzer for a model that predicts a mean *and* a variance.

    ``y_fit`` is the predicted mean mu(x); ``variance`` is the predicted
    sigma^2(x). Besides the usual regression metrics it can judge the
    uncertainty itself: :meth:`coverage` is the share of targets that fall
    within mu +/- n_sigma * sigma (about 68% for one sigma, if calibrated).
    """

    def __init__(self, y_true, y_fit, variance):
        super().__init__(y_true, y_fit)
        self.variance = np.asarray(variance)

    @property
    def sigma(self):
        """Predicted standard deviation of each row: the square root of ``variance``."""
        return np.sqrt(self.variance)

    def within_interval(self, n_sigma=1.0):
        """Per-row booleans: is the target within mu +/- n_sigma * sigma?"""
        return np.abs(self._truth() - self.y_fit) < n_sigma * self.sigma

    def coverage(self, n_sigma=1.0):
        """Share of targets within mu +/- n_sigma * sigma (see ``within_interval``)."""
        return float(np.mean(self.within_interval(n_sigma)))

    def nll(self):
        """Mean negative log-likelihood of the targets under N(mu, sigma^2)."""
        return float(
            np.mean(
                0.5 * np.log(2 * np.pi * self.variance)
                + (self._truth() - self.y_fit) ** 2 / (2 * self.variance)
            )
        )


class ClassificationAnalyzer(PredictionAnalyzer):
    """Binary classification metrics computed from ``y_true``/``y_fit``.

    The classes are the labels 0 and 1. ``y_fit`` holds either class-1
    probabilities (logistic regression's sigmoid output, a network's) or hard
    0/1 labels (a tree's). Label-based metrics take a ``threshold`` (default
    0.5) and derive their confusion-matrix counts (``tp``/``tn``/``fp``/``fn``)
    from it on each call, while probability-based metrics (BCE/ROC/AUC) use the
    raw scores directly.

    ``proba`` is optional: an (n, 2) matrix of class probabilities for models
    that return hard labels in ``y_fit`` but can also score them (KNN, forests).
    When present, BCE/ROC/AUC use ``proba[:, 1]`` instead of the labels.

    For more than two classes see :class:`MulticlassAnalyzer`.
    """

    def __init__(self, y_true, y_fit, proba=None):
        super().__init__(y_true, y_fit)
        self.proba = None if proba is None else np.asarray(proba)

    def _truth(self):
        y_true = super()._truth()
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError(
                "ClassificationAnalyzer works with the labels 0 and 1, but "
                f"y_true holds {np.unique(y_true)[:8].tolist()}. For more than "
                "two classes use MulticlassAnalyzer; to convert other labels "
                "use DolceSet.encode_target()."
            )
        return y_true

    def _scores(self):
        # What the model says about "class 1": probabilities when it has them
        if self.proba is not None:
            return self.proba[:, 1]
        return np.asarray(self.y_fit, dtype=float)

    # --- Metrics ---

    @staticmethod
    def _safe_divide(numerator, denominator):
        """Divide, returning 0.0 when the denominator is 0 (empty count)."""
        return numerator / denominator if denominator != 0 else 0.0

    def confusion_matrix(self, threshold=0.5):
        """Counts ``(tp, tn, fp, fn)``; positive means ``y_fit >= threshold``."""
        pred_pos = np.asarray(self.y_fit) >= threshold
        actual_pos = self._truth().astype(bool)
        tp = int(np.sum(pred_pos & actual_pos))
        tn = int(np.sum(~pred_pos & ~actual_pos))
        fp = int(np.sum(pred_pos & ~actual_pos))
        fn = int(np.sum(~pred_pos & actual_pos))
        return tp, tn, fp, fn

    def fp_rate(self, threshold=0.5):
        """False positive rate: the share of actual negatives called positive."""
        _, tn, fp, __ = self.confusion_matrix(threshold)
        return self._safe_divide(fp, fp + tn)

    def fn_rate(self, threshold=0.5):
        """False negative rate: the share of actual positives called negative."""
        tp, _, __, fn = self.confusion_matrix(threshold)
        return self._safe_divide(fn, fn + tp)

    def accuracy(self, threshold=0.5):
        """Share of rows classified correctly."""
        tp, tn, _, __ = self.confusion_matrix(threshold)
        return (tp + tn) / self.n

    def precision(self, threshold=0.5):
        """Share of the rows called positive that are (0.0 if none were called)."""
        tp, _, fp, __ = self.confusion_matrix(threshold)
        return self._safe_divide(tp, tp + fp)

    def recall(self, threshold=0.5):
        """Share of the actual positives that were found (0.0 if there are none)."""
        tp, _, __, fn = self.confusion_matrix(threshold)
        return self._safe_divide(tp, fn + tp)

    def f1(self, threshold=0.5):
        """Harmonic mean of precision and recall."""
        p = self.precision(threshold)
        r = self.recall(threshold)
        return self._safe_divide(2 * p * r, p + r)

    def bce(self):
        """Mean binary cross-entropy of the class-1 scores against y_true."""
        return compute_bce_loss(self._truth().astype(float), self._scores())

    def _roc_points(self):
        # (FPR, TPR) pairs for every distinct score used as a threshold, from
        # (0, 0) to (1, 1). Shared by the ROC plot and AUC.
        truth = self._truth().astype(bool)
        n_pos = int(truth.sum())
        n_neg = truth.size - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError("ROC and AUC need both classes present in y_true.")
        scores = self._scores()
        order = np.argsort(-scores, kind="mergesort")
        scores, truth = scores[order], truth[order]
        true_positives = np.cumsum(truth)
        false_positives = np.cumsum(~truth)
        # Rows with the same score enter the positive set together, so keep
        # only the last row of each group of ties.
        last_of_group = np.r_[np.where(np.diff(scores))[0], scores.size - 1]
        fpr = np.r_[0.0, false_positives[last_of_group] / n_neg]
        tpr = np.r_[0.0, true_positives[last_of_group] / n_pos]
        return fpr, tpr

    # --- Plots ---

    def roc_curve(self):
        """Plots the ROC curve (true against false positive rate); returns the axes.

        The dashed diagonal is a classifier that guesses. Needs both classes in
        ``y_true``.
        """
        fpr, tpr = self._roc_points()

        sns.set_theme()
        ax = sns.lineplot(x=fpr, y=tpr)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        return ax

    def auc(self):
        """Area under the ROC curve: how often a positive outscores a negative."""
        fpr, tpr = self._roc_points()
        return _trapezoid(tpr, fpr)


class KNNClassificationAnalyzer(ClassificationAnalyzer):
    """Binary classification analyzer for KNN, carrying class probabilities."""


class MulticlassAnalyzer(PredictionAnalyzer):
    """Classification metrics for three or more classes.

    ``y_fit`` holds the predicted class labels. ``proba`` is an optional
    (n_samples, n_classes) matrix of class probabilities, with columns in the
    order of ``classes``; the probability-based metrics (:meth:`log_loss`,
    :meth:`auc`) need it and say so when it is missing.

    ``classes`` lists every class the model knows, so the confusion matrix keeps
    its K x K shape even if the evaluated data lacks some class. It defaults to
    the labels found in the data.
    """

    def __init__(self, y_true, y_fit, proba=None, classes=None):
        super().__init__(y_true, y_fit)
        if classes is None:
            classes = np.unique(self.y_fit if self.y_true is None else self.y_true)
        self.classes = np.sort(np.asarray(classes))
        self.proba = None if proba is None else np.asarray(proba)
        if self.proba is not None and self.proba.shape != (self.n, len(self.classes)):
            raise ValueError(
                f"proba must have shape (n_samples, n_classes) = "
                f"{(self.n, len(self.classes))}, got {self.proba.shape}."
            )

    def _probabilities(self):
        if self.proba is None:
            raise ValueError(
                "This metric needs class probabilities, but the model only "
                "returned predicted labels."
            )
        return self.proba

    def _positions(self, labels):
        # Index of each label within `classes`
        positions = np.searchsorted(self.classes, labels)
        positions = np.clip(positions, 0, len(self.classes) - 1)
        if np.any(self.classes[positions] != labels):
            unknown = np.unique(labels[self.classes[positions] != labels])
            raise ValueError(
                f"Labels {unknown[:8].tolist()} are not among the known classes "
                f"{self.classes.tolist()}."
            )
        return positions

    # --- Metrics ---

    def confusion_matrix(self):
        """K x K counts: rows are the true class, columns the predicted one."""
        k = len(self.classes)
        matrix = np.zeros((k, k), dtype=int)
        np.add.at(
            matrix, (self._positions(self._truth()), self._positions(self.y_fit)), 1
        )
        return matrix

    def accuracy(self):
        """Share of rows whose predicted class is the true one."""
        return float(np.mean(self._truth() == self.y_fit))

    def _counts(self):
        matrix = self.confusion_matrix()
        tp = np.diag(matrix).astype(float)
        return tp, matrix.sum(axis=0) - tp, matrix.sum(axis=1) - tp

    @staticmethod
    def _divide(numerator, denominator):
        # 0 where the denominator is 0 (a class that was never predicted, ...)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )

    @staticmethod
    def _average(per_class, support, average):
        if average is None:
            return per_class
        if average == "macro":
            return float(per_class.mean())
        if average == "weighted":
            return float(np.average(per_class, weights=support))
        raise ValueError("average must be one of None, 'macro', 'weighted', 'micro'")

    def precision(self, average="macro"):
        """Share of predictions of a class that were right.

        ``average``: None for one value per class; "macro" (plain mean over
        classes), "weighted" (mean weighted by class size) or "micro" (pool
        every prediction, which equals the accuracy).
        """
        tp, fp, fn = self._counts()
        if average == "micro":
            return self._divide(tp.sum(), tp.sum() + fp.sum()).item()
        return self._average(self._divide(tp, tp + fp), tp + fn, average)

    def recall(self, average="macro"):
        """Share of a class's members that were found. See :meth:`precision`."""
        tp, fp, fn = self._counts()
        if average == "micro":
            return self._divide(tp.sum(), tp.sum() + fn.sum()).item()
        return self._average(self._divide(tp, tp + fn), tp + fn, average)

    def f1(self, average="macro"):
        """Harmonic mean of precision and recall. See :meth:`precision`."""
        tp, fp, fn = self._counts()
        if average == "micro":
            return self._divide(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum()).item()
        per_class = self._divide(2 * tp, 2 * tp + fp + fn)
        return self._average(per_class, tp + fn, average)

    def log_loss(self):
        """Mean negative log-probability given to the true class."""
        proba = np.clip(self._probabilities(), 1e-15, 1 - 1e-15)
        columns = self._positions(self._truth())
        return float(-np.mean(np.log(proba[np.arange(self.n), columns])))

    def auc(self):
        """One-vs-rest AUC, averaged over the classes.

        For each class, how well does its probability column rank that class's
        members above everything else; ties count half.
        """
        y_true, proba = self._truth(), self._probabilities()
        scores = []
        for k, label in enumerate(self.classes):
            positives = proba[y_true == label, k]
            negatives = proba[y_true != label, k]
            if len(positives) == 0 or len(negatives) == 0:
                raise ValueError(
                    f"AUC needs class {label} and at least one other class in y_true."
                )
            wins = (positives[:, None] > negatives[None, :]).sum()
            ties = (positives[:, None] == negatives[None, :]).sum()
            scores.append((wins + 0.5 * ties) / (len(positives) * len(negatives)))
        return float(np.mean(scores))


def classification_analyzer(y_true, y_fit, n_classes, proba=None):
    """The analyzer that fits a classifier's output: binary or multiclass.

    Used by the classifiers' ``predict``. With two classes (or fewer) the
    binary :class:`ClassificationAnalyzer` is returned; otherwise a
    :class:`MulticlassAnalyzer` over the classes ``0..n_classes-1``.
    """
    if n_classes > 2:
        return MulticlassAnalyzer(
            y_true, y_fit, proba=proba, classes=np.arange(n_classes)
        )
    return ClassificationAnalyzer(y_true, y_fit, proba=proba)
