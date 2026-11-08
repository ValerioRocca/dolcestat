"""Analyzers: what ``predict`` returns, with the evaluation metrics of a prediction."""

from .analyzer import (
    ClassificationAnalyzer,
    GaussianRegressionAnalyzer,
    KNNClassificationAnalyzer,
    KNNRegressionAnalyzer,
    LinearRegressionAnalyzer,
    MulticlassAnalyzer,
    PredictionAnalyzer,
    RegressionAnalyzer,
    classification_analyzer,
)

__all__ = [
    "ClassificationAnalyzer",
    "GaussianRegressionAnalyzer",
    "KNNClassificationAnalyzer",
    "KNNRegressionAnalyzer",
    "LinearRegressionAnalyzer",
    "MulticlassAnalyzer",
    "PredictionAnalyzer",
    "RegressionAnalyzer",
    "classification_analyzer",
]
