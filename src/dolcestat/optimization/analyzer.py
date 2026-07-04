import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseOptimizer


class OptimizerAnalyzer:
    def __init__(self, optimizers, labels=None):
        # Accept either a single BaseOptimizer or an iterable of them.
        if isinstance(optimizers, BaseOptimizer):
            optimizers = [optimizers]
        else:
            optimizers = list(optimizers)
        if not optimizers:
            raise ValueError("Provide at least one BaseOptimizer object.")
        if not all(isinstance(o, BaseOptimizer) for o in optimizers):
            raise TypeError(
                "All objects must be instances of BaseOptimizer "
                "(e.g. GradientDescent, NewtonMethod)."
            )
        self.optimizers = optimizers
        self.labels = self._resolve_labels(labels)

    @property
    def gd(self):
        # Backward-compatible alias for the first (often only) model.
        return self.optimizers[0]

    def _default_label(self, optimizer):
        # GradientDescent exposes a flavor; fall back to the class name.
        flavor = getattr(optimizer, "flavor", None)
        return flavor if flavor is not None else type(optimizer).__name__

    def _resolve_labels(self, labels):
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(self.optimizers):
                raise ValueError(
                    f"Got {len(labels)} labels for {len(self.optimizers)} optimizers."
                )
            return labels
        base = [self._default_label(o) for o in self.optimizers]
        seen = {}
        resolved = []
        for name in base:
            if base.count(name) > 1:
                seen[name] = seen.get(name, 0) + 1
                resolved.append(f"{name} ({seen[name]})")
            else:
                resolved.append(name)
        return resolved

    def plot_loss(
        self, from_iter: int = 0, to_iter: int = None, log_scale: bool = False, ax=None
    ):
        sns.set_theme()
        if ax is None:
            fig, ax = plt.subplots()

        multiple = len(self.optimizers) > 1
        for optimizer, label in zip(self.optimizers, self.labels):
            loss = optimizer.get_loss()
            end = len(loss) if to_iter is None else min(to_iter, len(loss))
            iterations = list(range(from_iter, end))
            sns.lineplot(
                x=iterations,
                y=loss[from_iter:end],
                ax=ax,
                label=label if multiple else None,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        if multiple:
            ax.set_title("Loss comparison")
            ax.legend(title="optimizer")
        else:
            optimizer = self.optimizers[0]
            ax.set_title(f"Loss ({optimizer.loss_function.upper()})")
        if log_scale:
            ax.set_yscale("log")
        return ax
