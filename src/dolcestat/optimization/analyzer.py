"""Stores the convergence analyzer for training runs."""

import matplotlib.pyplot as plt
import seaborn as sns

from dolcestat.core.history import History


class OptimizerAnalyzer:
    """Plots and compares the convergence of one or more training runs.

    Built from :class:`History` objects — the record a ``Trainer`` returns. It
    reads diagnostics only, never the parameters themselves, which is the
    separation that let optimizers stop owning their own storage.
    """

    def __init__(self, runs, labels=None):
        """
        Args:
            runs: a History, or an iterable of them
            labels: one label per run, or None to number them
        """
        if isinstance(runs, History):
            runs = [runs]
        else:
            runs = list(runs)
        if not runs:
            raise ValueError("Provide at least one History object.")
        if not all(isinstance(run, History) for run in runs):
            raise TypeError(
                "All objects must be History instances (as returned by "
                "Trainer.run, or held on a fitted model's .history)."
            )
        self.runs = runs
        self.labels = self._resolve_labels(labels)

    @staticmethod
    def _loss_series(run, per_step: bool):
        """
        Returns:
            The loss curve for ``run``: per optimizer step when ``per_step``,
            otherwise per epoch.
        """
        return run.batch_loss if per_step else run.epoch_loss

    def _default_label(self, run):
        return "run"

    def _resolve_labels(self, labels):
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(self.runs):
                raise ValueError(f"Got {len(labels)} labels for {len(self.runs)} runs.")
            return labels
        base = [self._default_label(run) for run in self.runs]
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
        self,
        from_iter: int = 0,
        to_iter: int = None,
        log_scale: bool = False,
        per_step: bool = False,
        ax=None,
    ):
        """
        Draws the convergence curve(s).

        Args:
            from_iter: first index to plot
            to_iter: last index to plot, or None for all of it
            log_scale: whether to put the loss on a log scale
            per_step: plot the per-step batch loss instead of the per-epoch
                full-dataset loss. Under mini-batching the per-step curve shows
                the gradient noise the per-epoch curve averages out.
            ax: existing matplotlib axes to draw on, or None to create one

        Returns:
            The matplotlib axes
        """
        sns.set_theme()
        if ax is None:
            fig, ax = plt.subplots()

        multiple = len(self.runs) > 1
        for run, label in zip(self.runs, self.labels):
            loss = self._loss_series(run, per_step)
            end = len(loss) if to_iter is None else min(to_iter, len(loss))
            iterations = list(range(from_iter, end))
            sns.lineplot(
                x=iterations,
                y=loss[from_iter:end],
                ax=ax,
                label=label if multiple else None,
            )

        ax.set_xlabel("Step" if per_step else "Epoch")
        ax.set_ylabel("Loss")
        if multiple:
            ax.set_title("Loss comparison")
            ax.legend(title="run")
        else:
            ax.set_title("Loss")
        if log_scale:
            ax.set_yscale("log")
        return ax
