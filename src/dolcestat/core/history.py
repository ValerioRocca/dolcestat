"""Stores the training-run recorder."""

import numpy as np


class History:
    """
    Per-step and per-epoch record of a training run.

    Separating this from the optimizer is the point: parameter *state* lives in
    ``Parameters.value``, and the *diagnostics* live here. Optimizers used to own
    both, which is why every optimizer had to inherit the storage machinery and
    why momentum could only be recovered by indexing back into the trace.

    Two losses are recorded, because under mini-batching they answer different
    questions:

    - ``batch_loss`` — one entry per optimizer step, on that step's batch only.
      Free: the forward pass already computed it. Noisy by nature, which makes
      mini-batch gradient noise visible rather than smoothing it away.
    - ``epoch_loss`` — one entry per epoch, over the whole dataset, evaluated at
      the weights the epoch starts from. Costs one extra full-batch forward pass
      per epoch, and is the series convergence is judged on: a tolerance applied
      to consecutive *step* losses would fire on noise.

    ``len(history)`` counts optimizer steps, and is the direct successor to the
    old ``len(optimizer.get_loss())``.

    Note. No weight trace is kept. The old one existed because "the current
    weights" meant "the last row of the history" and because momentum was
    reverse-engineered from it; both of those needs are gone, and the final
    weights are simply ``Parameters.value``.
    """

    def __init__(self):
        self.steps = []
        self.epochs = []

    def record_step(self, epoch: int, step: int, batch_loss: float) -> None:
        """
        Records one optimizer step.

        Args:
            epoch: index of the epoch the step belongs to
            step: global step index, counted across all epochs
            batch_loss: loss on the batch this step was taken from
        """
        self.steps.append(
            {"epoch": epoch, "step": step, "batch_loss": float(batch_loss)}
        )

    def record_epoch(self, epoch: int, epoch_loss: float) -> None:
        """
        Records one epoch's full-dataset loss.

        Args:
            epoch: index of the epoch
            epoch_loss: loss over the whole dataset at the weights the epoch
                starts from
        """
        self.epochs.append({"epoch": epoch, "epoch_loss": float(epoch_loss)})

    @property
    def batch_loss(self) -> np.ndarray:
        """
        Returns:
            Per-step batch losses, in step order
        """
        return np.array([entry["batch_loss"] for entry in self.steps])

    @property
    def epoch_loss(self) -> np.ndarray:
        """
        Returns:
            Per-epoch full-dataset losses, in epoch order
        """
        return np.array([entry["epoch_loss"] for entry in self.epochs])

    @property
    def n_steps(self) -> int:
        """
        Returns:
            Number of optimizer steps taken
        """
        return len(self.steps)

    @property
    def n_epochs(self) -> int:
        """
        Returns:
            Number of epochs recorded
        """
        return len(self.epochs)

    def __len__(self) -> int:
        return len(self.steps)
