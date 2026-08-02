"""Stores the training loop."""

from dolcestat.core.history import History
from dolcestat.core.input_validation import (
    validate_n_epochs,
    validate_tol,
    validate_X,
    validate_X_y,
    validate_y,
)
from dolcestat.core.losses import Loss
from dolcestat.core.optimizer import Optimizer
from dolcestat.core.samplers import MiniBatchIteratingSampler, Sampler
from dolcestat.preprocessing.core import DolceSet


class Trainer:
    """
    Owns the training loop: the one place forward/backward/step is sequenced.

    Everything else is a strategy plugged into it — the model supplies the
    forward map and its parameters, the loss compares predictions to targets,
    the sampler decides how much data each step sees, and the optimizer decides
    how a gradient becomes a weight update. None of them knows about the others.

    ```python
    trainer = Trainer(
        model=Sequential(DenseLayer(LU(), 2, 1)),
        loss=MeanSquaredError(),
        optimizer=GradientDescent(alpha=0.1),
        sampler=BatchSampler(),
    )
    history = trainer.run(data, n_epochs=500)
    ```

    Units. A **step** is one optimizer update, i.e. one batch. An **epoch** is
    one full pass of the sampler over the data. With a ``BatchSampler`` the two
    coincide: one epoch is exactly one step.

    Target shape. ``DolceSet`` keeps targets 1-D, ``(n,)`` — the shape the rest
    of the library and every metric expects — while a layer emits a 2-D column,
    ``(n, 1)``. This class is the single boundary between the two conventions:
    targets are reshaped to a column **once**, here, on the way in. Everything
    downstream of this point is uniformly 2-D, so losses can compare shapes
    strictly instead of quietly reshaping, and a multi-output ``(n, k)`` target
    passes through untouched.
    """

    def __init__(
        self,
        model,
        loss: Loss,
        optimizer: Optimizer,
        sampler: Sampler = None,
    ):
        """
        Args:
            model: object exposing forward(X, training), backward(grad) and
                get_parameters() — e.g. a Sequential
            loss: loss function used to compare predictions and targets
            optimizer: update rule applied to the model's parameters
            sampler: strategy splitting the data into batches each epoch.
                Defaults to MiniBatchIteratingSampler().
        """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        # Built here rather than in the signature: a default argument is
        # evaluated once at import and would be shared by every Trainer.
        self.sampler = sampler if sampler is not None else MiniBatchIteratingSampler()

    @staticmethod
    def _as_column(y):
        """
        Normalizes targets to the 2-D layout used inside the loop.

        Args:
            y: targets, 1-D (n,) as DolceSet stores them or already 2-D

        Returns:
            The same targets shaped (n, 1), or unchanged when already 2-D
        """
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def run(
        self,
        data: DolceSet,
        n_epochs: int = 100,
        tol: float = 1e-6,
        max_steps: int = None,
        track_full_loss: bool = True,
    ) -> History:
        """
        Trains the model and returns the record of the run.

        Args:
            data: training data (DolceSet)
            n_epochs: maximum number of passes over the data
            tol: convergence tolerance. Training stops once the full-dataset
                loss changes by less than this between consecutive epochs. Pass
                None to disable the check and always run the full budget —
                needed by rules whose loss legitimately sits still for a while,
                such as the perceptron's, where a step that touches a correctly
                classified sample changes nothing. Ignored when track_full_loss
                is False.
            max_steps: hard budget on optimizer steps, or None for no budget
            track_full_loss: whether to evaluate the whole dataset once per
                epoch. Turning it off saves one forward pass per epoch and
                disables the convergence check.

        Returns:
            History of the run
        """
        validate_n_epochs(n_epochs)
        validate_tol(tol)
        validate_X(data.X)
        validate_y(data.y)
        validate_X_y(data.X, data.y)

        history = History()

        # T1. Collect the trainable parameters once: the optimizer updates these
        #     same objects in place on every step.
        parameters = self.model.get_parameters()

        # Targets become 2-D columns here and stay that way (see class docstring).
        full_targets = self._as_column(data.y)

        step = 0
        budget_spent = False

        for epoch in range(n_epochs):

            # T2. Full-dataset loss at the weights this epoch starts from.
            #     Evaluated before any of this epoch's updates, so the curve is
            #     comparable across samplers: it always answers "how good were
            #     the weights entering this epoch?", whether the epoch then
            #     takes one full-batch step or a hundred mini-batch ones.
            #     training=False leaves no backward-pass cache behind.
            if track_full_loss:
                full_pred = self.model.forward(data.X, training=False)
                history.record_epoch(
                    epoch, self.loss.forward(full_pred, full_targets, training=False)
                )

            for batch in self.sampler.sample(data):

                # T3. Zero the gradients left by the previous step. The loop owns
                #     this, not the optimizer, so the optimizer never needs to
                #     remember the parameter list between calls.
                for parameter in parameters:
                    parameter.zero_grad()

                # T4. Forward pass: compute predictions, and cache in each layer
                #     its input (needed for dL/dW) and, where applicable, its
                #     activation output (needed for da/dz in the backward pass).
                y_pred = self.model.forward(batch.X)

                # T5. Loss forward: the scalar for the record, and the cached
                #     true/predicted values the loss backward pass needs.
                batch_loss = self.loss.forward(y_pred, self._as_column(batch.y))

                # T6. Loss backward: dL/d(y_pred), the gradient at the model's
                #     output, where the chain rule starts.
                grad = self.loss.backward()

                # T7. Model backward: propagate that gradient back through the
                #     layers, each one both passing dL/d(its input) further down
                #     and accumulating dL/d(its parameters) into Parameters.grad.
                #     Concretely, per layer l, this does two things via the chain
                #     rule:
                #     1. Propagate the gradient back to the previous layer:
                #        dL/dz(l) = dL/dz(l+1) dz(l+1)/da(l) da(l)/dz(l) where:
                #        - dL/dz(l+1) is just dL/dz passed from the layer above.
                #        - dz(l+1)/da(l) is the matrix [weights(l+1), bias(l+1)]
                #        - da(l)/dz(l) is the derivative of the activation function
                #          (this is why the activation output is cached in T4).
                #        Note. The output layer has no next layer: there,
                #        dL/dz(l) = dL/da(l) da(l)/dz(l), where dL/da(l) is T6's
                #        grad.
                #     2. Compute the gradients of the layer's own parameters:
                #        dL/dW(l) = dL/dz(l) dz(l)/dW(l) where:
                #        - dL/dz(l) is computed in the previous step.
                #        - dz(l)/dW(l) is just the input to the layer (this is why
                #          it's cached in T4).
                #        These are stored in the layer's Parameter objects so T8
                #        can simply retrieve them.
                self.model.backward(grad)

                # T8. Apply the update rule. The optimizer reads the gradients
                #     accumulated in T7 and writes new parameter values.
                self.optimizer.step(parameters)

                history.record_step(epoch, step, batch_loss)
                step += 1

                if max_steps is not None and step >= max_steps:
                    budget_spent = True
                    break

            if budget_spent:
                break

            # T9. Convergence, judged on the full-dataset loss and only at epoch
            #     boundaries. Under mini-batching the per-step loss is noisy, so
            #     a tolerance applied to consecutive steps would fire spuriously.
            if track_full_loss and tol is not None and epoch > 0:
                delta = abs(
                    history.epochs[-1]["epoch_loss"] - history.epochs[-2]["epoch_loss"]
                )
                if delta < tol:
                    break

        return history
