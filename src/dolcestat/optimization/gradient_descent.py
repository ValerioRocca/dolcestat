"""Stores first-order optimizers built on the step(parameters) protocol."""

import numpy as np

from dolcestat.core.optimizer import Optimizer

from .input_validation import (
    validate_alpha,
    validate_momentum_rate,
    validate_momentum_type,
)


class GradientDescent(Optimizer):
    """
    Gradient descent, with optional Polyak or Nesterov momentum.

    Plain gradient descent steps straight downhill::

        w = w - alpha * g

    Momentum instead accumulates a velocity across steps, damping the zig-zag
    across a narrow valley and building speed along its floor. The velocity is
    held here, one buffer per parameter, rather than being recovered from a
    weight history — the optimizer is the natural owner, and it keeps Parameters
    a dumb container usable by any optimizer::

        v = beta * v + g          # both variants accumulate the same velocity
        w = w - alpha * v                     # polyak   (heavy ball)
        w = w - alpha * (g + beta * v)        # nesterov (correction form)

    Nesterov, precisely. The textbook form evaluates the gradient at a
    *look-ahead* point, shifting the weights before the backward pass. A
    ``step()`` called after backpropagation structurally cannot do that: ``.grad``
    already reflects the unshifted weights. This is the correction form used by
    PyTorch — asymptotically equivalent, and it needs no extra hook in the
    training loop, but it is not step-for-step identical to true look-ahead.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        momentum_type: str = None,
        momentum_rate: float = 0.9,
    ):
        """
        Args:
            alpha: learning rate; must be in (0, 1)
            momentum_type: one of None, "polyak" or "nesterov". None disables
                momentum.
            momentum_rate: momentum coefficient in [0, 1); ignored when
                momentum_type is None.
        """
        # 1. Validate the hyperparameters.
        validate_alpha(alpha)
        validate_momentum_type(momentum_type)
        validate_momentum_rate(momentum_rate)

        # 2. Assign them.
        self.alpha = alpha
        self.momentum_type = momentum_type
        self.momentum_rate = momentum_rate

        # 3. One velocity buffer per parameter, created on first use and keyed
        #    by parameter identity. Empty while momentum is disabled.
        self._velocity = {}

    def _get_velocity(self, parameter) -> np.ndarray:
        """
        Returns:
            The velocity buffer for ``parameter``, zero-initialized on first use
        """
        if parameter not in self._velocity:
            self._velocity[parameter] = np.zeros_like(parameter.value, dtype=float)
        return self._velocity[parameter]

    def step(self, parameters: list) -> None:
        """
        Updates every parameter from the gradient stored on it.

        Args:
            parameters: list of Parameters carrying the gradients accumulated
                by the backward pass
        """
        for parameter in parameters:

            # 1. No momentum: step straight down the gradient.
            if self.momentum_type is None:
                update = parameter.grad

            else:
                # 2. Accumulate the velocity. Both variants share this line;
                #    they differ only in what they then apply.
                velocity = (
                    self.momentum_rate * self._get_velocity(parameter) + parameter.grad
                )
                self._velocity[parameter] = velocity

                # 2a. Polyak (heavy ball): move along the accumulated velocity.
                if self.momentum_type == "polyak":
                    update = velocity

                # 2b. Nesterov: correct the velocity with the current gradient,
                #     approximating a step taken from the look-ahead point.
                else:
                    update = parameter.grad + self.momentum_rate * velocity

            # 3. Apply. Rebinding rather than using -= keeps an integer-valued
            #    parameter from silently truncating the update.
            parameter.value = parameter.value - self.alpha * update
