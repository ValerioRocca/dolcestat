"""Stores neural network container classes."""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.neural_networks.input_validation import (
    validate_n_epochs,
    validate_tol,
)
from dolcestat.neural_networks.losses import Loss
from dolcestat.neural_networks.optimizers import GradientDescent, Optimizer
from dolcestat.neural_networks.samplers import MiniBatchIteratingSampler, Sampler
from dolcestat.preprocessing.core import DolceSet


class NeuralNetwork(ABC):
    """
    Abstract base class for neural network containers.

    A subclass supplies only its topology -- ``forward``, ``backward`` and
    ``get_parameters``. Training and prediction are inherited, because the loop
    below touches nothing topology-specific: it calls those three methods and
    knows nothing about chains, branches or layer counts.
    """

    #: Per-epoch training loss of the last ``fit(track_history=True)`` run.
    #: None until such a run happens. See ``get_loss``.
    loss = None

    @abstractmethod
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Computes the Neural Network's output."""

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the network.

        Args:
            grad_out: dL/d(output), same shape as forward's output

        Returns:
            dL/d(input), same shape as forward's input. Side effect:
            accumulates dL/d(param) into each Parameter.grad.
        """

    @abstractmethod
    def get_parameters(self) -> list:
        """
        Returns:
            List of Parameters held by the network.
        """

    def fit(
        self,
        data: DolceSet,
        loss: Loss,
        n_epochs: int = 100,
        sampler: Sampler = None,
        optimizer: Optimizer = None,
        tol: float = 1e-6,
        track_history: bool = False,
    ) -> "NeuralNetwork":
        """
        Trains the network on ``data``.

        Args:
            data: training data (DolceSet)
            loss: loss function used to compare predictions and targets.
                Required: a general network has no sensible default, and
                defaulting to a classification loss would silently do the wrong
                thing for regression.
            n_epochs: maximum number of passes over the (sampled) data
            sampler: strategy used to split data into (mini-)batches each epoch.
                Defaults to MiniBatchIteratingSampler(). Whether training is
                batch, stochastic or mini-batch is decided here.
            optimizer: update rule applied to the parameters after each batch.
                Defaults to GradientDescent().
            tol: convergence tolerance. Training stops once the epoch loss
                changes by less than this between consecutive epochs. Pass None
                to run the full budget.
            track_history: whether to record the per-epoch loss for later
                inspection via ``get_loss``. Off by default -- a run that is
                only being fitted does not need the trace.

        Returns:
            self
        """
        validate_n_epochs(n_epochs)
        validate_tol(tol)

        # F1. Resolve the strategy defaults here rather than in the signature.
        #     A default argument is evaluated once, at import, so every network
        #     trained with defaults would otherwise share one sampler and one
        #     optimizer instance.
        if sampler is None:
            sampler = MiniBatchIteratingSampler()
        if optimizer is None:
            optimizer = GradientDescent()
        if track_history:
            self.loss = np.array([])

        previous_epoch_loss = None

        for epoch in range(n_epochs):
            # F2. Extract one or more (mini-)batches depending on the chosen strategy
            batched_data = sampler.sample(data)
            batch_losses = []

            for batch in batched_data:
                X_batch = batch.X

                # F3. Targets become a 2-D column here and stay that way. DolceSet
                #     stores y flat, (n,), while a single-output layer predicts an
                #     (n, 1) column; left unaligned the two broadcast to (n, n) and
                #     every loss below would silently return a meaningless number.
                #     Doing it once, here, is why the losses can simply assert that
                #     the shapes match.
                y_batch = batch.y.reshape(-1, 1)

                # F4. Zero the gradients (from previous iterations). Gradients for this
                #     batch will be computed during step F7.
                for p in self.get_parameters():
                    p.zero_grad()

                # F5. Forward pass: compute the model's predictions and caches, for
                #     each layer, its input (used to compute dL/dW) and (if applicable)
                #     sigmoid output (used to compute dz(l+1)/da(l) in the backward step).
                logits = self.forward(X_batch)

                # F6. Based on the predictions, compute the loss and caches true and
                #     predicted values for the backward step.
                loss_value = loss.forward(logits, y_batch)
                batch_losses.append(loss_value)

                # F7. Loss backward pass: compute dL/da(l) - with l output layer -
                #     using the true/predicted values cached in F6. A loss reading
                #     several predictions at once returns one gradient per head.
                grad = loss.backward()

                # F8. Model backward pass: for each layer, this function does two things:
                #     1. Propagate the gradient back to the previous layer via chain rule:
                #        dL/dz(l) = dL/dz(l+1) dz(l+1)/da(l) da(l)/dz(l) where:
                #        - dL/dz(l+1) is just dL/dz passed from the layer above.
                #        - dz(l+1)/da(l) is the matrix [weights(l+1), bias(l+1)]
                #        - da(l)/dz(l) is the derivative of the activation function (this
                #          is why we cached the sigmoid output in F5).
                #        Note. The output layer of course does not have a next layer.
                #        For the output layer, dL/dz(l) = dL/da(l) da(l)/dz(l), where
                #        dL/da(l) is computed in F7.
                #     2. Compute the gradients of the layer's parameters (weights and bias)
                #        again via chain rule: dL/dW(l) = dL/dz(l) dz(l)/dW(l) where:
                #        - dL/dz(l) is computed in the previous step.
                #        - dz(l)/dW(l) is just the input to the layer (this is why we cached
                #          it in F5).
                #        The gradients are stored in layer's parameters object, enabling
                #        the optimizer to simply retrieve them in the next step (F9).
                self.backward(grad)

                # F9. Update the model's parameters via chosen optimizer (default is
                #     plain mini-batch gradient descent).
                optimizer.step(self.get_parameters())

            # F10. Summarize the epoch by the mean of its batch losses. This is
            #      computed whether or not it is kept, because the convergence
            #      check below needs it; track_history only decides whether the
            #      trace survives the call.
            epoch_loss = float(np.mean(batch_losses))
            if track_history:
                self.loss = np.append(self.loss, epoch_loss)

            # F11. Stop early once the epoch loss stops moving.
            if (
                tol is not None
                and previous_epoch_loss is not None
                and abs(epoch_loss - previous_epoch_loss) < tol
            ):
                break
            previous_epoch_loss = epoch_loss

        return self

    def get_loss(self, iteration=None):
        """
        Returns the per-epoch training loss recorded by ``fit``.

        Mirrors the accessor on ``dolcestat.optimization``'s BaseOptimizer, so
        the convergence trace is read the same way everywhere in the library.

        Args:
            iteration: epoch index to read. When None, the whole trace.

        Returns:
            The loss at ``iteration``, or the full 1-D array of epoch losses

        Raises:
            ValueError: if the last fit did not record a history.
        """
        if self.loss is None:
            raise ValueError(
                "No training history was recorded. Call "
                "fit(..., track_history=True) to keep the per-epoch loss."
            )
        if iteration is not None:
            return self.loss[iteration]
        return self.loss

    def predict(self, X: np.ndarray):
        """
        Computes the model's predictions for ``X``.

        Args:
            X: input features (numpy array)

        Returns:
            The network's output with ``training=False`` -- a single array for a
            chain, one array per head for a MultiHead.
        """
        return self.forward(X, training=False)


class Sequential(NeuralNetwork):
    """
    Feed-forward network that runs a fixed sequence of layers.

    Example of initialization.:
    ```python
    my_seq = Sequential(
        DenseLayer(input_size=32, activation=ReLU(), output_size=16),
        DenseLayer(input_size=16, activation=ReLU(), output_size=8),
        DenseLayer(input_size=8, activation=LU(), output_size=1)
    )
    ```
    """

    def __init__(self, *layers):
        """
        Args:
            *layers: Layer instances to run in sequence, input to output
        """
        self.layers = layers

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Runs ``X`` through each layer in sequence.

        Args:
            X: input features (numpy array)
            training: whether the forward pass is used for training

        Returns:
            Network output
        """
        for layer in self.layers:
            X = layer.forward(X, training=training)
        return X

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through each layer in reverse order.

        Args:
            grad_out: dL/d(output), same shape as forward's output

        Returns:
            dL/d(input), same shape as forward's input. Side effect:
            accumulates dL/d(param) into each layer's Parameter.grad.
        """
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def get_parameters(self) -> list:
        """
        Returns:
            List of Parameters (weights and biases) across all layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MultiHead(NeuralNetwork):
    """
    A shared trunk feeding several independent heads — one output per head.

    Use this when one input must produce several distinct predictions: the mean
    and variance of a heteroscedastic regression, the gate/means/variances of a
    mixture density network, a policy and a value.

    Why this is not a Sequential. A ``Sequential`` is a *chain*: element i's
    output is element i+1's input, one array in, one array out. Here the trunk's
    output feeds two consumers at once, so the computation graph has a node of
    out-degree two — a tree, not a path. The heads are incomparable, neither
    precedes the other, and no total ordering exists. In other words a chain
    expresses **composition**, ``f = f_k . ... . f_1``, while a branch expresses
    **pairing**, ``<g1, g2> . trunk``, a map into a product space; no amount of
    chaining produces a product codomain. PyTorch draws exactly this line —
    ``nn.Sequential`` cannot branch or return a tuple, so multi-head models
    subclass ``nn.Module`` and reuse ``nn.Sequential`` as the trunk, which is
    what this class does too.

    ```python
    net = MultiHead(
        Sequential(                                 # shared trunk
            DenseLayer(ReLU(), input_size=32, output_size=16),
            DenseLayer(ReLU(), input_size=16, output_size=8),
        ),
        DenseLayer(LU(), 8, 1),                     # mean head     -> mu
        DenseLayer(Softplus(), 8, 1),               # variance head -> sigma^2
    )
    mu, var = net.predict(X)
    ```

    Widths. Every head consumes the *same* trunk output, so every head's first
    layer must declare ``input_size`` equal to the trunk's output width. What is
    free, and free to differ between heads, is each head's ``output_size``, depth
    and activation — a mixture density network can pair two width-k heads with a
    width-1 shared-variance head.
    """

    def __init__(self, trunk: NeuralNetwork, *heads):
        """
        Args:
            trunk: network run once per forward pass, whose output every head
                consumes — typically a Sequential
            *heads: one head per output, each either a bare Layer or a
                NeuralNetwork. At least one is required.
        """
        if not heads:
            raise ValueError(
                "MultiHead needs at least one head; got none. A network with a "
                "single output is a Sequential."
            )

        self.trunk = trunk

        # A bare Layer exposes parameters(), a NeuralNetwork exposes
        # get_parameters(). Normalizing here — the one place that has to know the
        # difference — lets a depth-1 head be written as a plain DenseLayer while
        # keeping forward, backward and get_parameters free of type checks. Note
        # that self.heads therefore always holds NeuralNetworks, even where a
        # Layer was passed in.
        self.heads = tuple(
            head if isinstance(head, NeuralNetwork) else Sequential(head)
            for head in heads
        )

    def forward(self, X: np.ndarray, training: bool = True) -> tuple:
        """
        Runs ``X`` through the trunk, then through every head.

        Args:
            X: input features (numpy array)
            training: whether the forward pass is used for training

        Returns:
            One output per head, in the order the heads were given. A tuple
            rather than a stacked array: the heads may differ in width and always
            differ in meaning, so the caller unpacks them by name.
        """
        shared = self.trunk.forward(X, training=training)
        return tuple(head.forward(shared, training=training) for head in self.heads)

    def backward(self, grad_outs) -> np.ndarray:
        """
        Backpropagates one gradient per head, then their sum through the trunk.

        Args:
            grad_outs: one dL/d(output) per head, each the shape of that head's
                forward output

        Returns:
            dL/d(input), same shape as forward's input. Side effect: accumulates
            dL/d(param) into every Parameter of the trunk and of each head.
        """
        if len(grad_outs) != len(self.heads):
            raise ValueError(
                f"Expected one gradient per head: {len(self.heads)} head(s), but "
                f"got {len(grad_outs)} gradient(s). The loss must return one "
                "gradient per prediction it was given."
            )

        # B1. Each head backpropagates independently down to the trunk's output.
        #     The heads share no parameters, so these passes cannot interfere.
        #
        # B2. The trunk then receives their SUM. This is the one piece of new
        #     mathematics in the class: the trunk's output feeds several
        #     consumers, so by the multivariate chain rule each trunk activation's
        #     total derivative is the sum of its derivatives along each path,
        #     dL/dh = sum over heads of (dL/dh through that head). A Sequential
        #     never has to show this because there the sum has a single term.
        #
        #     The shapes always agree, whatever the heads' output widths: a head's
        #     backward contracts over its own output index (grad_z @ W.T) and so
        #     lands back in the trunk's output width regardless.
        grad_shared = None
        for head, grad_out in zip(self.heads, grad_outs):
            grad_head = head.backward(grad_out)
            grad_shared = grad_head if grad_shared is None else grad_shared + grad_head

        return self.trunk.backward(grad_shared)

    def get_parameters(self) -> list:
        """
        Returns:
            List of Parameters: the trunk's, then each head's in head order. The
            optimizer sees one flat list and never learns that heads exist.
        """
        params = list(self.trunk.get_parameters())
        for head in self.heads:
            params.extend(head.get_parameters())
        return params
