"""Stores neural network container classes."""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.core.losses import Loss
from dolcestat.core.optimizer import Optimizer
from dolcestat.core.samplers import Sampler
from dolcestat.core.trainer import Trainer
from dolcestat.optimization.gradient_descent import GradientDescent
from dolcestat.preprocessing.core import DolceSet


class NeuralNetwork(ABC):
    """Abstract base class for neural network containers."""

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
        **run_kwargs,
    ) -> "NeuralNetwork":
        """
        Trains the network on ``data``.

        This is a convenience wrapper: it assembles a Trainer and runs it. The
        loop itself lives in Trainer, so a network, a GLM and a perceptron all
        train through exactly one implementation. Build the Trainer yourself
        when you want to drive the loop directly.

        It lives on the base class because it touches nothing topology-specific:
        it hands ``self`` to the Trainer, which only ever calls forward, backward
        and get_parameters. A chain and a branching network train identically.

        Args:
            data: training data (DolceSet)
            loss: loss function used to compare predictions and targets.
                Required: a general network has no sensible default, and
                defaulting to a classification loss would silently do the wrong
                thing for regression.
            n_epochs: number of passes over the (sampled) data
            sampler: strategy used to split data into (mini-)batches each epoch.
                Defaults to MiniBatchIteratingSampler().
            optimizer: optimizer used to update the layers' parameters.
                Defaults to GradientDescent().
            **run_kwargs: forwarded to Trainer.run (tol, max_steps,
                track_full_loss)

        Returns:
            self, with the run recorded in ``self.history``
        """

        # F1. Resolve the strategy defaults here rather than in the signature.
        #     Default arguments are evaluated once, at import, so every network
        #     trained with defaults would share one optimizer instance — and,
        #     once optimizers carry momentum buffers, leak momentum between
        #     unrelated networks.
        if optimizer is None:
            optimizer = GradientDescent()

        # F2. Hand the pieces to the loop and record the run.
        trainer = Trainer(model=self, loss=loss, optimizer=optimizer, sampler=sampler)
        self.history = trainer.run(data, n_epochs=n_epochs, **run_kwargs)
        return self

    def predict(self, X: np.ndarray):
        """
        Computes the model's predictions for ``X``.

        Args:
            X: input features (numpy array)

        Returns:
            The network's output with ``training=False`` — a single array for a
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
