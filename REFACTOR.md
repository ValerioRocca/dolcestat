# Refactor: aligning `dolcestat` on the `neural_networks` design

**Status:** draft for review. No `.py` / `.ipynb` file has been changed.
**Goal:** make `neural_networks`' design vocabulary the codebase-wide one, and
retire the parallel vocabulary in `optimization` / `linear_models`.

---

## 1. What "the `neural_networks` style" actually is

Six conventions, extracted from the submodule as written:

| # | Convention | Where it shows up |
|---|---|---|
| C1 | **Polymorphic strategy objects, not string flags** | `Loss`, `ActivationFunction`, `Sampler`, `ParametersInitializer` ABCs |
| C2 | **`forward` / `backward` pairs**, with `forward` caching what `backward` needs, gated by `training: bool` | `Layer`, `ActivationFunction`, `Loss`, `NeuralNetwork` |
| C3 | **`Parameters` (value + grad) as the unit of trainable state** | [parameters.py](src/dolcestat/neural_networks/parameters.py) |
| C4 | **Composition over god-objects** — the container only orchestrates | `Sequential` delegates to `Layer` / `Loss` / `Sampler` / optimizer |
| C5 | **Strategies injected at call time**, not baked into a constructor as strings | `Sequential.fit(data, loss=…, sampler=…, optimizer=…)` |
| C6 | **Explicit bias parameter**, not a folded ones-column | `DenseLayer.b` vs `BaseOptimizer`'s `np.column_stack` |

Plus the surface style: module docstrings (`"""Stores …"""`), Google-style
`Args:` / `Returns:` docstrings, type hints on signatures, numbered step
comments.

### 1.1 The counter-claim: where `neural_networks` is *not* the better model

Adopt the design, not the defects. Three places where copying the newer module
verbatim would be a **regression**:

- **It has no input validation at all.** `optimization`, `linear_models`,
  `neighbors` and `trees` each ship an `input_validation.py`; `neural_networks`
  ships none. The target style is "NN composition **+** the existing validation
  rigor" — `neural_networks` should *gain* a validation module, not the reverse.
- **It has no training history.** `BaseOptimizer` records weights/predictions/loss
  every iteration so `OptimizerAnalyzer` can plot convergence. `Sequential.fit`
  records nothing — it even computes `loss_value`
  ([networks.py:146](src/dolcestat/neural_networks/networks.py#L146)) and throws
  it away. Unification must port this *forward*, not drop it.
- **It carries live bugs** (D18–D21 below), one of which becomes materially worse
  the moment optimizers hold state.

---

## 2. The load-bearing decisions

### D1 — Optimizer protocol: `fit()`-owns-everything → `step(parameters)`

**Current.** `BaseOptimizer` subclasses own the training data, the loop, the
batching, the gradient math, *and* the history
([base.py:62-104](src/dolcestat/optimization/base.py#L62-L104),
[gradient_descent.py:85-146](src/dolcestat/optimization/gradient_descent.py#L85-L146)).
`neural_networks` inverts this: `Sequential.fit` owns the loop and calls
`optimizer.step(self.get_parameters())`
([networks.py:173](src/dolcestat/neural_networks/networks.py#L173)) — a method
that exists nowhere in the hierarchy, which is why that line raises today.

**Decision.** Define an `Optimizer` ABC whose sole abstract method is
`step(parameters)`: consume each `Parameters.grad`, mutate each
`Parameters.value` in place. Batching, gradient computation and history move out
to the caller.

**Rationale.** This is the single change that makes one optimizer reusable across
a GLM and a network, and it is the reason all the other decisions cascade. It
also collapses `BaseOptimizer`'s six responsibilities down to one.

**Sub-decision — who calls `zero_grad`?** Today `Sequential.fit` zeroes the
parameters itself ([networks.py:136-137](src/dolcestat/neural_networks/networks.py#L136-L137)).
PyTorch puts `zero_grad()` on the optimizer. Recommend **keeping it on the
training loop**: the optimizer then never needs to know the parameter list
between `step` calls, which keeps it stateless w.r.t. the model.

---

### D2 — Loss: free functions + string dispatch → `Loss` objects

**Current.** `loss_and_activation.py` exposes `compute_{mse,bce}_{loss,gradient,hessian}`,
dispatched through four parallel `if/elif` blocks on `self.loss_function`
([base.py:147-177](src/dolcestat/optimization/base.py#L147-L177)). Adding a third
loss means editing four methods plus `_infer_loss_function` plus
`validate_loss_function`.

**Decision.** Replace with `Loss` objects. Requires writing
`MeanSquaredError(Loss)` — `neural_networks/losses.py` currently only has
`BinaryCrossEntropy`.

**Rationale.** C1. A new loss becomes one new class and zero edits elsewhere.

**⚠ The semantic gap that makes this non-trivial.** The two `backward`s do not
return the same thing:

| | returns | signature |
|---|---|---|
| `compute_mse_gradient` | **dL/dw** — chain rule through the linear layer already folded in | `(X, y_true, y_pred)` |
| `Loss.backward` | **dL/dy_pred** — stops at the model output | `()` |

They are only reconcilable by treating the GLM as a one-layer network, so that
dL/dw comes from `DenseLayer.backward` instead of from the loss (see D11). Any
plan that keeps `optimization` computing dL/dw directly will *not* be able to
share `Loss` classes.

---

### D3 — Activation: free functions → `ActivationFunction` objects

**Current.** `identity` / `sigmoid` are plain functions
([loss_and_activation.py:25,64](src/dolcestat/optimization/loss_and_activation.py#L25)),
attached to models as `_activation = staticmethod(identity)`
([linear_regression.py:20](src/dolcestat/linear_models/linear_regression.py#L20),
[logistic_regression.py:19](src/dolcestat/linear_models/logistic_regression.py#L19)),
and separately re-dispatched by `_apply_activation_function`.

**Decision.** Use `LU()` / `Sigmoid()` from `neural_networks/activations.py`.
Delete the function versions and the `staticmethod` indirection.

**Rationale.** C1/C2 — the object form also carries `backward`, which the plain
functions never had, so the model no longer needs a separate gradient path.

**Naming sub-decision.** `LU` = *Linear Unit* — a plain linear mapping, i.e. ReLU
without the negative-side zeroing. The name is a deliberate coinage that pairs
with `ReLU`. **Resolved → §10.3: keep `LU`.**

---

### D4 — Bias: folded ones-column → explicit `Parameters`

**Current.** `BaseOptimizer.load_training` appends a constant column
([base.py:97](src/dolcestat/optimization/base.py#L97)) and the bias is the last
weight. `DenseLayer` keeps `self.b` separate. **The codebase already flags this
itself** at [networks.py:188-191](src/dolcestat/neural_networks/networks.py#L188-L191).

**Decision.** Adopt the explicit bias (C6).

**Rationale.** Removes three separate re-implementations of "append the ones
column at predict time" — [base.py:186](src/dolcestat/optimization/base.py#L186),
[abstract.py:87](src/dolcestat/linear_models/abstract.py#L87),
[perceptron.py:89](src/dolcestat/neural_networks/perceptron.py#L89) — plus a
fourth in `LinearRegression`'s closed form
([linear_regression.py:32](src/dolcestat/linear_models/linear_regression.py#L32)).
Each is a chance for train/predict skew.

**⚠ Breaks a documented public contract.** `model.weights` is currently the
concatenated `[w1, w2, b]`, and notebooks 02 and 03 print it under exactly that
label. **Resolved → §10.4: accept the notebook edits**, no compatibility property.

---

### D5 — Weight state: history matrix → `Parameters` + a separate recorder

**Current.** State *is* the history: `self.weights` is an `(n_iters, n_params)`
matrix grown by `np.vstack`, and "the current weights" means "the last row"
([base.py:100](src/dolcestat/optimization/base.py#L100),
[base.py:122-137](src/dolcestat/optimization/base.py#L122-L137)).

**Decision.** Split *state* from *diagnostics*. `Parameters.value` holds state;
a new `History` recorder object, owned by the training loop, holds the
per-iteration loss/weight trace.

**Rationale.** C4. Conflating the two is what forces every optimizer to inherit
the storage machinery, and it is why momentum is currently reverse-engineered
from the trace (D6). As a bonus, `Sequential` finally gets loss curves (§1.1) and
`OptimizerAnalyzer` can point at the recorder instead of at the optimizer.

**Note.** `OptimizerAnalyzer` reaches into `optimizer.get_loss()`,
`optimizer.flavor` and `optimizer.loss_function`
([analyzer.py:31,62,79](src/dolcestat/optimization/analyzer.py#L31)) — all three
accessors change. Its `gd` alias ([analyzer.py:24-27](src/dolcestat/optimization/analyzer.py#L24-L27))
is already marked "backward-compatible" and is used by notebook 02; good moment
to retire it.

---

### D6 — Momentum: derived from history → explicit velocity buffers

**Current.** Momentum is recovered as `momentum_rate * (prev_weights - weights)`
by indexing back into the trace
([gradient_descent.py:113-125](src/dolcestat/optimization/gradient_descent.py#L113-L125)).
Correct as written — expanding Polyak gives the standard heavy-ball
`w - αg + β(wₜ - wₜ₋₁)` — but it only works *because* the full history exists.

**Decision.** Hold an explicit velocity buffer per parameter inside the
optimizer, keyed by parameter identity.

**Rationale.** D5 removes the trace that the current trick depends on, and
`Parameters` objects carry no history of their own. Something must hold velocity;
the optimizer is the right owner (keeping `Parameters` a dumb container usable by
*any* optimizer).

**⚠ Nesterov changes meaning.** The current implementation is *true look-ahead*:
it shifts the weights **before** computing the gradient
([gradient_descent.py:113-117](src/dolcestat/optimization/gradient_descent.py#L113-L117)).
A `step()` called *after* backprop structurally cannot do that — `.grad` already
reflects the unshifted weights. Two options:

1. **PyTorch-style correction form** (`v = βv + g; update = g + βv`). Same
   asymptotics, no API change. **Recommended.**
2. **Two-phase optimizer** (`pre_forward(params)` / `step(params)`), preserving
   true look-ahead at the cost of a more complex protocol every training loop
   must honour.

Whichever is chosen, say so in the docstring — notebook 04 teaches the
difference between Polyak and Nesterov, so a silent substitution is a
*pedagogical* regression even where it is a numerical no-op.

---

### D7 — Batching: `flavor` strings → injected `Sampler`

**Current.** `flavor="batch"|"sgd"|"mini_batch"` + `batch_fraction` +
a `mini_batch_size` derived in `load_training` + `utils.sample_rows_Xy`
([gradient_descent.py:95-106](src/dolcestat/optimization/gradient_descent.py#L95-L106)).
`neural_networks/samplers.py` already solves this properly.

**Decision.** Delete `flavor`, `batch_fraction`, `mini_batch_size` and
`optimization/utils.py`; inject a `Sampler`. Mapping:
`"batch"` → `BatchSampler`, `"sgd"` → `MiniBatchSampler(batch_size=1)`,
`"mini_batch"` → `MiniBatchSampler`.

**Rationale.** C1/C5. Also removes an odd coupling where the *optimizer* needed
to know the dataset row count just to size a batch.

**Two sub-decisions.** *(both resolved → §10.7)*
- *Absolute vs relative batch size.* `optimization` uses `batch_fraction` (0-1),
  `neural_networks` uses `batch_size` (rows). **Resolved: absolute `batch_size`.**
  Not a free swap — notebook 04 passes `batch_fraction=0.2`.
- *Iteration semantics.* `optimization` counts `n_iters` **iterations**;
  `Sequential` counts `n_epochs` **epochs × batches**. Convergence checks and
  `tol` are defined against the former. Notebook 04 prints
  `len(plain.get_loss())` as "iterations until convergence" in four places — those
  numbers change meaning under an epoch-based loop.

---

### D8 — `DolceSet` cannot be constructed from arrays

**Current.** `DolceSet.__init__` takes no arguments
([core.py:24](src/dolcestat/preprocessing/core.py#L24)); it is populated via
`load_from_polars_dataframe`. `samplers.py` therefore works around it by building
an empty instance and assigning fields directly
([samplers.py:10-30](src/dolcestat/neural_networks/samplers.py#L10-L30)) — the
docstring says as much.

**Decision.** Give `DolceSet` an array constructor (`__init__(X=None, y=None, …)`
or a `from_arrays` classmethod) and delete `_make_batch`.

**Rationale.** This is a wart *inside* the module we are standardising on, and it
is load-bearing for the refactor: every batch allocation goes through it. It also
makes the library testable without a Polars round-trip.

---

### D9 — `RosenblattPerceptron` inherits from the wrong axis

**Current.** `class RosenblattPerceptron(BaseOptimizer)`
([perceptron.py:11](src/dolcestat/neural_networks/perceptron.py#L11)) — a
**model** subclassing an **optimizer**, purely to inherit `load_training`,
history and `predict`.

**Decision.** Re-express it as a model. Cleanest: a `Sequential` with one
`DenseLayer(activation=Step(), …)` trained by a `PerceptronRule` optimizer —
which is exactly what the perceptron rule *is*, and it demonstrates the new
abstractions on the simplest possible network.

**Rationale.** The current inheritance is an artefact of `BaseOptimizer` being a
god-object; once D1/D5 strip it down there is nothing left to inherit. Note the
class docstring already explains the folded-bias inheritance
([perceptron.py:21-22](src/dolcestat/neural_networks/perceptron.py#L21-L22)) — that
prose goes away with D4.

---

### D10 — Newton does not fit `step(parameters)`

**Current.** `NewtonMethod` needs `X` and a **single joint Hessian** over one
flat parameter vector
([newton.py:31-37](src/dolcestat/optimization/newton.py#L31-L37)).

**The problem.** `step(parameters)` receives a *list of independent arrays* and
no `X`. A per-parameter Hessian is not the same object as a joint one, and
`Loss.backward()` deliberately does not expose `X`.

**Decision — recommended.** Do **not** force Newton into the `Optimizer` ABC.
Keep it as a GLM-only fitting strategy alongside the closed form, and let
`Optimizer.step` mean "first-order update". Alternatives, both worse: a
flatten/unflatten adapter (real work, only ever exercised by one-layer models), or
dropping Newton (it is a teaching centrepiece of notebook 04).

**Rationale.** Second-order methods over a layered parameter list is a genuinely
different design problem. Pretending otherwise buys a uniform signature and pays
for it with an abstraction that lies.

---

### D11 — The unifying insight: a GLM *is* a one-layer network

**Decision.** Model the GLMs as
`LinearRegression  ≈ Sequential(DenseLayer(LU(),      n, 1)) + MeanSquaredError`
`LogisticRegression ≈ Sequential(DenseLayer(Sigmoid(), n, 1)) + BinaryCrossEntropy`

**Rationale.** This is what makes D2 tractable and what justifies the whole
exercise: the four `if/elif` blocks in `BaseOptimizer` disappear because the
activation lives on the layer and the loss gradient stops at the output.

**⚠ Four things must survive the collapse.** `GLMAbstract` is not just a wrapper:
- the **closed-form** path, which bypasses optimizers entirely
  ([linear_regression.py:27-40](src/dolcestat/linear_models/linear_regression.py#L27-L40));
- the **Newton** path (D10);
- `model.weights` as a public attribute (D4);
- `.optimization` → `OptimizerAnalyzer`
  ([abstract.py:98-108](src/dolcestat/linear_models/abstract.py#L98-L108)).

---

## 3. Package layout and dependency direction

### D12 — Where do the shared primitives live? *(decide before writing code)*

**The problem.** `losses.py`, `activations.py`, `parameters.py` and `samplers.py`
currently live under `neural_networks`, but after this refactor `linear_models`,
`optimization` and `metrics` all need them. Meanwhile
`neural_networks/networks.py` and `perceptron.py` already import *from*
`optimization`. The package dependency becomes **bidirectional**.

There is no import cycle *today* — those four modules are leaves (they import
only `numpy`/`abc`, plus `preprocessing` in `samplers`). But
`optimization.gradient_descent` importing `neural_networks.parameters` while
`neural_networks.networks` imports `optimization.gradient_descent` is exactly the
shape that becomes a hard `ImportError` the first time someone adds a
convenience re-export to `neural_networks/__init__.py`.

**Decision.** Promote the shared primitives out of `neural_networks` into a
neutral home — `dolcestat.core` (new) or `dolcestat.optimization` — leaving
`layers.py` / `networks.py` / `perceptron.py` behind as the genuinely
network-specific part. Dependencies then flow one way:
`core ← optimization ← {linear_models, neural_networks}`.

**Rationale.** Conceptually, "a loss function" is not a neural-network concept;
it is shared vocabulary. Also avoids `linear_models` importing from
`neural_networks`, which reads backwards to anyone learning from the source.

**Knock-on:** the README module table and the post-commit hook both enumerate
`src/dolcestat/*/`, so a new package needs a README row.

### D13 — `loss_and_activation.py` has three external dependents

Deleting it (D2/D3) breaks:
1. [metrics/analyzer.py:4](src/dolcestat/metrics/analyzer.py#L4) — `compute_bce_loss`
2. [linear_regression.py:4](src/dolcestat/linear_models/linear_regression.py#L4) — `identity`
3. [logistic_regression.py:2](src/dolcestat/linear_models/logistic_regression.py#L2) — `sigmoid`

`metrics` is the awkward one: it is otherwise model-agnostic and should not
depend on the optimizer package at all. It should call
`BinaryCrossEntropy().forward(...)` from the D12 neutral home.

---

## 4. Public API consistency

### D14 — `predict` returns two different kinds of thing

| | argument | returns |
|---|---|---|
| `GLMAbstract.predict` | `DolceSet` | `PredictionAnalyzer` |
| `Sequential.predict` | raw `np.ndarray` | raw `np.ndarray` |
| `BaseOptimizer.predict` | `DolceSet` | raw `np.ndarray` |
| `RosenblattPerceptron.predict` | `DolceSet` | raw `np.ndarray` |

**Decision.** One contract: `predict(data: DolceSet) -> PredictionAnalyzer`.

**Rationale.** The analyzer-returning form is the library's most distinctive
idea (notebooks 02/03/06 lean on `model.predict(data).r2()`), and it is
inconsistently applied. Here the *older* module has the better convention.

### D15 — `fit` return value

`GLMAbstract.fit` returns `self` — notebook 03 chains
`LogisticRegression("newton").fit(data)`. `Sequential.fit` returns `None`.
**Decision.** Return `self` everywhere.

### D16 — Data lifecycle: `load_training` vs data-at-`fit`

`BaseOptimizer` and `DecisionTree` **independently duplicate** the same
`data=None` + `load_training()` + `is_training_loaded` / `can_train` lifecycle
([base.py:57-104](src/dolcestat/optimization/base.py#L57-L104),
[decision_tree.py:55-76](src/dolcestat/trees/decision_tree.py#L55-L76)).
`Sequential` just takes data in `fit()`.

**Decision.** Adopt data-at-`fit`.

**Rationale.** It deletes a duplicated lifecycle, and it removes the reason for
the "optimizer must be data-less" rule in
[linear_models/input_validation.py:12-18](src/dolcestat/linear_models/input_validation.py#L12-L18) —
a validation rule that exists purely to guard against a footgun the new design
does not have. **Breaks notebook 04**, which constructs `GradientDescent(data=data)`
eight times.

### D17 — Target shape: 1-D vs 2-D column

`DolceSet.y` is 1-D (`.flatten()` at [core.py:51](src/dolcestat/preprocessing/core.py#L51));
a single-output layer emits `(n, 1)`. `BinaryCrossEntropy.forward` papers over the
mismatch with a reshape
([losses.py:57-60](src/dolcestat/neural_networks/losses.py#L57-L60)) — the comment
there exists precisely because this was never decided.

**Decision.** Pick a canonical target shape and enforce it once at the
`DolceSet` boundary, then delete the per-loss reshape. (Minor companion: `metrics`
calls its vectors `y_fit`, `neural_networks` calls them `y_pred` — unify.)

---

## 5. Defects to fix in `neural_networks` *before* it becomes the template

These are live bugs in the module being promoted. Fix them first, or the refactor
propagates them.

### D18 — Mutable default arguments *(gets worse under D6)*

```python
def fit(self, data, loss: Loss = BinaryCrossEntropy(), n_epochs: int = 100,
        sampler: Sampler = MiniBatchIteratingSampler(),
        optimizer: BaseOptimizer = GradientDescent()) -> None:
```
[networks.py:107-114](src/dolcestat/neural_networks/networks.py#L107-L114) — these
are evaluated **once, at import**. Every `Sequential` trained with default
arguments shares *one* `GradientDescent` instance. Harmless while optimizers are
stateless; the moment D6 adds velocity buffers, momentum leaks between unrelated
networks. Same pattern at
[layers.py:47](src/dolcestat/neural_networks/layers.py#L47) (`HeInitializer()`,
currently harmless).

**Decision.** `None` sentinels, resolved inside the body. Separately: defaulting a
*general* network's loss to `BinaryCrossEntropy` silently does the wrong thing for
regression — make `loss` required.

### D19 — Sigmoid+BCE gradient silently vanishes at saturation

`BinaryCrossEntropy.backward` divides by `p(1-p)`
([losses.py:73-74](src/dolcestat/neural_networks/losses.py#L73-L74)) and
`Sigmoid.backward` multiplies it straight back
([activations.py:97](src/dolcestat/neural_networks/activations.py#L97)). Inside the
clip range these cancel exactly. **Outside it they do not:** `forward` caches the
*clipped* `y_pred` ([losses.py:55,64](src/dolcestat/neural_networks/losses.py#L55))
while `Sigmoid` caches the *unclipped* `A`, so a saturated unit yields a gradient
scaled by `A(1-A) / (A_clip(1-A_clip))` → 0 instead of the correct `(A - y)/n`.

The `optimization` module has this right: `compute_bce_gradient` uses the fused,
stable `(p - y)` form
([loss_and_activation.py:43-50](src/dolcestat/optimization/loss_and_activation.py#L43-L50)).

**Decision.** Add a fused `SigmoidBinaryCrossEntropy` (PyTorch's
`BCEWithLogitsLoss`) and prefer it. Another case where the older module is right.

### D20 — `Parameters.grad` can be integer-typed

`__init__` does `np.zeros_like(value)`
([parameters.py:18](src/dolcestat/neural_networks/parameters.py#L18)) — no
`dtype`. An integer-valued initializer yields an **integer** grad buffer, so
`W.grad += …` truncates every accumulation to zero. `zero_grad()` gets this right
with `dtype=float` ([parameters.py:22](src/dolcestat/neural_networks/parameters.py#L22)),
so the bug only bites before the first `zero_grad`. **Decision.** Add `dtype=float`.

### D21 — Smaller items

- `Sequential.fit` calls `self.get_parameters()` twice per batch
  ([networks.py:136,173](src/dolcestat/neural_networks/networks.py#L136)), rebuilding
  the list each time — hoist it.
- `loss_value` is computed and never used
  ([networks.py:146](src/dolcestat/neural_networks/networks.py#L146)) — this is the
  natural hook for the D5 recorder.
- The mini-batch samplers index `y[indices]` unguarded
  ([samplers.py:87,118](src/dolcestat/neural_networks/samplers.py#L87)) and raise an
  opaque `TypeError` when `data.can_train` is `False`.
- `MiniBatchSampler` will raise if `batch_size > n_samples` (`replace=False`);
  `MiniBatchIteratingSampler` handles the short tail correctly.
- `gradient_descent.py:128` uses `== None` instead of `is None`; if
  `momentum_type` ever escaped validation, `updated_weights` would be unbound.

---

## 6. Style sweep (mechanical, low risk)

| Item | `neural_networks` | Rest of codebase |
|---|---|---|
| Docstrings | Google `Args:` / `Returns:` | NumPy `Parameters\n----------` (GD, Newton) or absent (`trees`, `neighbors`, `metrics`) |
| Type hints | present | absent |
| Module docstring | `"""Stores …"""` | absent |
| Branching | `if/elif` | `match/case` in [neighbors/base.py:20,55](src/dolcestat/neighbors/base.py#L20) |
| Numbered step comments | yes | yes — already shared |

**Decision.** Google-style + type hints + module docstrings everywhere; pick one
of `if/elif` vs `match/case`. `black`/`isort` are already enforced by the Stop
hook in `.claude/settings.json`, so no formatter work is needed.

---

## 7. Suggested sequencing

Ordered so that nothing is built on a foundation that is about to move.

| Phase | Content | Risk |
|---|---|---|
| **0** | D18–D21: fix the bugs in `neural_networks` | none — no external callers |
| **1** | D12: decide the package layout; D8: `DolceSet` array constructor | low, mechanical |
| **2** | D2/D3: write `MeanSquaredError`, `SigmoidBinaryCrossEntropy`; keep `loss_and_activation.py` as a thin shim | low — additive |
| **3** | D1/D5/D6: `Optimizer` ABC + `step()` + `History`; rewrite `GradientDescent` on top | **high** — the core change |
| **4** | D4/D11/D16: GLMs as one-layer networks, explicit bias, data-at-`fit` | **high** — breaks notebooks 02/03/04 |
| **5** | D9/D10: perceptron re-homed; Newton settled as a GLM strategy | medium |
| **6** | D13/D14/D15/D17: API consistency sweep; delete `loss_and_activation.py`, `optimization/utils.py` | medium |
| **7** | D6 note, §6: docs, docstrings, notebook prose | low |

The natural stopping point if this proves too large is **end of phase 3**:
`optimizer.step()` works, `Sequential` trains, and `GLMAbstract` still drives the
old path. Phases 4+ are what actually unify the two worlds.

---

## 8. Breakage inventory

**Notebooks** (all prose lives here per the project convention, so each break is a
docs edit too):
- `02_linear_regression` — `GradientDescent(loss_function=…, flavor=…)` kwargs (D2/D7); `optimization.gd` alias (D5); `weights [w1, w2, b]` label (D4)
- `03_logistic_regression` — `weights [w1, w2, b]` label (D4); chained `.fit()` (D15)
- `04_optimization` — heaviest hit: 8 × `GradientDescent(data=…)` (D16), `flavor` / `batch_fraction` (D7), `len(get_loss())` as an iteration count (D5/D7), `NewtonMethod(data=…)` (D10/D16), `OptimizerAnalyzer` construction (D5)
- `08_perceptron` — `RosenblattPerceptron(data=…)` + `.fit()` + `.predict(data)` (D9/D16)

**Public API:** `model.weights` layout (D4) · `optimizer.get_loss()` / `get_weights()` (D5) · every `GradientDescent` / `NewtonMethod` keyword (D2/D7/D16) · `predict` return type (D14) · `dolcestat.optimization.loss_and_activation` module (D13).

**Not touched by the optimizer work, but in scope for §6 style and D14/D15:**
`trees`, `neighbors`, `metrics`, `preprocessing`.

---

## 9. Open questions

1. **D12** — new `dolcestat.core` package, or promote into `dolcestat.optimization`?
2. **D4** — keep `weights` as a compatibility property, or edit notebooks 02/03?
3. **D6** — PyTorch-style Nesterov correction (simple, approximate) or a two-phase
   optimizer API (faithful, more complex)?
4. **D7** — `batch_size` (conventional) or `batch_fraction` (scale-free, current)?
   And do we keep iteration-based or move to epoch-based convergence?
5. **D10** — is "Newton is a GLM strategy, not an `Optimizer`" acceptable, or is a
   uniform optimizer interface worth the flatten/unflatten adapter?
6. **Scope** — stop after phase 3 (make `step()` work, leave GLMs alone), or go all
   the way to phase 7?
7. **Tests** — there are none in the repo (`.pytest_cache` exists but no test
   files). A refactor of this size across four packages, with no notebook
   execution in CI, has no safety net. Worth adding characterisation tests that
   pin current numeric output *before* phase 3?

*(All resolved → §10.)*

---

## 10. Review round 1 — answers and resolutions

### 10.1 — D1: does `GradientDescent.fit()` call `self.step()`?

**Conceptually yes; structurally the loop should not live on the optimizer.**

`fit` genuinely *is* "loop { sample → forward → loss → backward → step }". But for
`GradientDescent.fit()` to run that loop it needs the data, the loss, the
activation and the forward map — which is precisely the `load_training` +
`_apply_activation_function` + `_compute_gradient` machinery that D1 exists to
delete. Making `fit` call `self.step()` keeps the god-object and just adds a
method to it.

**Decision: extract the loop into a `Trainer`.** One implementation, used by
everything:

```python
class Optimizer(ABC):
    @abstractmethod
    def step(self, parameters: list) -> None:
        """Consume each Parameters.grad; update each Parameters.value in place."""

class Trainer:
    """Owns the training loop. The only place forward/backward/step is sequenced."""
    def __init__(self, model, loss, optimizer, sampler): ...
    def run(self, data, n_epochs) -> History: ...
```

- `Sequential.fit(data, …)` → builds a `Trainer` and returns `self` (D15).
- `GLMAbstract.fit(data)` → same `Trainer`, over its internal one-layer model (§10.11).
- Notebook 04 → drives a `Trainer` directly and plots `trainer.history`.

**Why this is better than a `fit()` shim.** Notebook 04 is *about* optimizers, and
an explicit `Trainer` makes the loop it teaches visible instead of hiding it
inside `GradientDescent`. Optimizers become genuinely stateless w.r.t. the model
(they see only a parameter list), which is what lets one optimizer serve a GLM,
a perceptron and a deep network.

**Consequence:** `Optimizer` has **no** `fit`. `BaseOptimizer.fit` is deleted, not
reimplemented — so the abstract-`fit` contract at
[base.py:179-181](src/dolcestat/optimization/base.py#L179-L181) goes away, and
notebook 04's `opt.fit()` calls all become `Trainer` calls. Add this to the §8
breakage inventory.

---

### 10.2 — D2: add `Loss.get_gradient(X, y_true, y_pred)` alongside `backward()`?

**Verdict: no for the general case — but the instinct behind it is right, and it
survives in two narrower places.**

The proposal works, and it would let phases 2–3 land without touching the GLMs.
The problem is what it puts inside `Loss`:

- `compute_mse_gradient` is `(2/n)·Xᵀ(p − y)`. That is only correct for a
  **linear model with identity activation**. `compute_bce_gradient` is only
  correct for **linear + sigmoid**. So `get_gradient` would silently encode an
  assumed architecture inside the loss — the exact coupling D2 removes. A loss
  function does not know what produced its input, and should not.
- Every future loss would owe a second, architecture-specific derivation. The
  mixture-density work in `TODO.md` has no meaningful "dL/dw for the linear
  case" at all.
- `backward()` and `get_gradient()` would have to stay mutually consistent
  forever, with nothing enforcing it.

**And the cheaper equivalent already exists.** Your own framing — a GLM is a
1-depth network — is the argument against needing the method:

```python
# layers.py:98 — this IS get_gradient
self.W.grad += self.X.T @ grad_z
```

`DenseLayer.backward` already computes `Xᵀ·grad_z`. Composed with
`Sigmoid.backward` (`grad_z = dL/dA · A(1−A)`) and `BCE.backward`
(`dL/dA = (A−y)/(A(1−A)n)`), it reduces to `Xᵀ(A−y)/n` — `compute_bce_gradient`
exactly. So `get_gradient` would reimplement a code path we are keeping anyway.

**Where the idea is genuinely needed — keep it, but on the right object and
under a name that admits its scope:**

1. **Newton (D10).** Second-order updates need `X` *and* the loss curvature, and
   `backward()` structurally cannot supply either. This is real, and §10.11
   places it.
2. **Fused stable paths (D19).** A fused sigmoid+BCE would legitimately expose a
   direct `dL/dZ`. You've chosen to keep them separate (§10.19), so this one
   does not apply here.

**If you want the migration scaffold anyway:** adding `get_gradient` as an
explicitly temporary bridge during phases 2–3 is defensible — it decouples
"introduce `Loss` objects" from "rewrite the GLMs". Mark it `# TRANSITIONAL —
delete in phase 4` so it does not outlive its purpose.

---

### 10.3 — D3: `LU` is Linear Unit *(correction accepted)*

My note that it "reads as a truncated ReLU" was wrong. `LU` = **Linear Unit**, a
plain linear mapping — ReLU without the negative-side zeroing — and the name
deliberately parallels `ReLU`.

**Decision: keep `LU`.** The `LU` / `ReLU` pairing is self-consistent and reads
well in a library whose point is to make the maths legible. Add one clarifying
line to the class docstring (`"""Linear Unit: identity activation, i.e. ReLU
without rectification."""`) so the name is unambiguous to a reader meeting it
first. D3 otherwise stands: the plain `identity` / `sigmoid` functions in
`loss_and_activation.py` go away.

---

### 10.4 — D4: accept the notebook edits

**Decision: no compatibility property.** `weights` stops being the concatenated
`[w1, w2, b]`.

**Replacement API:** expose the two parameters separately —
`model.weights` (the `(n_features, n_outputs)` matrix) and `model.bias` —
reading straight through to the underlying layer's `Parameters.value`.
`coef_`/`intercept_` is the sklearn spelling, but `weights`/`bias` matches the
vocabulary already used throughout `neural_networks` (C6), which is the point of
the refactor.

**Notebook edits required:**
- `02_linear_regression`, cells 2–3: `print("weights [w1, w2, b]:", …)`
  → print `model.weights` and `model.bias` on separate lines.
- `03_logistic_regression`, cells 2 and 4: same, including the
  GD-vs-Newton weight comparison.

Small pedagogical upside: the notebooks stop having to explain that the last
weight is secretly the intercept.

---

### 10.7 — D7: absolute `batch_size`, and a fix for iteration semantics

**Batch size: absolute.** `Sampler` keeps `batch_size: int`; `batch_fraction`,
`mini_batch_size` and the `load_training`-time derivation are deleted. Notebook
04's `batch_fraction=0.2` on 200 rows becomes `batch_size=40`.

**Iteration semantics — the underlying problem.** Today one `n_iters` iteration =
one gradient step, and `tol` compares consecutive losses. Under an epoch-based
loop, one epoch = *many* steps, so `n_iters`, `tol` and notebook 04's
`len(get_loss())` all silently change meaning.

**Proposed resolution — record per step, converge per epoch.**

| | unit | who counts it |
|---|---|---|
| **step** | one `optimizer.step()` = one batch | `History`, per-entry |
| **epoch** | one full `sampler.sample(data)` pass | `Trainer.run(n_epochs=…)` |

1. **`History` records per step**, each entry tagged `(epoch, step)`. Nothing is
   lost relative to today, and `len(history.steps)` is the direct successor to
   `len(optimizer.get_loss())`.
2. **Convergence is checked per epoch, never per step.** Under mini-batching the
   per-step loss is noisy, so a `tol` on consecutive steps fires spuriously —
   which is exactly why the current code pays for a **full-batch** forward pass
   every iteration purely to keep the curve comparable across flavors
   ([gradient_descent.py:132-139](src/dolcestat/optimization/gradient_descent.py#L132-L139)).
   Moving the check to the epoch boundary makes that cost principled instead of
   incidental.
3. **Record two losses:** `batch_loss` (free — already computed in the forward
   pass, and currently thrown away at
   [networks.py:146](src/dolcestat/neural_networks/networks.py#L146)) and
   `epoch_loss` (one extra full-batch forward per epoch, behind a
   `track_full_loss: bool = True` flag so large problems can opt out).
   `OptimizerAnalyzer` plots `epoch_loss` by default and `batch_loss` on request
   — which makes mini-batch gradient noise *visible*, a better teaching artefact
   than today's smoothed curve.
4. **`Trainer.run(data, n_epochs, tol=…, max_steps=None)`.** `n_iters` is retired;
   `max_steps` is the escape hatch for a hard budget.

**The property that makes this safe:** with `BatchSampler`, one epoch is exactly
one step, so `epochs == steps` and every number notebook 04 currently prints for
`flavor="batch"` is reproduced **exactly**. Only the `sgd` and `mini_batch` cells
change, and they change in the direction of being more honest.

---

### 10.11 — D11: how do Newton and the closed form survive if a GLM is a `Sequential`?

This is the right thing to be puzzled by; my original write-up asserted the
collapse without showing the mechanism.

**The resolving idea: the model is not the fitting algorithm.**

`Sequential` / `DenseLayer` define only the **parameterisation and the forward
map** — "there is a weight matrix `W`, a bias `b`, and prediction is
`activation(XW + b)`". *How you find good values for `W` and `b`* is a separate
concern. The common currency is the `Parameters` objects: **any** fitting
strategy is free to compute values however it likes, as long as it writes them
into `layer.W.value` / `layer.b.value`. `predict` then works identically
regardless of which strategy filled them.

So a GLM has one model and three interchangeable fitting strategies:

| strategy | mechanism | touches `Trainer`? | gradients? |
|---|---|---|---|
| gradient descent | `Trainer` loop → `optimizer.step` | yes | yes |
| Newton | own loop; joint Hessian on `[X \| 1]` | no | own |
| closed form | one shot, `(XᵀX)⁻¹Xᵀy` | no | none |

All three end with the same two lines:

```python
layer.W.value = w_flat[:-1].reshape(n_features, 1)
layer.b.value = w_flat[-1:]
```

**The friction this exposes — worth naming.** Newton and the closed form
*naturally* work on the bias-augmented matrix `[X | 1]` and produce a single flat
vector; D4's explicit bias means they must split it on the way out. That is the
real cost of D4, it is unavoidable, and it is contained: **one shared
`_unpack_flat_weights(w_flat, layer)` helper**, used by exactly two call sites.
Cheap, and it keeps the augmented-matrix maths (which is how the textbook
derivation reads) intact where it belongs.

**Sub-decision: `LinearRegression` should *have* a `Sequential`, not *be* one.**
Composition, not inheritance. Inheriting would drag in
`fit(data, loss, n_epochs, sampler, optimizer)` and `predict(X) -> ndarray`,
whereas the GLM needs `fit(data)` / `predict(data) -> PredictionAnalyzer` (D14)
and its `optimizer="closed form"` string API. Composition reuses the machinery
while keeping the GLM's teaching-oriented surface — and it keeps "a GLM is a
1-depth network" as an *illuminating analogy* rather than a load-bearing
`isinstance` relationship.

---

### 10.12 — D12: confirmed, `dolcestat.core`

Moving out of `neural_networks` into a new `dolcestat/core/`:
`parameters.py`, `losses.py`, `activations.py`, `samplers.py` — plus the new
`optimizer.py` (the `Optimizer` ABC, §10.1), `trainer.py` and `history.py`.

Staying in `neural_networks`: `layers.py`, `networks.py`, `perceptron.py`.

Resulting one-way dependency:
`core ← optimization ← {linear_models, neural_networks}`, with `metrics`
depending on `core` only (§10.13).

**⚠ Repo-convention knock-on.** The post-commit hook enumerates
`src/dolcestat/*/` and warns for any package missing from `README.md`, and the
README table maps *module → notebook*. `core` is infrastructure with no natural
notebook of its own. Cleanest fit with the existing convention: give it a row
pointing at [04 · optimization](notebooks/04_optimization.ipynb), which is where
losses, samplers and the training loop are already taught.

---

### 10.13 — D13: a solution for `loss_and_activation.py`'s three dependents

Two of the three dissolve on their own; only `metrics` needs a real decision.

**1–2. `linear_models` (`identity`, `sigmoid`) — no replacement needed.** Under
§10.11 the activation lives on the `DenseLayer`, so `_activation =
staticmethod(identity)` and `_apply_activation_function` both disappear rather
than being re-pointed. The imports are simply deleted.

**3. `metrics` (`compute_bce_loss`) — the awkward one.** `metrics` is otherwise
model-agnostic and must not depend on `optimization`. The naive fix,
`BinaryCrossEntropy().forward(y_pred, y_true)`, has a wart: `forward` **caches**
`y_pred`/`y_true` for the backward pass
([losses.py:64-65](src/dolcestat/neural_networks/losses.py#L64-L65)), so a pure
metrics call would silently allocate and strand a backward-pass cache.

**Decision: give `Loss.forward` the `training` flag the other components already
have.**

```python
def forward(self, y_pred, y_true, training: bool = True) -> float:
    ...
    if training:          # cache only when a backward pass will follow
        self.y_pred, self.y_true = y_pred, y_true
```

`metrics` then calls `BinaryCrossEntropy().forward(y_fit, y_true, training=False)`
and imports from `dolcestat.core.losses`.

**Why this over a module-level pure function or a local reimplementation:** it is
*literally the C2 convention already used by `ActivationFunction.forward` and
`Layer.forward`*. `Loss` is currently the only forward/backward component in the
codebase **without** a `training` flag — an internal inconsistency in the module
we are standardising on. This fixes that and solves `metrics` in the same stroke,
with no duplicated log-loss maths. *(Add to the §5 defect list.)*

---

### 10.17 — D17: target shape, three options

**Option A — canonical 1-D `(n,)`; the model squeezes its output.**
*Pros:* matches `DolceSet` today ([core.py:51](src/dolcestat/preprocessing/core.py#L51)),
matches `metrics`/`trees`/`neighbors`, matches sklearn, zero notebook churn.
*Cons:* breaks down for multi-output heads — softmax/multiclass and the
mixture-density work queued in `TODO.md` — and `squeeze` is a trap when `n == 1`.

**Option B — canonical 2-D `(n, 1)`, generalising to `(n, k)`.**
*Pros:* uniform with layer outputs; multi-output needs no special-casing; deletes
the reshape hack outright; directly supports the `TODO.md` roadmap.
*Cons:* requires dropping `.flatten()` in `DolceSet`, which ripples into
`metrics`, `trees` and `neighbors` — three packages **outside** this refactor's
scope — plus every notebook that prints `y`.

**Option C — 1-D at the `DolceSet` boundary, normalised once on entry to the
training loop.** `Trainer` reshapes `(n,) → (n, 1)` once; internals are uniformly
2-D; `predict` reshapes back for `metrics`.
*Pros:* deletes the per-loss hack (the stated goal) with a blast radius of one
file; `trees`/`neighbors`/`metrics` never learn about it; `(n, k)` works for free.
*Cons:* two conventions coexist — external 1-D, internal 2-D — so the boundary
must be explicitly documented or it will confuse a reader.

**Recommendation: C.** It is the only option that achieves the actual objective
(one conversion point, hack deleted) without forcing changes on three packages
this refactor is not otherwise touching. If multi-output work later makes the
external 1-D convention itself the obstacle, C → B is a contained follow-up,
because by then there is exactly one place where the conversion happens.

---

### 10.19 — D19: keeping `Sigmoid` and `BinaryCrossEntropy` separate

**Agreed, and the reason is a good one:** in an educational library, sigmoid and
BCE composing *is* the lesson. A fused `BCEWithLogitsLoss` optimises away the
thing the reader is meant to see.

**The bug, precisely.** The cancellation is inexact only because the two classes
cache **different arrays**: `BCE.forward` caches the *clipped* `y_pred`
([losses.py:55,64](src/dolcestat/neural_networks/losses.py#L55)) while
`Sigmoid.forward` caches the *unclipped* `A`
([activations.py:83-85](src/dolcestat/neural_networks/activations.py#L83-L85)). So
`backward` divides by `A_clip(1−A_clip)` and multiplies back by `A(1−A)` —
different numbers at saturation, hence a gradient scaled toward 0.

**Solution: one source of clipping, and move it into `Sigmoid`.**

1. **`Sigmoid.forward` clips its output** to `[eps, 1−eps]` and caches the
   *clipped* `A` — the same value it returns. `A(1−A)` is then bounded below by
   `eps(1−eps)`, so `backward`'s division is bounded.
2. **`BCE.forward` caches the array it was given, unmodified**, and clips only
   locally inside the `log` for the returned scalar. `backward` then divides by
   exactly the number `Sigmoid.backward` will multiply by.
3. **Document the contract:** `Loss.backward` returns dL/dy_pred *at the
   `y_pred` it was handed*; an activation's `backward` must multiply by dA/dZ *at
   the same A*. Exact cancellation is a consequence of both honouring it.

Result: `dL/dZ = (A − y)/n` exactly, for all inputs, with the classes still fully
independent and composable. A saturated wrong prediction now yields ≈ ±1/n — the
correct, non-vanishing gradient — instead of ≈ 0.

**Residual cost, stated honestly:** the intermediate `dL/dA` can reach ~1e12 with
`eps = 1e-12`. That is far from `float64` overflow and the product is exact, so
this is safe — but it *is* the reason production libraries fuse the pair. Worth a
sentence in the notebook: keeping them separate is a deliberate
legibility-over-stability trade, and here is what fusing would buy.

**Side effect to accept:** `Sigmoid` now never outputs exactly 0 or 1. For a
probability model that is arguably correct anyway, and `BCE.forward` was already
doing it internally — this just makes it consistent and visible.

---

### 10.9 — Open questions, resolved

| # | Question | Resolution |
|---|---|---|
| 1 | D12 package layout | **New `dolcestat.core`** (§10.12) |
| 2 | D4 `weights` compatibility | **Accept notebook edits**, no property (§10.4) |
| 3 | D6 Nesterov | *Still open* — see below |
| 4 | D7 batch size / iterations | **Absolute `batch_size`; record per step, converge per epoch** (§10.7) |
| 5 | D10 Newton | **GLM fitting strategy, not an `Optimizer`** (§10.11) |
| 6 | Scope | **Go all the way to phase 7** |
| 7 | Tests | **None for now** — see risk note below |

**Still open — question 3 (Nesterov).** §10.1 changes its shape: with a `Trainer`
owning the loop, the two-phase option (`pre_forward(params)` / `step(params)`)
becomes much cheaper than I first estimated, since exactly one place calls it and
true look-ahead is preserved. Worth revisiting at phase 3 rather than deciding
now.

**Risk note on question 7.** Accepted, but recording the consequence: phases 3–4
rewrite the numerical core across four packages with no automated check that
results are unchanged. The mitigation available at zero infrastructure cost is to
**run notebooks 02/03/04/08 before and after each phase and diff the printed
numbers** — they already print weights, R² and iteration counts, which is
most of a characterisation suite. Note that §10.7 deliberately preserves exact
reproduction of the `flavor="batch"` numbers, which makes that diff meaningful.

---

## 11. Updated sequencing *(supersedes §7)*

| Phase | Content | Risk |
|---|---|---|
| **0** | D18–D21 + `Loss.forward(training=…)` (§10.13) | none |
| **1** | Create `dolcestat.core`, move the four leaf modules (§10.12); D8 `DolceSet` array constructor; README row | low |
| **2** | `MeanSquaredError`; D19 clipping fix (§10.19); D3 keeps `LU` | low, additive |
| **3** | **`Optimizer` ABC + `step()`; `Trainer`; `History`** (§10.1); rewrite `GradientDescent`; settle Nesterov | **high** |
| **4** | GLMs as composed 1-layer models (§10.11); explicit bias + `_unpack_flat_weights`; D16 data-at-`fit` | **high** |
| **5** | D9 perceptron re-homed; D10 Newton as a GLM strategy | medium |
| **6** | D13/D14/D15/D17 (option C) API sweep; delete `loss_and_activation.py`, `optimization/utils.py`, `optimization/base.py` | medium |
| **7** | §6 style sweep; rewrite notebooks 02/03/04/08 | low |
