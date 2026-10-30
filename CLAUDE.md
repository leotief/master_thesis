# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

- `code/` — Python research code (PyTorch). The experiments are run from `code/` directly; there is no package layout, `setup.py`, or test suite. Imports are flat (e.g. `from gmm import GMM`).
- `thesis/` — LaTeX source for the master's thesis (Karl Voit template). Build with `cd thesis && make pdf` (plain `make` only prints the help; uses `pdflatex` + `biber`). Output is `thesis/main.pdf`.

## Preferences

When the user says "remember" something, save it in this repo (in this file), not in Claude's memory directory.

Thesis writing:
- Never put explanations beside equations (no `&& \text{...}` annotation columns in `align`) or beside algorithm lines (no `\Comment{...}` or `\hfill` notes in `algorithmic`). Explain the steps in prose before or after.
- Write each equation row on one source line; don't wrap a row over several LaTeX lines.
- When asked to simplify, change only what is needed; keep the existing steps, order and level of detail.

## Running experiments

There is no test runner. The main entry points are:

```
cd code && jupyter notebook      # demo.ipynb — main notebook
cd code && ~/.pyenv/versions/ml/bin/python pg_demo.py   # PG identity demo
cd code && ~/.pyenv/versions/ml/bin/python hs_demo.py   # Hubbard-Stratonovich demo
```

`demo.ipynb` fits a `GMM` + sklearn `LogisticRegression`, constructs a partially observed test point, and compares all expectation estimators. The inference cell defines a local `evaluate(name, fn, n_runs)` helper that calls a zero-argument lambda.

Figures are written to `code/figures/`.

## Core architecture

The thesis investigates computing **E[σ(wᵀx) | x_obs]** — the expected logistic-regression prediction marginalised over missing features — under a learned density p(x). Four estimators are compared:

1. **`MCBaseline`** (`baselines.py`) — vanilla Monte Carlo: draw x_mis from p(x_mis | x_obs) and average σ(wᵀx).
2. **`PCPG`** (`pcpg.py`) — Pólya-Gamma / characteristic-function estimator. One class; the outer integral over ω ~ PG(1,0) and the inner Gaussian integral over u | ω ~ N(0, ω) are each done by quadrature (`"quad"`) or Monte Carlo (`"mc"`); the inner one can also be `"exact"`: the closed form of E[e^{z/2 − ωz²/2} | x_obs] from `model.z_mixture` (Gaussian per component), so only the outer rule's error is left — a diagnostic, not a method (it needs the mixture of z, which PCPG exists to avoid). The outer quadrature rule is `pg_method` (any `pg_quadrature` method, default `"golub_welsch"`), the inner one is always Gauss-Hermite. The conditional characteristic function φ(s) of the GMM is evaluated at s = −u·w_mis.

## PCPG parameters

```python
PCPG(model)                            # quadrature over both: deterministic, the main method
PCPG(model, outer="mc")                # MC over ω, Gauss-Hermite over u
PCPG(model, inner="mc")                # Golub-Welsch over ω, MC over u
PCPG(model, outer="mc", inner="mc")    # MC over both
PCPG(model, inner="exact")             # Golub-Welsch over ω, inner integral in closed form (needs z_mixture)

PCPG(model, outer="quad", inner="quad", n_outer=19, n_inner=20, seed=42, pg_method="golub_welsch", flip=True)
pcpg.estimate(w, x_partial, n_outer=None, n_inner=None)        # counts default to the constructor's
pcpg.estimate_diff(w, x_partial, n_outer=None, n_inner=None)   # the same as a tensor with gradients
MCBaseline(model, n_samples).estimate(w, x_partial)
```

- `n_outer` / `n_inner` — nodes (quad) or samples (mc) of the outer / inner integral. Cost is `n_outer * n_inner` cf evaluations.
- Quadrature rules for the constructor's counts are precomputed in `__init__`; other counts passed at call time are computed on first use and cached.
- Gauss-PG (golub_welsch) quadrature: any `n_outer` works (exact at 200 digits, checked up to 40); cost grows with n (≈1.6 s at 19, 25 s at 40).

## PC density models

Two probabilistic circuits with the same interface as `GMM` (`forward`, `fit`, `sample`, `characteristic_function`, `z_mean`, `z_mixture`, all batched over points), so `PCPG`, `MCBaseline` and `MixtureAdaptiveQuad`/`MixtureGaussHermite` work on them unchanged:

- `side/utils/pc.py` — `PC`, a small RAT-SPN-like circuit written from scratch (random region trees per repetition, elementwise products, sum layers, root mixture over repetitions). Every pass goes through `_circuit` with other leaf values: log-densities (log-likelihood) or, at missing features, the leaf Gaussian's log CF (CF pass, complex). Sampling is an upward pass per point plus top-down choices per sample.
- `side/utils/spflow_pc.py` — `SPFlowPC`, an SPFlow `RatSPN` with Normal leaves; the CF and mixture passes dispatch over the RAT-SPN module types (RepetitionMixingLayer subclasses Sum, so it is checked first).

Both have a mixture pass (`z_mixture`): p(z | x_obs) as a Gaussian mixture with one component per induced tree, R · C^(2^depth − 1) components, exponential in depth — exact reference for small PCs only.

## estimate.py — estimating the expected prediction

`code/estimate.py` (run from `code/`) is a linear script in the layout of `train.py`, no `main`. Config at the top (`MODEL`, `FEATURES` 5–5000, `MISSING_RATES` 30/50/80 %, `SEED`, PCPG's nodes `N_OUTER·N_INNER` = MC's budget, `N_TEST`, `TIME_REPEATS`), then one top-level loop over all settings whose body is the story for one setting: setup (data with `int(rate·D)` features per test point hidden, density and logistic regression on the complete training points, and the `estimators` dict: mean imputation, MC, PCPG, PCPG exact inner) → estimation and evaluation (the exact values of the first `N_TEST` test points by `MixtureAdaptiveQuad`, exact via `model.z_mixture`, and every estimator on all test points, RMSE and time by `timed_rmse` in `experiment_utils.py`). The toggle `NORM` standardizes the features (`load_data(..., standardize=True)`, needed because `make_classification`'s informative features have variance ≈ n_informative/3) and rescales w to norm 1; one run gives one figure. At the end `compare_estimates` (in `experiment_utils.py`) draws `comparison<SUFFIX>` (`_norm` at the end with `NORM`; RMSE on top, time for the whole test set below) into `code/figures/estimate/<MODEL>/` and `thesis/figures/estimate/<MODEL>/`. With `PCPG_VARIANTS = True` it also draws `pcpg_variants` (the four quad/MC combinations against the budget, at `MAIN_FEATURES`). Densities are cached in `code/cache/estimate/`.

## train.py — training on incomplete data

`code/train.py` (run from `code/`) is a linear script, no `main`. Config at the top defines the settings once (`FEATURES`, `MISSING_RATES`, one fixed `SEED`) and the two functions the story uses: `sigmoid_prediction` and `objective` (mean BCE + sklearn's L2 penalty; for the expected prediction the marginal likelihood). Then one top-level loop over all settings whose body is the story for one setting: setup (data with features hidden, density fit on the incomplete training points, the estimators PCPG/MC/Exact) → training (MC, PCPG, Exact with their `estimate_diff` in the loss, plain Adam on the same minibatches — own generator for the order — fixed learning rate; Complete data and Mean imputation by sklearn on complete / mean-filled points) → evaluation (mean |p − p_Exact| of the exact expected predictions on incomplete test points; accuracy on incomplete test points with three prediction rules: the exact expected prediction, mean imputation with the training means, and every model's own rule (mean imputation for Complete data / Mean imputation, its estimator's `estimate` for MC / PCPG / Exact); accuracy of σ(wᵀx) on complete points; printed per setting). One run is one configuration: `NORM` switches between raw data with C = 1 and standardized data with C = 1e-3 (the best setup of estimate.py); the density cache name contains `INFORMATIVE` and `_norm`. At the end `compare` saves the scores as `code/cache/train/<MODEL>/scores<SUFFIX>.json` and draws `comparison<SUFFIX>_accuracy_missing` (rows: the three prediction rules), and once both configurations are saved also `comparison<SUFFIX without _norm>_difference` (MC and PCPG, raw solid, standardized dashed) `..._accuracy_complete` and `..._accuracy_own` (`plot_train_accuracy`: accuracy on complete points, and on incomplete points with every model's own rule against Complete data with the exact expected prediction; yellow band Complete data; rows raw, standardized). These can be redrawn from the saved json without retraining, with `SUFFIX = _<INFORMATIVE>_<N_OUTER>_<N_INNER>[_norm]` as in estimate.py, into `code/figures/train/<MODEL>/` and `thesis/figures/train/<MODEL>/`; Complete data is the target band. Helpers that are not part of the story (`fit_density` with cache in `code/cache/train/<MODEL>/`, `accuracy`, `compare`; `optimum` is only used by `side/utils/toy.py`) are in `code/experiment_utils.py`, the data (`load_data`, `hide`) in `side/utils/data.py`. MC's −log of an average is biased high by ≈ Var/(2Sp²), which shrinks w. `code/side/train_demo.py` (run from `code/side`) writes the step-by-step figure `figures/train/step_by_step.pdf`: training on a 2-D toy problem (points with a missing feature as lines, the loss over w as a plane), PCPG and MC side by side.

## Gauss-PG quadrature

`PCPG` with `outer="quad"` (and the default `pg_method`) uses fixed quadrature nodes/weights for ω ~ PG(1,0), computed once at `__init__` via the Golub-Welsch algorithm in `pg_gauss_quadrature(n)` (public function in `pcpg.py`). Key details:

- The Laplace transform of PG(1,0) is L(t) = sech(√(t/2)); moments are derived from this analytically using `mpmath` numerical differentiation.
- The entire Golub-Welsch computation (moments → Hankel matrix → Cholesky → Jacobi eigendecomposition) is done in `mpmath` at `dps=200` decimal places — float64 Hankel matrices become singular for n > ~12.
- float64 loses accuracy from n ≈ 12 and Cholesky fails from n = 20; at dps=200 the rule is exact to float64 far beyond that (checked up to n = 40). There is no n ≤ 19 limit.
- The computation takes a few seconds per `PCPG(...)` construction call with an outer quadrature.

`pg_demo.py` is the companion demo script with: `plot_convergence_functions`, `plot_convergence_errors`, `plot_pg_distribution`, `plot_pg_samples`.

`hs_demo.py` demonstrates the Hubbard-Stratonovich identity e^{-ωt²/2} = E_{u~N(0,ω)}[e^{iut}] standalone, without PG.

## Data

`side/utils/data.py` has plain functions, each a one-liner to call:
- `classification(features, informative=None, samples=10_000, test_size=0.2, seed=42)` — sklearn's `make_classification`, stratified split, returns numpy `x_train, x_test, y_train, y_test`.
- `hide(x, missing_rate, rng)` — copy of a point `(D,)` or points `(B, D)` with `int(missing_rate·D)` features per point set to `NaN`.
- `load_data(n_features, missing_rate, informative, seed, device)` — for `train.py`/`estimate.py`: complete and incomplete train/test points and labels, as tensors.
- `loader(x, batch_size=64)` — shuffled `DataLoader` over a tensor, what the density models `fit` on.

## Density model interface

Both estimators expect the density model to expose:
- `sample(n, x_obs)` — for MC.
- `characteristic_function(s, x_obs)` — for PCPG (evaluates the conditional CF of p(x_mis | x_obs) at frequencies s).

`GMM` (`gmm.py`) implements both: diagonal-covariance mixture trained by gradient descent on log-likelihood, parameterised with `log_vars` to keep variances positive.

Missingness convention: a partially observed input is a 1-D tensor with `NaN` in missing positions; `obs_mask = ~torch.isnan(x_partial)` is the standard idiom used everywhere.

Estimator interface: every estimator (`PCPG`, `MCBaseline`, `MeanImputation`, `MixtureGaussHermite`, `MixtureAdaptiveQuad`) has `estimate(w, x_partial, ...)` for evaluation (no gradients, chunked) and `estimate_diff(w, x_partial, ...)` returning a tensor with gradients through w for training (all but `MixtureAdaptiveQuad`); `estimate` calls `estimate_diff`, there is no `__call__`. Batching: `estimate` takes one point `(D,)` and returns a float, or a batch `(B, D)` with its own missing pattern per row and returns a numpy array `(B,)`. The conditional model methods (`sample`, `characteristic_function`, `z_mixture`, `z_mean`) take `x_obs` of shape `(D,)` or `(B, D)` and never index by the mask: masks are multiplied in and all tensors keep the full D features. `characteristic_function(t, x_obs)` takes frequencies over all D features, `(N, D)` or `(B, N, D)`, and ignores the observed entries. Large batches are split into chunks by `in_chunks` (a copy in `pcpg.py` and in `side/utils/baselines.py`). `rmse` in `experiment_utils.py` evaluates all test points in one batched call; `estimate.test_points` returns them stacked as `(N_TEST, D)`.

## demo.py — script entry point

`code/demo.py` is the primary script version of the notebook. Top-level config variables:

```python
n_gauss_quad = 20   # Gauss-Hermite nodes (inner quadrature)
n_pg_quad    = 20   # Golub-Welsch PG nodes (outer quadrature)
n_gauss      = 100  # inner MC samples for eval_estimators()
n_pg         = 100  # outer MC samples for eval_estimators()
pcpg_ratio   = 4    # n_gauss / n_pg for PCPG-MC (see budget allocation below)
K            = 4    # GMM components
n_features   = 15
```

Four functions at the bottom, call whichever sections are needed:
```python
eval_estimators()
allocation_analysis()
results, convergence_budgets, n_gauss_quad = convergence_analysis()
variance_analysis(results, convergence_budgets, n_gauss_quad)
```

## PCPG-MC budget allocation

For a total budget `N = M * L` with desired ratio `r = L/M = pcpg_ratio`:

```
M = sqrt(N / r),   L = N // M
```

`demo.py` sets this as `n_pg_pcpg = max(1, int(np.sqrt(N / pcpg_ratio)))`.

**Recommended ratio: 4.** Reasoning:
- σ²_A (inner variance) is theoretically infinite; L must be large enough to tame the e^{1/(8γ)} blowup for small γ. L ≥ 20–50 is sufficient in practice.
- The variance formula scales the outer term as σ²_B · √ratio / √N, so large ratio hurts when σ²_B is non-negligible.
- ratio=4 gives L≈63 at N=1000 (safe), ratio=10 is comfortable but inflates the outer term unnecessarily.
- With many missing features σ²_B ≈ 0 (CF damping), so ratio matters less — any value in [4, 10] is fine.

## Same-seed pitfall with PCPG estimators

**Do not re-instantiate** a `PCPG` with an MC part inside a loop. Each constructor resets the internal RNG to `seed=42`, so every call returns the same sample — measured variance will be 0. Instantiate once before the loop:

```python
pcpg = PCPG(px, outer="mc", inner="mc")   # once
vals = [pcpg.estimate(w, x_partial, n_outer=1, n_inner=1) for _ in range(n_reps)]  # varied
```

## Variance components

Estimated from single-sample runs (n_reps ≈ 1000):
- `sigma2_mc = np.var(mc1_vals)` — MC per-sample variance (≤ 0.25)
- `sigma2_B  = 4 * np.var(gh1_vals)` — outer PG variance (GH eliminates inner)
- `sigma2_A  = 4 * np.var(pcpg11_vals) - sigma2_B` — inner Gaussian variance (∞ in theory)

`print_variance_components` and `plot_variance_analysis` in `utils.py` handle display/plotting.

## Plotting conventions

Plotting functions in `utils.py` must **never** call `plt.show()` or `print()`. They save to `figures/` and call `plt.close()`. The caller decides whether to display output.

**Every plot uses one style, described in [code/PLOTTING.md](code/PLOTTING.md). Read it before writing or changing any plot.** In short: LaTeX with Latin Modern at the thesis sizes, figures sized to the thesis text width, no grid, and no top or right spines. The palette is pastel (Catppuccin), with one fixed role per colour: butter yellow for σ(z) and the goal, indigo for the bells and GMM components (mist blue for the bell as a target), coral for the waves and everything Gauss-Hermite, grey for p(z). Partial sums are drawn as lines that get darker and heavier, against a pale wide target band in the target's colour. It is implemented in `code/side/utils/plots.py` (`STYLE`/`@styled` for thesis figures, `_BIG_STYLE` for the step-by-step figures, and the colour constants `_PASTEL`, `_DEEP`, `_tone`, `_BAND`, `_SIGMA_BAND`, `_BELL_BAND`, `_BELL_LINES`, `_WAVE_LINES`).

The thesis figures of the methods chapter are generated by `code/side/pcpg_demo.py` (run from `code/side`). They are saved to `code/side/figures/pcpg/{bells,waves,cf,summary}.pdf` and also written to `thesis/figures/pcpg/`, which is where the thesis includes them from. The same script also writes the large step-by-step figure `figures/pcpg/step_by_step.pdf`.

## Dependencies

Not pinned. Required: `torch`, `numpy`, `scikit-learn`, `matplotlib`, `polyagamma`, `spflow` (1.1.0, for `side/utils/pc.py`), `mpmath` (for the PG quadrature of `PCPG`). The `ml` pyenv version has all of these — run scripts with `~/.pyenv/versions/ml/bin/python`.
