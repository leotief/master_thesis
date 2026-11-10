# CODE.md

Everything important about the research code in `code/` (PyTorch). Run scripts from `code/` (the demos in `side/` from `code/side/`) with `~/.pyenv/versions/ml/bin/python`. There is no package layout, `setup.py` or test suite; imports are relative to the directory the script runs from (`from pclr import PCLR, cfi`, `from side.utils.gmm import GMM`).

## Code style

Comments in words, not formulas (basically no written-out math, the math is in the thesis): a short header per file or section that says what it is and the idea, a line or two per function (what it does, what the density needs), and a short comment per step of a longer function (what it does and why). Not more: no derivations, no restating the code.

## The problem

Compute **E[σ(wᵀx) | x_obs]**, the expected logistic-regression prediction with the missing features marginalised out under a learned density p(x) (a GMM or a probabilistic circuit), for evaluation and, differentiably, for training on incomplete data.

## Files

| File | What it is |
|---|---|
| `estimate.py` | experiment: estimating the expected prediction (accuracy and time of all estimators) |
| `pclr.py` | `PCLR`, the joint model of density and logistic regression (thesis: PCLR), its prediction and objective; below it the method for the expected prediction, `cfi` (thesis: CFI), and the batching helpers `chunks` / `join`. PGHS was removed (it is in commit 3a214fe; its thesis text is in `scratchpad/pghs.tex`) |
| `train.py` | experiment: the training comparison (PCLR against imputation baselines on LR-generated data) and the convergence of w; the models (`PCLR` and the baselines of `side/utils/baselines.py`) are built in a dict, then fit and scored in a loop |
| `experiment_utils.py` | helpers of the two experiments, not part of their story: `fit_density` (with cache, GMM only), `logistic_regression`, `accuracy`, `optimum` (only for `side/utils/toy.py`); for `estimate.py` `predictions`, `kl`, `timed_kl`, `compare_estimates`; for `train.py` `training_scores`, `print_training_scores`, `compare_training` (saves the scores and draws the comparison and weights figures), `compare_convergence`; the GMM's `K` |
| `PLOTTING.md` | the plotting style — read before writing or changing any plot |
| `todo.txt` | open ideas and questions |
| `side/utils/gmm.py` | `GMM`, diagonal mixture fit by gradient descent (`log_vars` keep variances positive) |
| `side/utils/baselines.py` | the estimators of `estimate.py` (`monte_carlo`, `mean_imputation`, `conditional_mean`, `mixture_gauss_hermite`, `mixture_adaptive_quad`, functions like `cfi`) and the models of `train.py` (`CompleteData`, `MeanImputation`, `ConditionalMean`, with `fit` / `w` / `predict` like `PCLR`; a model is a way to train w plus one of the estimators to predict with) |
| `side/utils/data.py` | `classification`, `hide`, `load_data` (for `estimate.py`), `load_lr_data` (for `train.py`), `load_real` (four real datasets), `loader` |
| `side/utils/toy.py` | the 2-D toy problem of the step-by-step figures |
| `side/utils/plots.py` | all plotting (`STYLE`/`@styled`, colour constants, every `plot_*`) |
| `side/estimate_demo.py` | step-by-step figures of CFI / MC / GMM Gauss-Hermite on the toy problem (`cfi.pdf`, `mc.pdf`, `gmm_gh.pdf`), and the methods-chapter figures `{waves,cf}.pdf` (also into `thesis/figures/cfi/`) |
| `side/train_demo.py` | step-by-step figure of training on the toy problem, CFI and MC side by side |

Figures go to `code/figures/<folder>/` (`code/side/figures/` for the demos), and the experiment figures directly also to `thesis/figures/<folder>/` (`THESIS_FIGURES` in `plots.py`). Paper figures go to `paper/figures/` (see `paper/PAPER.md`). Fitted densities are cached in `code/cache/estimate/` and `code/cache/train/<MODEL>/`.

## The methods for the expected prediction

Every method is a plain function with one signature, no classes:

```python
method(density, w, x_partial, **settings)    # x_partial (D,) or (B, D), nan = missing -> tensor () or (B,)
```

with gradients through w and the density's parameters (all but `mixture_adaptive_quad`), so the same call is used for training and evaluation. Large batches are split internally into chunks (`chunks` / `join` at the end of `pclr.py`, `MAX_ELEMENTS` elements of the largest tensor), so no wrapper is needed. Where a fixed `predict(w, x)` is wanted (the toy demos), bind density and settings with `functools.partial(method, density, **settings)`.

```python
cfi(density, w, x_partial, n_freq=200)               # pclr.py, the main method (thesis: CFI)
monte_carlo(density, w, x_partial, n_samples=1000)          # baselines.py
mean_imputation(means, w, x_partial)                        # baselines.py, the means to fill in instead of a density
conditional_mean(density, w, x_partial)                     # baselines.py, fills in E[x_mis | x_obs], needs conditional_mean
mixture_gauss_hermite(density, w, x_partial, n_nodes=50)    # baselines.py, needs z_mixture
mixture_adaptive_quad(density, w, x_partial, epsrel=1e-12)  # baselines.py, the reference: scipy quad, point by point, no gradients, float64
```

### CFI (`cfi`, characteristic-function inversion)

E[σ(z)] = ½ + ∫₀^∞ Im φ_z(s)/sinh(πs) ds, exact, by the trapezoid rule on K = `n_freq` frequencies. The step h per point is min(π/(|E z| + 6·std z + 1), 12/K) from `density.z_mean` and `density.z_var`; it is detached (the value doesn't depend on it). Needs `characteristic_function`, `z_mean`, `z_var`.

## Density models

Interface (all batched over points): `forward` (log-likelihood), `fit(loader)`, `sample(n, x_obs)` (for MC), `characteristic_function(s, x_obs)` (for CFI), `z_mean(w, x_obs)` (CFI's step), `z_var(w, x_obs)` (CFI's step), `conditional_mean(x_obs)` (the conditional-mean baseline of `train.py`), `z_mixture(w, x_obs)` (p(z | x_obs) as a Gaussian mixture, for the exact baselines).

- `GMM` — diagonal-covariance mixture, K components (`K` in `experiment_utils.py`); `z_mixture` has K components.
- **No PC at the moment.** `side/utils/spflow_pc.py` (an SPFlow `RatSPN`) was deleted, and with it the `"pc"` branch of `fit_density`. A PC needs the same interface; the CF and moment passes are described in the methods chapter (`thesis/chapters/methods.tex`, Section "Probabilistic Circuits"). It is in the git history (commit 3a214fe) if needed.

Conventions:
- A partially observed input has `NaN` at the missing features; `obs_mask = ~torch.isnan(x_partial)` is the idiom.
- Conditional methods take `x_obs` of shape `(D,)` or `(B, D)` and never index by the mask: masks are multiplied in and all tensors keep the full D features. `characteristic_function(t, x_obs)` takes frequencies over all D features, `(N, D)` or `(B, N, D)`, and ignores the observed entries.

## Evaluation (`experiment_utils.py`)

`predictions(method, first, w, points, **settings)` calls a method without gradients and returns a numpy array; `timed_kl` takes the same `(method, first, ..., **settings)`. `kl` computes in float64: the estimates are float32, and `1 − p̂` in float32 is too coarse close to 1, which once gave negative KL values (drawn as vertical drops on the log axis). `estimate.py` keeps its methods as a dict name → `(method, first, settings)`, `first` being the density (the training means for mean imputation).

## Data (`side/utils/data.py`)

- `classification(features, informative=None, samples=10_000, test_size=0.2, seed=42, standardize=False)` — sklearn's `make_classification`, stratified split, numpy `x_train, x_test, y_train, y_test`.
- `hide(x, missing_rate, rng)` — copy of `(D,)` or `(B, D)` with `int(missing_rate·D)` features per point set to `NaN`.
- `load_data(n_features, missing_rate, informative, seed, device, standardize=False)` — complete and incomplete train/test points and labels as tensors.
- `loader(x, batch_size=64)` — shuffled `DataLoader`, what the density models `fit` on.

## estimate.py — estimating the expected prediction

CFI against MC and mean imputation: how accurate and how fast. A linear script, no `main`, one loop over all settings, on the GMM, and two figures.

- **Comparison against the number of features:** one loop over `FEATURES` (5–5000) × `MISSING_RATES` (30/50/80 %). Per setting: data with `int(rate·D)` features per test point hidden, density and logistic regression on the complete training points, the `methods` dict name → `(method, first, settings)` with MC at `MC_SAMPLES` and CFI at `CFI_FREQS`; the exact values of the first `N_TEST` test points by `mixture_adaptive_quad`, then the mean Bernoulli KL(p ‖ p̂) (in float64, clamped to [1e-6, 1 − 1e-6]) and the time of every method by `timed_kl`. `compare_estimates` draws `comparison<SUFFIX>` (KL on top, time for the whole test set below).
- **CFI with different numbers of frequencies:** in the same loop, CFI also with every count of `CFI_COUNTS`, on the same points and against the same exact values. `plot_cfi_frequencies` draws `cfi_frequencies<SUFFIX>` in the layout of the comparison (KL on top, time below, against the number of features), every count its own shade of indigo (light to dark, from few to many frequencies) and its own marker, at most four counts.

Both figures go to `code/figures/estimate/<MODEL>/` and `thesis/figures/estimate/<MODEL>/`, with `SUFFIX = _<INFORMATIVE>_<MC_SAMPLES>_<CFI_FREQS>[_norm]`. `NORM` standardizes the features (`make_classification`'s informative features have variance ≈ n_informative/3), which keeps z moderate. The logistic regression is sklearn's default C = 1 (`logistic_regression` in `experiment_utils.py`).

## pclr.py — the joint model

`PCLR(density, n_features, n_freq=100, lam=1.0, C=1.0, eps=1e-6, train_density=True, max_iter=1000)`, an `nn.Module` holding the density p_θ(x) (made trainable again) and the LR weights `w` (start at 0), the joint model p(x, y) = p_θ(x)·σ(wᵀx)^y·(1 − σ(wᵀx))^(1−y) of the thesis. For an incomplete point the log-likelihood splits into log p_θ(x_obs) (the density's `forward`) and log p(y | x_obs) (BCE of the expected prediction by `cfi`). The model are only θ and w; CFI is how one of its queries is computed, not a part of it.

```python
model.predict(x_partial)                     # p(y=1 | x_obs), tensor (B,), gradients through θ and w
model.log_density(x_partial)                 # log p_θ(x_obs), tensor (B,)
model.log_likelihood(x_partial, y)           # (mean log p_θ(x_obs), mean log p(y | x_obs))
model.objective(x_partial, y, n_train)       # −(log p_θ(x_obs) + lam·log p(y | x_obs)) + |w|²/(2 C n_train)
```

`model.fit(x_partial, y)` trains the density and w jointly, or only w (`train_density=False`), by L-BFGS on all points at once until convergence.

## train.py — the training comparison

How well PCLR and the usual ways to train a logistic regression do when the training points have missing features. A linear script, no `main`. All models are optimized until convergence by L-BFGS (sklearn for the baselines, `PCLR.fit` full batch for PCLR), so only the methods differ.

- **Data:** `load_lr_data` (`side/utils/data.py`), where logistic regression is the right model: features from a GMM with random means and spreads, a fixed random `w_true`, labels drawn with σ(w_trueᵀx). Training and test points get missing features at the same rate; 5000 training and 20,000 test points.
- **Config:** `FEATURES` (4–256, powers of 2), `MISSING_RATES` (30/50/80 %), `SEED` (the seed of the data and the hidden features), PCLR's `N_FREQ`, `LAMBDA` and L-BFGS's `MAX_ITER`; the penalty is the default C = 1 everywhere.
- **Loop:** features × missing rates: the data, the density fit on the incomplete training points (cached in `code/cache/train/<MODEL>/`), every model called directly into a `results` dict. Every model returns its learned `w` and its own predictions for the incomplete test points, and is scored on **complete** test points (σ(wᵀx), only the training differs), on **incomplete** test points (its own rule), and by `|w − w_true| / |w_true|`.
- **Output:** `code/cache/train/<MODEL>/training_scores<SUFFIX>.json` (`SUFFIX = _<LAMBDA>`), and into `code/figures/train/<MODEL>/` and `thesis/figures/train/<MODEL>/`: `training_complete_test<SUFFIX>` and `training_incomplete_test<SUFFIX>` by `plot_training_comparison` (a column per missing rate, log-loss on top and accuracy below, every panel its own y-axis, complete data as the butter band), and `weights<SUFFIX>` by `plot_weight_error` (the relative error `|w − w_true| / |w_true|` against the number of features, a panel per missing rate, complete data as the butter band).

- **Convergence:** at `CONVERGENCE_FEATURES` (100) and `CONVERGENCE_RATE` (50 %), the relative error of w against the number of training points `N_TRAINS` (2¹⁰ to 2¹⁵, powers of 2). The largest dataset is generated once and every size uses its first n points, so `w_true` is the same for all (`load_lr_data` draws `w_true` after the features, so datasets of different sizes would each have another). Saved as `code/cache/train/<MODEL>/convergence_errors<SUFFIX>.json` and drawn as `convergence<SUFFIX>` by `plot_convergence` (log–log, default log ticks as in the estimate figures, complete data as the butter band); about 1.5 minutes, 3.6 GB GPU memory at the largest size.

The models are built in a `models` dict, name → model, every one with `fit(x_train, y_train)`, its weights `w` and `predict(x_test)` (its own rule for incomplete test points): `PCLR` from `pclr.py` (on a `copy.deepcopy` of the density) and the baselines of `side/utils/baselines.py`. Then every model is fit, only `Complete data` on the complete training points, and scored:

| name | training | own prediction on incomplete test points |
|---|---|---|
| Complete data | LR on the complete training points — the upper bound | expected prediction (CFI) under the fitted density |
| Mean imputation | LR on the gaps filled by the training means | the same filling |
| Conditional mean | LR on the gaps filled by the density's E[x_mis \| x_obs] (`GMM.conditional_mean`) | the same filling |
| PCLR fixed density | `pclr` with `train_density=False`: only w, the density kept as fit | expected prediction under the fitted density |
| PCLR | density and w jointly | expected prediction under its own density |

`load_real` (four complete UCI/OpenML datasets) is still in `data.py`, for a later table on real data; `generative_classifier` was removed. Only the GMM so far (CFI needs `z_var`).

## Plotting

Plot functions never call `plt.show()` or `print()`; they save and `plt.close()`. Every plot uses the one style of [PLOTTING.md](PLOTTING.md) (LaTeX with Latin Modern, figures sized to the text width, no grid, no top/right spines, the pastel Catppuccin palette with one fixed role per colour), implemented in `side/utils/plots.py`.

## Dependencies

Not pinned: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `polyagamma`, `mpmath` (`spflow` 1.1.0 only for the deleted PC). The `ml` pyenv version has all of them.

## Pitfalls

- `PCLR` trains the density it gets in place; the model functions pass it a `copy.deepcopy`, so the fitted density stays unchanged for the other models.
- `fit_density` caches by path. The cache names contain the setting (features, missing rate, seed, and for the convergence the number of points); if `load_lr_data` changes how the data is drawn, delete `code/cache/train/gmm/` or the cached densities no longer match the data.
- `load_lr_data` draws `w_true` after the features, so `w_true` depends on `n_train` and `n_test`. The convergence therefore takes nested subsets of one large dataset.
- Training must be comparable: all models are optimized to convergence (L-BFGS). An earlier version trained PCLR with a fixed number of Adam steps against converged sklearn baselines, which made PCLR look worse at many features.
- CFI with few frequencies is inaccurate when z is narrow and far from 0 (few features, few missing): K has to be ≥ 12·(|E z| + 6·std z + 1)/π. 64 frequencies are at the float32 limit in every setting of `estimate.py`.

## Status and findings (November 2026)

- **Estimation (`estimate.py`):** CFI is at the float32 limit (KL ≈ 1e-13 to 1e-15) for 5 to 5000 features and every missing rate, and about 10–20× faster than MC with 1000 samples (KL ≈ 1e-4). From about 50 features 16 frequencies are enough; few features need more (64 is safe).
- **Training (`train.py`, LR-generated data, 5000 training points):** PCLR's w is the closest to `w_true` of the methods on incomplete data up to about 16–64 features. With many features and few points per weight, conditional-mean imputation is about as good or better: it shrinks w (biased, but more stable), while PCLR is the consistent estimate and noisier with little data. On incomplete test points all methods are very close.
- **Convergence (100 features, 50 %):** PCLR's error falls steadily with the number of training points (0.80 → 0.12 from 2¹⁰ to 2¹⁵), conditional mean levels off at about 0.33, mean imputation at about 0.6. The crossover is at a few thousand points. Checked: with the *true* density instead of the fitted one the results are the same, so the density fit is not the cause; and PCLR's objective is lower at its w than at `w_true`, so it is not the optimizer.
- **Joint vs. fixed density:** almost identical (λ = 1, the density term dominates, and the GMM is the right model). Joint training only helped where the fixed density was poor (4–5 features at 80 % missing).
- **Earlier experiments:** on `make_classification` (Gaussian clusters per class) the generative classifier (a GMM per class, Bayes' rule) wins, because its model is exactly right there and an LR is not. On real data (`load_real`, K = 4) the results were mixed: PCLR best of the LR methods on MAGIC and Banknote, mean imputation best on Spambase (skewed count features, poor GMM fit) and Breast Cancer (too small). Real data should be reported as a table, not a figure.

### Open points

1. A PC as the density model (the paper is about PCs): its CF, `z_mean`, `z_var`, `conditional_mean`.
2. Fair regularization: C per method chosen on a validation split (now C = 1 everywhere, too weak for PCLR at many features and few points).
3. Several seeds for the final figures (now one seed; the curves over the number of features zigzag).
4. Real data as a table (`load_real`), with a larger K or the PC.
5. The prior-work approach (Khosravi et al., moments / Taylor series of the sigmoid) as a competitor in `estimate.py`.
6. Cleanups: old figures no longer generated (`figures/train/gmm/training_1.0.pdf`, `pclr_*` figures, `side/figures/estimate/pghs.pdf`, `thesis/figures/pghs/`).
