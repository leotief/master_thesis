# CODE.md

Everything important about the research code in `code/` (PyTorch). Run scripts from `code/` (the demos in `utils/` from `code/utils/`) with `~/.pyenv/versions/ml/bin/python`. There is no package layout, `setup.py` or test suite; imports are relative to the directory the script runs from (`from pclr import PCLR, cfi`, `from utils.gmm import GMM`).

## Code style

Comments:
- **No math.** No formulas, symbols or tensor shapes in comments (no `p(k | x_obs)`, `sum_k`, `(B, K)`); the math is in the thesis. Say it in simple words.
- **A short header per file or section:** what it is and the idea.
- **A comment at the top of a function:** in one or two plain sentences, what it does and why we need it (who uses it, what for). Not where the name already says it (e.g. `__init__`); explain its steps inside instead.
- **Inside a function, only where needed:** a short comment at a step that is not obvious from the code. Start with what it does in a few words ("init the means with k-means"), then why. Simple steps get none.
- Not more: no derivations, no restating the code, no history of what was tried before (that goes in this file).

## The problem

Compute **E[σ(wᵀx) | x_obs]**, the expected logistic-regression prediction with the missing features marginalised out under a learned density p(x) (a GMM or a probabilistic circuit), for evaluation and, differentiably, for training on incomplete data.

## Files

| File | What it is |
|---|---|
| `estimate.py` | experiment: estimating the expected prediction (accuracy and time of all estimators) |
| `pclr.py` | `PCLR`, the joint model of density and logistic regression (thesis: PCLR), its prediction and objective; below it the method for the expected prediction, `cfi` (thesis: CFI), and the batching helpers `chunks` / `join`. PGHS was removed (it is in commit 3a214fe; its thesis text is in `scratchpad/pghs.tex`) |
| `train.py` | experiment: the training comparison (PCLR against imputation baselines on `make_classification`) and the learning curve; the models (`PCLR` and the baselines of `utils/baselines.py`) are built in a dict, then fit and scored in a loop |
| `real_data.py` | experiment: the real-data comparison, every model on 6 real datasets at 10/30/50 % missing over 5 seeds; writes the tables `paper/tables/real_data_<score>_<percent>.tex` and prints them |
| `utils/experiments.py` | helpers of the experiments, not part of their story: `fit_density` (with cache, GMM only), `logistic_regression`, `accuracy`; for `estimate.py` `predictions`, `kl`, `timed_kl`, `score_methods`, `compare_estimates` (takes the plot function: `plot_estimate_comparison` or `plot_cfi_frequencies`; saves the scores as `cache/estimate/<MODEL>/<name>.json`); for `train.py` `prediction_scores`, `training_scores`, `print_training_scores`, `compare_training` (saves the scores and draws the comparison figures), `compare_learning_curve` (all three with `latex=False` for figures only in `figures/`) |
| `PLOTTING.md` | the plotting style — read before writing or changing any plot |
| `todo.txt` | open ideas and questions |
| `utils/gmm.py` | `GMM`, diagonal mixture fit by gradient descent (`log_vars` keep variances positive) |
| `utils/baselines.py` | the estimators of `estimate.py` (`conditional_sampling`, `mean_imputation`, `conditional_mean_imputation`, `mixture_gauss_hermite`, `mixture_adaptive_quad`, functions like `cfi`) and the models of `train.py` and `real_data.py` (`CompleteData`, `MeanImputation`, `ConditionalMeanImputation`, `MICE`, `GenerativeClassifier` (a joint GMM over (x, y), uses `GMM.component_log_probs`), with `fit` / `predict` like `PCLR`, all but the generative classifier also with `w`; a model is a way to train w plus one of the estimators to predict with) |
| `utils/data.py` | `classification`, `hide`, `load_data` (`make_classification`, for `train.py` and `estimate.py`), `load_real` (four real datasets), `loader` |
| `utils/toy.py` | the 2-D toy problem of the step-by-step figures (with `optimum`, the L-BFGS optimum of w) |
| `utils/plots.py` | all plotting (`STYLE`/`@styled`, colour constants, every `plot_*`) |
| `utils/estimate_demo.py` | step-by-step figures of CFI / MC / GMM Gauss-Hermite on the toy problem (`cfi_steps.pdf`, `mc_steps.pdf`, `gmm_gh_steps.pdf`), and the methods-chapter figures `{waves,cf}.pdf` (also into `thesis/figures/cfi/`) |
| `utils/train_demo.py` | step-by-step figure of training on the toy problem, CFI and MC side by side |

Figures go to `code/figures/<folder>/` (`code/utils/figures/` for the demos), and the experiment figures directly also to `thesis/figures/<folder>/` (`THESIS_FIGURES` in `plots.py`). Paper figures go to `paper/figures/` (see `paper/PAPER.md`). Fitted densities are cached in `code/cache/estimate/` and `code/cache/train/<MODEL>/`.

## The methods for the expected prediction

Every method is a plain function with one signature, no classes:

```python
method(density, w, x_partial, **settings)    # x_partial (D,) or (B, D), nan = missing -> tensor () or (B,)
```

with gradients through w and the density's parameters (all but `mixture_adaptive_quad`), so the same call is used for training and evaluation. Large batches are split internally into chunks (`chunks` / `join` at the end of `pclr.py`, `MAX_ELEMENTS` elements of the largest tensor), so no wrapper is needed. Where a fixed `predict(w, x)` is wanted (the toy demos), bind density and settings with `functools.partial(method, density, **settings)`.

```python
cfi(density, w, x_partial, n_freq=200)               # pclr.py, the main method (thesis: CFI)
conditional_sampling(density, w, x_partial, n_samples=1000) # baselines.py, monte carlo with samples of p(x_mis | x_obs)
mean_imputation(means, w, x_partial)                        # baselines.py, the means to fill in instead of a density
conditional_mean_imputation(density, w, x_partial)         # baselines.py, fills in E[x_mis | x_obs], needs conditional_mean
mixture_gauss_hermite(density, w, x_partial, n_nodes=50)    # baselines.py, needs z_mixture
mixture_adaptive_quad(density, w, x_partial, epsrel=1e-12)  # baselines.py, the reference: scipy quad, point by point, no gradients, float64
```

### CFI (`cfi`, characteristic-function inversion)

E[σ(z)] = ½ + ∫₀^∞ Im φ_z(s)/sinh(πs) ds, exact, by the trapezoid rule on K = `n_freq` frequencies. The step h per point is min(π/(|E z| + 6·std z), 12/K) from `density.z_mean` and `density.z_var`; it is detached (the value doesn't depend on it). Needs `char_func`, `z_mean`, `z_var`.

## Density models

Interface (all batched over points): `forward` (log-likelihood), `fit(loader)`, `sample(n, x_obs)` (for MC), `char_func(s, x_obs)` (for CFI), `z_mean(w, x_obs)` (CFI's step), `z_var(w, x_obs)` (CFI's step), `conditional_mean(x_obs)` (for conditional mean imputation), `z_mixture(w, x_obs)` (p(z | x_obs) as a Gaussian mixture, for the exact baselines).

- `GMM` — diagonal-covariance mixture, K components (`K` in the config of every experiment script, passed to `fit_density(model, x, path, device, K)`; the cache names start with `K<K>_`); `z_mixture` has K components. The constructor `GMM(x, K, device)` starts it from k-means on the mean-filled points (centres, cluster variances and weights), then `fit` runs Adam; with the means at the first K points instead (before 2026-11-11), one component took all the points from about 64 features on, so conditional mean imputation equalled mean imputation and PCLR's density carried no information. Every variance is at least `MIN_VAR` (1e-2, `GMM.variances()`): without it a component collapses onto a value many points share (the zeros of Pima and Spambase), and joint PCLR training then drives the density term up without bound and wrecks the classifier (Pima log-loss 0.72 instead of 0.49). Densities cached before this were fitted without the floor (`code/cache/estimate/` still has old densities).
- **No PC at the moment.** `utils/spflow_pc.py` (an SPFlow `RatSPN`) was deleted, and with it the `"pc"` branch of `fit_density`. A PC needs the same interface; the CF and moment passes are described in the methods chapter (`thesis/chapters/methods.tex`, Section "Probabilistic Circuits"). It is in the git history (commit 3a214fe) if needed.

Conventions:
- A partially observed input has `NaN` at the missing features; `obs_mask = ~torch.isnan(x_partial)` is the idiom.
- Conditional methods take `x_obs` of shape `(D,)` or `(B, D)` and never index by the mask: masks are multiplied in and all tensors keep the full D features. `char_func(t, x_obs)` takes frequencies over all D features, `(N, D)` or `(B, N, D)`, and ignores the observed entries.

## Evaluation (`utils/experiments.py`)

`predictions(method, first, w, points, **settings)` calls a method without gradients and returns a numpy array; `timed_kl` takes the same `(method, first, ..., **settings)`. `kl` computes in float64: the estimates are float32, and `1 − p̂` in float32 is too coarse close to 1, which once gave negative KL values (drawn as vertical drops on the log axis). `estimate.py` keeps its methods as a dict name → `(method, first, settings)`, `first` being the density (the training means for mean imputation).

## Data (`utils/data.py`)

- `classification(features, informative=None, samples=10_000, test_size=0.2, seed=42, standardize=False, redundant=0.0)` — sklearn's `make_classification`, stratified split, numpy `x_train, x_test, y_train, y_test`.
- `hide(x, missing_rate, rng)` — copy of `(D,)` or `(B, D)` with `int(missing_rate·D)` features per point set to `NaN`.
- `load_data(n_features, missing_rate, informative, seed, device, standardize=True, samples=10_000, test_size=0.2, redundant=0.0)` — complete and incomplete train/test points and labels as tensors.
- `INFORMATIVE = 0.25`, `REDUNDANT = 0.5` — the feature split of the synthetic data, the same in every experiment (`estimate.py` and `train.py` import it): 1/4 informative, 1/2 linear combinations of those, 1/4 noise, so the features are strongly correlated and the signal is low-dimensional, as in most tabular data. Fixed (decided 2026-11-18); don't tune it further.
- `loader(x, batch_size=64)` — shuffled `DataLoader`, what the density models `fit` on.

## estimate.py — estimating the expected prediction

CFI against conditional sampling (MC), mean imputation and conditional mean imputation: how accurate and how fast. A linear script, no `main`, one loop over all settings, on the GMM, and two figures.

- **Comparison against the number of features:** one loop over `FEATURES` (5–5000) × `MISSING_RATES` (10/30/50 %). Per setting: data with every entry of the test points hidden with probability `rate`, density and logistic regression on the complete training points, the `methods` dict name → `(method, first, settings)` with conditional sampling at `N_SAMPLES` and CFI at `CFI_FREQS`; the exact values of the first `N_TEST` test points by `mixture_adaptive_quad`, then the mean Bernoulli KL(p ‖ p̂) (in float64, clamped to [1e-6, 1 − 1e-6]) and the time of every method by `score_methods` (`timed_kl` per method, and the printout). `compare_estimates` draws `estimators<ESTIMATORS_SUFFIX>` (KL on top, time for the whole test set below).
- **CFI with different numbers of frequencies** (only if `CFI_FREQUENCIES`, on by default since the figure is in the paper; set it to `False` to skip): in the same loop, CFI also with every count of `CFI_COUNTS`, on the same points and against the same exact values. `compare_estimates` with `plot_cfi_frequencies` draws `cfi_frequencies<FREQUENCIES_SUFFIX>` in the layout of the comparison (KL on top, time below, against the number of features), every count its own shade of indigo (light to dark, from few to many frequencies) and its own marker, at most four counts.

The data is `make_classification` (`load_data`, 8000 complete training points, 2000 test points), standardized by default (`NORM = True`, as in `train.py` and as the real data; decided 2026-11-18): the LR converges (on raw features sklearn's L-BFGS did not in 3 settings), the L2 penalty treats every feature alike, and the data is treated like the real data. CFI does not need it (it adapts its step to the spread of z, which on raw features grows with the number of features) and is as accurate on raw features. Standardizing does not help the diagonal GMM separate its components at many features (see Status and findings): that comes from how `make_classification` mixes the features, not from their scale. Both figures go to `code/figures/estimate/<MODEL>/`, `thesis/figures/estimate/<MODEL>/` and `paper/figures/estimate/<MODEL>/` (the paper ones at the ICML text width, see `save_figure`), and the scores to `code/cache/estimate/<MODEL>/<figure name>.json`, named by the method settings they depend on, every value with its name (the data split is the same everywhere, so it is not in the names): `ESTIMATORS_SUFFIX = _<N_SAMPLES>samples_<CFI_FREQS>freqs[_raw]`, `FREQUENCIES_SUFFIX = [_raw]`. Standardized features are the default and get no suffix, raw ones `_raw` (in the figures, the json and the cached densities of both experiments). The logistic regression is sklearn's default C = 1 (`logistic_regression` in `utils/experiments.py`).

## pclr.py — the joint model

`PCLR(n_freq=100, C=1.0, density_mode="untrained")` with `fit(x_partial, y, density, max_iter=1000)` (the clamp of the predictions in the loss is the constant `EPS = 1e-6` in `pclr.py`), an `nn.Module` holding the density p_θ(x) (made trainable again) and the LR weights `w` (start at 0), the joint model p(x, y) = p_θ(x)·σ(wᵀx)^y·(1 − σ(wᵀx))^(1−y) of the thesis. For an incomplete point the log-likelihood splits into log p_θ(x_obs) (the density's `forward`) and log p(y | x_obs) (BCE of the expected prediction by `cfi`). The model are only θ and w; CFI is how one of its queries is computed, not a part of it.

```python
model.predict(x_partial)                     # p(y=1 | x_obs), tensor (B,), gradients through θ and w
model.density(x_partial)                     # log p_θ(x_obs), tensor (B,)
model.log_likelihood(x_partial, y)           # (mean log p_θ(x_obs), mean log p(y | x_obs))
model.objective(x_partial, y)                # −(log p_θ(x_obs) + log p(y | x_obs)) + |w|²/(2 C n), n the number of points
```

`model.fit(x_partial, y, density)` trains a new untrained GMM of the same size (default, `density_mode="untrained"`; a copy of the fitted density with `density_mode="fitted"`) and w jointly, or only w on the fitted density (`density_mode="fixed"`), by L-BFGS on all points at once until convergence.

## train.py — the training comparison

How well PCLR and the usual ways to train a logistic regression do when the training points have missing features. Layout: config; two model dicts, `MODELS` (the reference `Complete data`, the imputations, PCLR) and `PCLR_VARIANTS` (PCLR from an untrained density, from the fitted density, with the fitted density fixed); the two experiments `comparison(models, prefix)` and `learning_curve(models, prefix)`, each loading the data and fitting the density once per setting and training and scoring every model of the dict (the reference, recognized by `isinstance(model, CompleteData)`, on the complete training and test points); and four run lines (the variants, their names starting with `pclr_variants_`, then the main comparison with no prefix), each can be commented out. All models are optimized until convergence by L-BFGS (sklearn for the baselines, `PCLR.fit` full batch for PCLR), so only the methods differ.

- **Data:** sklearn's `make_classification` (`load_data`, with the split `INFORMATIVE`, `REDUNDANT` of `utils/data.py`, standardized (`NORM = True`, raw files get `_raw`); without redundant features the observed features say little about the missing ones and all methods are close, with them PCLR leads mean imputation clearly, e.g. log-loss 0.580 against 0.635 at 16 features and 50 % missing), 8000 training and 2000 test points. Training and test points get missing features at the same rate. There is no true `w` (the LR is not the true model of this data), so there is no weight error; models are scored by their predictions only.
- **Config:** `FEATURES` (4–256, powers of 2), `MISSING_RATES` (10/30/50 %, every entry missing with that probability), `SEED` (the hidden features), `NORM`, PCLR's `N_FREQ` (L-BFGS runs `fit`'s default of 1000 iterations); the penalty is the default C = 1 everywhere.
- **Loop:** features × missing rates: the data, the density fit on the incomplete training points (cached in `code/cache/train/<MODEL>/`), the `models` dict, every model fit (only `Complete data` on the complete training points), then scored by `training_scores` on **complete** test points (σ(wᵀx), only the training differs) and on **incomplete** test points (its own rule): log-loss and accuracy.
- **Output:** `code/cache/train/<MODEL>/<prefix>scores<suffix>.json` (`<suffix>` = `NORM_SUFFIX`, empty for standardized features, `_raw` for raw ones), and into `code/figures/train/<MODEL>/`, `thesis/figures/train/<MODEL>/` and `paper/figures/train/<MODEL>/`: `<prefix>complete_test<suffix>` and `<prefix>incomplete_test<suffix>` by `plot_training_comparison` (a column per missing rate, log-loss on top and accuracy below, every panel its own y-axis, complete data as the butter band only on complete test points).
- **Learning curve:** at `CURVE_FEATURES` (100) and `CURVE_RATE` (30 %), the log-loss and accuracy on the same `CURVE_TEST` (5000) incomplete test points against the number of training points `N_TRAINS` (2⁸ to 2¹⁴). One large dataset (`load_data` with `samples` and `test_size`), every size uses its first n training points. Saved as `code/cache/train/<MODEL>/<prefix>learning_curve<suffix>.json` and drawn as `<prefix>learning_curve<suffix>` by `plot_learning_curve` (log-loss left, accuracy right, complete data as the butter band).
- **PCLR variants:** `PCLR_VARIANTS`, their names starting with `pclr_variants_`: `PCLR` (default, the trained density starts untrained), `PCLR fitted start` (`density_mode="fitted"`), `PCLR fixed density` (`density_mode="fixed"`, the fitted density).

Every model has `fit(x_train, y_train, density)` (the incomplete training points and the density fit on them; not every model uses it), its weights `w` and `predict(x_test)` (its own rule for incomplete test points). `PCLR` from `pclr.py` trains a new untrained GMM (`GMM(x, K, device)`, the default) or its own copy of the fitted density and creates `w` in `fit`; the others are in `utils/baselines.py`. The same model objects are trained again on every setting.

| name | training | own prediction on incomplete test points |
|---|---|---|
| Complete data | LR on the complete training points — the reference without missing values | σ(wᵀx) on the **complete** test points (it has no density) |
| Mean imputation | LR on the gaps filled by the training means | the same filling |
| Conditional mean imputation | LR on the gaps filled by the density's E[x_mis \| x_obs] (`GMM.conditional_mean`) | the same filling |
| PCLR fixed density | `pclr` with `density_mode="fixed"`: only w, the density kept as fit | expected prediction under the fitted density |
| PCLR | density and w jointly | expected prediction under its own density |

## real_data.py — the real-data table

Every model on real datasets (`DATASETS`: magic, phoneme, spambase, banknote, qsar, pima; `load_real` also has breast cancer, ionosphere and blood; complete, continuous, binary, standardized on the training points, 80/20 split by seed), at `MISSING_RATES` (10/30/50 %, every entry missing with that probability, in training and test), for `SEEDS` (5, each a new split and new hidden features). Densities cached in `code/cache/real/<MODEL>/` (the generative classifier fits its own joint GMM, uncached). Same layout as `train.py`: config, `MODELS` (with the reference `No missing values`; the PCLR variants commented out), the loop. Every model is trained on the incomplete training points and predicts the incomplete test points with its own rule; the reference "No missing values" (`CompleteData`) is trained and tested on the complete points. Scored by `prediction_scores` (log-loss, accuracy). `real_data_tables` prints both tables and writes them as LaTeX tabulars (booktabs, mean ± std, the best non-reference model per row bold) to `paper/tables/real_data_<score>_<percent>.tex`, one per missing rate. Models: no missing values, mean imputation, conditional mean imputation, MICE (`IterativeImputer`), generative classifier (one GMM over (x, y) with K components, each with a Bernoulli label leaf; p(y | x_obs) by Bayes' rule over the components; the same budget as PCLR's density. It replaced a GMM per class, which had twice the components and blew up on Spambase and QSAR), PCLR (joint, the density starting untrained; its variants as commented-out entries of `MODELS`). Run on 2026-11-11 (results in the tables). Joint PCLR diverged on pima until the GMM got its minimum variance (see Density models). Only the GMM so far (CFI needs `z_var`).

## Plotting

Plot functions never call `plt.show()` or `print()`; they save and `plt.close()`. Every plot uses the one style of [PLOTTING.md](PLOTTING.md) (LaTeX with Latin Modern, figures sized to the text width, no grid, no top/right spines, the pastel Catppuccin palette with one fixed role per colour), implemented in `utils/plots.py`.

## Dependencies

Not pinned: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `polyagamma`, `mpmath` (`spflow` 1.1.0 only for the deleted PC). The `ml` pyenv version has all of them.

## Pitfalls

- `PCLR` trains the density it gets in place; the model functions pass it a `copy.deepcopy`, so the fitted density stays unchanged for the other models.
- **Precision at many features:** the GMM's log-likelihood per component (`component_log_probs`) is a sum over all observed features, in the thousands at many features. In float32 the differences between the components got lost: at 5000 features the posterior weights were off by up to 1.4e-3, and CFI's error grew to about 1e-4 in p (KL up to 5e-10). So `component_log_probs` computes in float64, and `forward` / `posterior_weights` cast back to the input's dtype; everything else stays float32. With that, float32 CFI is accurate to about 1e-7 in p in every setting, and in float64 to about 1e-14. The reference `mixture_adaptive_quad` works on a float64 copy of the density for the same reason (in float32 its mixture of z was off by up to 2e-5 at 5000 features). Checked 2026-11-18 by computing CFI and the reference each in float32 and float64.
- `fit_density` caches by path. The cache names contain the setting (features, informative and redundant fractions, missing rate, seed, `_raw` for raw features, and for the learning curve the number of points); if the data changes, change the cache names or delete the old caches, or the cached densities no longer match the data.
- Training must be comparable: all models are optimized to convergence (L-BFGS). An earlier version trained PCLR with a fixed number of Adam steps against converged sklearn baselines, which made PCLR look worse at many features.
- CFI with few frequencies is inaccurate when z is narrow and far from 0 (few features, few missing): K has to be ≥ 12·(|E z| + 6·std z)/π. 64 frequencies are at the float32 limit in every setting of `estimate.py`.

## Status and findings (November 2026)

Runs of 2026-11-11 (k-means start of the GMM, K = 4, missing entries MCAR per entry at 10/30/50 %):

- **Estimation (`estimate.py`, `make_classification` with the fixed split, 5–5000 features, 2026-11-18; the numbers are from raw features (`_raw`), standardized ones (the default now) give the same picture: CFI 1e-15 to 8e-11, MC 1e-5 to 6e-4, the imputations slightly better at 50–1581 features):** CFI (40 frequencies) at KL 3e-11 to 8e-16 in every setting, an error of at most about 2e-7 in p (the KL is larger only where p is close to 0 or 1); conditional sampling (1000 samples) at 7e-6 to 5e-4 and 10–20× slower (2.3 s against 0.12 s at 5000 features); the imputations at 4e-4 to 1.1. Conditional mean imputation is no better than mean imputation from about 500 features: the diagonal GMM barely separates its components there (distance of the means relative to the spread 1.6 at 16 features, 0.11 at 5000), because the large redundant features hide the cluster structure. Frequencies: 32 reach the floor almost everywhere, 8 leave up to 2e-4, 16 up to 7e-6 at few features. The float64 posterior weights (see Pitfalls) doubled CFI's time at 5000 features on the RTX 3060, whose float64 is slow. Results of the earlier split (1/2 informative, no redundant) in `0.5informative_0.0redundant/`.
- **Results of earlier data splits** are kept in subfolders named by the split, in `code/figures/<experiment>/gmm/`, `thesis/figures/…`, `paper/figures/…` and `code/cache/<experiment>/gmm/`: `0.5informative_0.0redundant/` (estimation and training; the low-correlation control for the appendix) and `0.25informative_0.25redundant/` (training).
- **Training (`train.py`, the fixed split, standardized, 21 settings, 2026-11-18):** on incomplete test points PCLR has the lowest log-loss in 17 settings (conditional mean imputation 2, mean imputation 2); mean log-loss 0.440 (PCLR) / 0.446 (conditional mean) / 0.461 (mean imputation), accuracy 0.789 / 0.786 / 0.778. The lead is clearest at few features and high missing rates (8 features, 50 %: 0.334 / 0.365 / 0.406) and gone at 256 features, where the diagonal GMM barely separates its components. On complete test points PCLR's w is worse at 128–256 features and 30–50 % missing (log-loss 0.455 against 0.397 at 128 features, 50 %); the blow-up of the earlier raw run (1.3 and 1.85 at 256 features) is gone. Variants: all three within 0.001 on average (untrained start 0.4402, fitted start 0.4412, fixed density 0.4451); training the density helps only a little. Learning curve (100 features, 30 %): with 256 training points PCLR 0.642 against 0.839 (conditional mean) and 0.854 (mean imputation), from about 1000 points PCLR and conditional mean imputation are equal.
- **Training (`train.py`, `make_classification` with 1/4 informative, 1/4 redundant, the earlier split, 21 settings):** PCLR has the lowest log-loss on incomplete test points in 14 settings, conditional mean imputation in 7, mean imputation in none; mean log-loss 0.4766 / 0.4766 / 0.489, accuracy 0.765 / 0.763 / 0.752. At 256 features and 50 % missing PCLR from the untrained (k-means) start is worse (0.553 against 0.528 from the fitted start); averaged over all settings the fitted start is slightly better (0.4725), the untrained start wins more settings (10 against 8). Learning curve: with 256 training points PCLR 0.72 against 1.35 (mean imputation) and 1.58 (conditional mean imputation); from about 1000 points PCLR and conditional mean imputation are equal and ahead of mean imputation.
- **Real data (`real_data.py`):** PCLR is the best LR-based method on MAGIC, Phoneme and Pima at every rate and on Banknote at 50 %, within 0.005 of the best on Spambase (mean imputation), behind MICE on QSAR. The generative classifier (joint GMM over (x, y)) is best overall on Phoneme and on MAGIC at 50 %, and no longer blows up on Spambase and QSAR.

Earlier findings:

- **Estimation (`estimate.py`):** CFI is at the float32 limit (KL ≈ 1e-13 to 1e-15) for 5 to 5000 features and every missing rate, and about 10–20× faster than MC with 1000 samples (KL ≈ 1e-4). From about 50 features 16 frequencies are enough; few features need more (64 is safe).
- **Training, earlier run on LR-generated data (`load_lr_data`, removed; `train.py` is now on `make_classification` and not rerun yet):** PCLR's w is the closest to `w_true` of the methods on incomplete data up to about 16–64 features. With many features and few points per weight, conditional-mean imputation is about as good or better: it shrinks w (biased, but more stable), while PCLR is the consistent estimate and noisier with little data. On incomplete test points all methods are very close.
- **Convergence of w, same earlier data (removed with `w_true`):** PCLR's error falls steadily with the number of training points (0.80 → 0.12 from 2¹⁰ to 2¹⁵), conditional mean levels off at about 0.33, mean imputation at about 0.6. The crossover is at a few thousand points. Checked: with the *true* density instead of the fitted one the results are the same, so the density fit is not the cause; and PCLR's objective is lower at its w than at `w_true`, so it is not the optimizer.
- **Joint vs. fixed density:** almost identical (λ = 1, the density term dominates, and the GMM is the right model). Joint training only helped where the fixed density was poor (4–5 features at 80 % missing).
- **Earlier experiments:** on `make_classification` (Gaussian clusters per class) the generative classifier (one GMM over (x, y) with K components, each with a Bernoulli label leaf; p(y | x_obs) by Bayes' rule over the components; the same budget as PCLR's density. It replaced a GMM per class, which had twice the components and blew up on Spambase and QSAR) wins, because its model is exactly right there and an LR is not. On real data (`load_real`, K = 4) the results were mixed: PCLR best of the LR methods on MAGIC and Banknote, mean imputation best on Spambase (skewed count features, poor GMM fit) and Breast Cancer (too small). Real data should be reported as a table, not a figure.

### Open points

1. A PC as the density model (the paper is about PCs): its CF, `z_mean`, `z_var`, `conditional_mean`.
2. Fair regularization: C per method chosen on a validation split (now C = 1 everywhere, too weak for PCLR at many features and few points).
3. Several seeds for the final figures (now one seed; the curves over the number of features zigzag).
4. Real data as a table (`load_real`), with a larger K or the PC.
5. The prior-work approach (Khosravi et al., moments / Taylor series of the sigmoid) as a competitor in `estimate.py`.
6. Cleanups: old figures no longer generated (`figures/train/gmm/training_1.0.pdf`, `pclr_*` figures, `utils/figures/estimate/pghs.pdf`, `thesis/figures/pghs/`).
