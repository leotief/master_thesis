# estimating the expected prediction E[sigmoid(w^T x) | x_obs] of a logistic
# regression when features of the test points are missing: cfi against conditional
# sampling (monte carlo), mean imputation and the conditional mean, how accurate and how
# fast, on sklearn's make_classification (utils/data.py). two figures, both error and
# time against the number of features:
#   estimators       the methods at fixed counts
#   cfi_frequencies  cfi with different numbers of frequencies (only with CFI_FREQUENCIES)
# the helpers are in utils/experiments.py

import itertools

import torch

from pclr import cfi
from utils.baselines import conditional_mean_imputation, conditional_sampling, mean_imputation, mixture_adaptive_quad
from utils.data import load_data
from utils.experiments import compare_estimates, fit_density, logistic_regression, score_methods
from utils.plots import plot_cfi_frequencies, plot_estimate_comparison

# ── config ────────────────────────────────────────────────────────────────────

# the density model and its number of components
MODEL = "gmm"
K     = 4

# the settings: every number of features (INFORMATIVE of them informative, the
# others noise), evenly spaced on a log axis from 5 to 5000, with every
# probability of an entry of a test point to be missing
FEATURES      = [round(5 * 10 ** (k / 2)) for k in range(7)]
MISSING_RATES = [0.1, 0.3, 0.5]
INFORMATIVE   = 0.5
SEED          = 0

# standardize the features (cfi does not need it: it adapts to the spread of z)
NORM = False

# the counts of the comparison: the samples of conditional sampling and cfi's frequencies
N_SAMPLES = 1000
CFI_FREQS = 40

# the second figure, cfi with every number of frequencies of CFI_COUNTS (False skips it)
CFI_FREQUENCIES = True
CFI_COUNTS      = [8, 16, 32, 64]

# the end of the names of the figures: the settings they depend on, every value with its name
NORM_SUFFIX = "_norm" if NORM else ""
ESTIMATORS_SUFFIX  = f"_{INFORMATIVE}informative_{N_SAMPLES}samples_{CFI_FREQS}freqs{NORM_SUFFIX}"
FREQUENCIES_SUFFIX = f"_{INFORMATIVE}informative{NORM_SUFFIX}"

# the kl is over the first N_TEST test points (the exact value is slow), the
# time is the median of TIME_REPEATS calls on all test points
N_TEST       = 50
TIME_REPEATS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── the methods against the number of features ────────────────────────────────

# the scores of every method in every setting, scores[setting][method] = (kl, time),
# and those of cfi with every number of frequencies, cfi_scores[setting][count] = (kl, time)
scores, cfi_scores = {}, {}

for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):

    # the training points (complete) and the test points, every entry hidden (nan) with probability missing_rate
    x_train, _, y_train, _, x_test_missing, _ = load_data(n_features, missing_rate, INFORMATIVE, SEED, device, standardize=NORM)

    # fit on complete points. the density only depends on the number of features (and NORM)
    px = fit_density(MODEL, x_train, f"cache/estimate/{MODEL}/K{K}_{n_features}features_{INFORMATIVE}informative{NORM_SUFFIX}.pt", device, K)
    w = logistic_regression(x_train, y_train)

    # the methods, every one called as method(first, w, x_partial, **settings). first is the
    # density, the training means for mean imputation. name -> (method, first, settings)
    methods = {
        "Mean imputation":      (mean_imputation, x_train.mean(dim=0), {}),
        "Conditional mean imputation": (conditional_mean_imputation, px, {}),
        "Conditional sampling": (conditional_sampling, px, {"n_samples": N_SAMPLES}),
        "CFI":                  (cfi, px, {"n_freq": CFI_FREQS}),
    }

    # the exact values of the first N_TEST test points, then every method on all test
    # points: its kl on those N_TEST and the time of the whole call
    truths = mixture_adaptive_quad(px, w, x_test_missing[:N_TEST]).numpy()
    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'Reference':<24}{truths[0]:.6f}")
    scores[(n_features, missing_rate)] = score_methods(methods, w, x_test_missing, truths, TIME_REPEATS)

    # cfi with every number of frequencies, on the same points
    if CFI_FREQUENCIES:
        counts = {count: (cfi, px, {"n_freq": count}) for count in CFI_COUNTS}
        cfi_scores[(n_features, missing_rate)] = score_methods(counts, w, x_test_missing, truths, TIME_REPEATS)

compare_estimates(plot_estimate_comparison, scores, FEATURES, MISSING_RATES, len(x_test_missing), f"estimate/{MODEL}", f"estimators{ESTIMATORS_SUFFIX}")
if CFI_FREQUENCIES:
    compare_estimates(plot_cfi_frequencies, cfi_scores, FEATURES, MISSING_RATES, len(x_test_missing), f"estimate/{MODEL}", f"cfi_frequencies{FREQUENCIES_SUFFIX}")
