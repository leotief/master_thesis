# how well cfi computes the expected prediction of a logistic regression when features of
# the test points are missing, against conditional sampling and two imputations: how
# accurate and how fast, from few to many features. two figures: the methods compared,
# and cfi with different numbers of frequencies

import itertools

import torch

from pclr import cfi
from utils.baselines import conditional_mean_imputation, conditional_sampling, mean_imputation, mixture_adaptive_quad
from utils.data import INFORMATIVE, REDUNDANT, load_data
from utils.experiments import compare_estimates, fit_density, logistic_regression, score_methods
from utils.plots import plot_cfi_frequencies, plot_estimate_comparison

# ── config ────────────────────────────────────────────────────────────────────

MODEL = "gmm"                                                   # the density model
K     = 4                                                       # its number of components

FEATURES      = [round(5 * 10 ** (k / 2)) for k in range(7)]    # numbers of features, evenly spaced on a log axis from 5 to 5000
MISSING_RATES = [0.1, 0.3, 0.5]                                 # how likely each entry of a test point is missing
SEED          = 0
NORM          = True                                            # standardize the features, as in every experiment

N_SAMPLES = 1000                                                # samples of conditional sampling
CFI_FREQS = 40                                                  # cfi's frequencies

CFI_FREQUENCIES = True                                          # draw the second figure
CFI_COUNTS      = [8, 16, 32, 64]                               # the numbers of frequencies in it

# the end of the file names, the settings the results depend on
NORM_SUFFIX = "" if NORM else "_raw"
ESTIMATORS_SUFFIX  = f"_{N_SAMPLES}samples_{CFI_FREQS}freqs{NORM_SUFFIX}"
FREQUENCIES_SUFFIX = NORM_SUFFIX

N_TEST       = 50                                               # test points with an exact value, which is slow
TIME_REPEATS = 1                                                # calls timed per method

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── the methods against the number of features ────────────────────────────────

# the error and time of every method in every setting, and of cfi with every number of frequencies
scores, cfi_scores = {}, {}

for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):

    x_train, _, y_train, _, x_test_missing, _ = load_data(n_features, missing_rate, INFORMATIVE, SEED, device, standardize=NORM, redundant=REDUNDANT)

    # the density and the logistic regression on the complete training points, so only
    # the test points have missing features
    px = fit_density(MODEL, x_train, f"cache/estimate/{MODEL}/K{K}_{n_features}features_{INFORMATIVE}informative_{REDUNDANT}redundant{NORM_SUFFIX}.pt", device, K)
    w = logistic_regression(x_train, y_train)

    # every method with what it needs: the density, or the training means for mean imputation, and its settings
    methods = {
        "Mean imputation":      (mean_imputation, x_train.mean(dim=0), {}),
        "Conditional mean imputation": (conditional_mean_imputation, px, {}),
        "Conditional sampling": (conditional_sampling, px, {"n_samples": N_SAMPLES}),
        "CFI":                  (cfi, px, {"n_freq": CFI_FREQS}),
    }

    # the exact values of the first test points, then every method's error on them and its time on all test points
    truths = mixture_adaptive_quad(px, w, x_test_missing[:N_TEST]).numpy()
    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'Reference':<24}{truths[0]:.6f}")
    scores[(n_features, missing_rate)] = score_methods(methods, w, x_test_missing, truths, TIME_REPEATS)

    if CFI_FREQUENCIES:
        counts = {count: (cfi, px, {"n_freq": count}) for count in CFI_COUNTS}
        cfi_scores[(n_features, missing_rate)] = score_methods(counts, w, x_test_missing, truths, TIME_REPEATS)

compare_estimates(plot_estimate_comparison, scores, FEATURES, MISSING_RATES, len(x_test_missing), f"estimate/{MODEL}", f"estimators{ESTIMATORS_SUFFIX}")
if CFI_FREQUENCIES:
    compare_estimates(plot_cfi_frequencies, cfi_scores, FEATURES, MISSING_RATES, len(x_test_missing), f"estimate/{MODEL}", f"cfi_frequencies{FREQUENCIES_SUFFIX}")
