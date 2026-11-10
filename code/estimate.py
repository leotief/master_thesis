# estimating the expected prediction E[sigmoid(w^T x) | x_obs] of a logistic
# regression when features of the test points are missing: cfi against monte carlo
# (and mean imputation), how accurate and how fast. two figures, both error and time
# against the number of features:
#   comparison       the methods at fixed counts
#   cfi_frequencies  cfi with different numbers of frequencies
# the helpers are in experiment_utils.py

import itertools

import torch

from experiment_utils import compare_estimates, fit_density, logistic_regression, timed_kl
from pclr import cfi
from side.utils.baselines import conditional_mean, mean_imputation, mixture_adaptive_quad, monte_carlo
from side.utils.data import load_data
from side.utils.plots import plot_cfi_frequencies, save_figure

# ── config ────────────────────────────────────────────────────────────────────

# the density model
MODEL = "gmm"

# the settings: every number of features (INFORMATIVE of them informative, the
# others noise), evenly spaced on a log axis from 5 to 5000, with every
# fraction of missing features per test point
FEATURES      = [round(5 * 10 ** (k / 2)) for k in range(7)]
MISSING_RATES = [0.3, 0.5, 0.8]
INFORMATIVE   = 0.5
SEED          = 0

# standardize the features
NORM = False

# the counts of the comparison: mc's samples and cfi's frequencies
MC_SAMPLES = 1000
CFI_FREQS  = 40

# the numbers of frequencies of the second figure
CFI_COUNTS = [8, 16, 32, 64]

SUFFIX = "_" + str(INFORMATIVE) + "_" + str(MC_SAMPLES) + "_" + str(CFI_FREQS) + ("_norm" if NORM else "")

# the kl is over the first N_TEST test points (the exact value is slow), the
# time is the median of TIME_REPEATS calls on all test points
N_TEST       = 50
TIME_REPEATS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── comparison against the number of features ─────────────────────────────────

# the scores of every method in every setting, scores[setting][method] = (kl, time),
# and those of cfi with every number of frequencies, cfi_scores[setting][count] = (kl, time)
scores, cfi_scores = {}, {}

for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):

    # the training points (complete) and the test points, with missing_rate of the features of every test point hidden (nan)
    x_train, _, y_train, _, x_test_missing, _ = load_data(n_features, missing_rate, INFORMATIVE, SEED, device, standardize=NORM)

    # fit on complete points. the density only depends on the number of features (and NORM)
    px = fit_density(MODEL, x_train, f"cache/estimate/{MODEL}/{n_features}features_{INFORMATIVE}informative{'_norm' if NORM else ''}.pt", device)
    w = logistic_regression(x_train, y_train)

    # the methods, every one called as method(first, w, x_partial, **settings). first is the
    # density, the training means for mean imputation. name -> (method, first, settings)
    methods = {
        "Mean imputation":  (mean_imputation, x_train.mean(dim=0), {}),
        "Conditional mean": (conditional_mean, px, {}),
        "MC":               (monte_carlo, px, {"n_samples": MC_SAMPLES}),
        "CFI":              (cfi, px, {"n_freq": CFI_FREQS}),
    }

    # the exact values of the first N_TEST test points, and every method on all test
    # points, its kl on those N_TEST and the time of the whole call
    setting = (n_features, missing_rate)
    scores[setting] = {}
    truths = mixture_adaptive_quad(px, w, x_test_missing[:N_TEST]).numpy()
    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'Reference':<20}{truths[0]:.6f}")
    for name, (method, first, settings) in methods.items():
        error, seconds, estimates = timed_kl(method, first, w, x_test_missing, truths, repeats=TIME_REPEATS, **settings)
        scores[setting][name] = (error, seconds)
        print(f"    {name:<20}{estimates[0]:.6f}        kl {error:.1e}        time {seconds:.3f} s")

    # cfi with every number of frequencies, on the same points
    cfi_scores[setting] = {}
    for count in CFI_COUNTS:
        error, seconds, _ = timed_kl(cfi, px, w, x_test_missing, truths, repeats=TIME_REPEATS, n_freq=count)
        cfi_scores[setting][count] = (error, seconds)
        print(f"    {'CFI, ' + str(count) + ' frequencies':<20}{'':<8}        kl {error:.1e}        time {seconds:.3f} s")

compare_estimates(scores, FEATURES, MISSING_RATES, len(x_test_missing), f"estimate/{MODEL}", f"comparison{SUFFIX}")


# the second figure, errors and times: missing rate -> count -> one value per number of features
errors, times = {}, {}
for rate in MISSING_RATES:
    errors[rate], times[rate] = {}, {}
    for count in CFI_COUNTS:
        errors[rate][count] = [cfi_scores[(n, rate)][count][0] for n in FEATURES]
        times[rate][count] = [cfi_scores[(n, rate)][count][1] for n in FEATURES]
save_figure(plot_cfi_frequencies, f"estimate/{MODEL}", f"cfi_frequencies{SUFFIX}", FEATURES, MISSING_RATES,
            errors, times, len(x_test_missing))
