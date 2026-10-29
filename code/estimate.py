# estimating the expected prediction E[sigmoid(w^T x) | x_obs] of a logistic
# regression when features of the test points are missing
#
# the story, for every setting (number of features, missing rate):
#   setup       the training and test points, with missing_rate of the
#               features of every test point hidden. a density model p(x)
#               (MODEL: "gmm", "pc" a small rat-spn-like circuit written from
#               scratch, or "spflow" the rat-spn of the spflow library) and a
#               logistic regression w (sklearn, C = 1, no intercept: the
#               estimators only use w^T x), both fit on the complete training
#               points
#   estimation  every estimator averages the prediction over the missing
#               features under p(x_mis | x_obs), for all test points at once:
#                 PCPG             pcpg.py, the method of the thesis. polya-gamma
#                                  and hubbard-stratonovich turn the sigmoid into
#                                  a double integral over the model's
#                                  characteristic function, here by quadrature
#                                  (N_OUTER x N_INNER nodes)
#                 MC               the missing features sampled from the model
#                                  (as many samples as pcpg has nodes), the
#                                  predictions averaged
#                 Mean imputation  sigmoid(w^T x) with the gaps filled by the
#                                  training means
#   evaluation  the rmse over N_TEST test points against the exact value: z =
#               w^T x given x_obs is a mixture of gaussians (model.z_mixture),
#               and MixtureAdaptiveQuad integrates the sigmoid against every
#               component, exact up to ~1e-12. for the pcs every induced tree is
#               a component, so only small circuits
# with NORM the features are standardized and w is rescaled to norm 1: the
# logistic regression overfits many features and w grows, and the informative
# features of make_classification get a variance that grows with their number.
# both make z larger and pcpg's fixed nodes worse, NORM takes that away.
# at the end the comparison of all settings, as one figure, and optionally
# the four variants of pcpg (quadrature or mc over the outer and the inner
# integral) against their budget.
# the helpers are in experiment_utils.py

import itertools

import numpy as np
import torch

from experiment_utils import compare_estimates, fit_density, logistic_regression, rmse, timed_rmse
from pcpg import PCPG
from side.utils.baselines import MCBaseline, MeanImputation, MixtureAdaptiveQuad
from side.utils.data import load_data
from side.utils.plots import plot_pcpg_variants, save_figure

# ── config ────────────────────────────────────────────────────────────────────

# the density model, "gmm" or "spflow"
MODEL = "gmm"

# the settings: every number of features (INFORMATIVE of them informative, the
# others noise), evenly spaced on a log axis from 5 to 5000, with every
# fraction of missing features per test point. INFORMATIVE = None is
# sklearn's default: 2 informative and 2 redundant features, the rest noise.
FEATURES      = [round(5 * 10 ** (k / 2)) for k in range(7)]
MISSING_RATES = [0.3, 0.5, 0.8]
INFORMATIVE   = 0.5
SEED          = 0

# standardize the features and rescale w to norm 1 (True), or use both as they are (False)
NORM = True

# pcpg's nodes: the outer rule for gamma ~ PG(1, 0) and the inner gauss-hermite rule. mc gets the same budget
N_OUTER, N_INNER = 20, 60
BUDGET           = N_OUTER * N_INNER

SUFFIX = "_" + str(INFORMATIVE) + "_" + str(N_OUTER) + "_" + str(N_INNER) + ("_norm" if NORM else "")

# the l2 penalty of the logistic regression
C = 1.0

# the rmse is over the first N_TEST test points (the exact value is slow), the
# time is the median of TIME_REPEATS calls on all test points
N_TEST       = 50
TIME_REPEATS = 1

PCPG_VARIANTS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the scores of every estimator in every setting, scores[setting][estimator] = (rmse, time)
scores = {}

for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):

    # ── setup ─────────────────────────────────────────────────────────────────

    # the training points (complete) and the test points, with missing_rate of the features of every test point hidden (nan)
    x_train, _, y_train, _, x_test_missing, _ = load_data(n_features, missing_rate, INFORMATIVE, SEED, device, standardize=NORM)

    # fit on complete points. the density only depends on the number of features (and NORM)
    px = fit_density(MODEL, x_train, f"cache/estimate/{MODEL}/{n_features}features_{INFORMATIVE}informative{'_norm' if NORM else ''}.pt", device)
    w = logistic_regression(x_train, y_train, C)
    if NORM:
        w = w / w.norm()

    # the estimators: every one takes w and a batch of incomplete points (B, D) and returns E[sigmoid(w^T x) | x_obs] 
    estimators = {
        "Mean imputation":  MeanImputation(x_train.mean(dim=0)),
        "MC":               MCBaseline(px, n_samples=BUDGET),
        "PCPG":             PCPG(px, n_outer=N_OUTER, n_inner=N_INNER),
        "PCPG exact inner": PCPG(px, inner="exact", n_outer=N_OUTER),
    }

    # ── estimation and evaluation ─────────────────────────────────────────────

    # the exact values of the first N_TEST test points, and every estimator on
    # all test points, its rmse on those N_TEST and the time of the whole call
    setting = (n_features, missing_rate)
    scores[setting] = {}
    truths = MixtureAdaptiveQuad(px).estimate(w, x_test_missing[:N_TEST])
    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'Reference':<20}{truths[0]:.6f}")
    for name, estimator in estimators.items():
        error, seconds, estimates = timed_rmse(estimator, w, x_test_missing, truths, repeats=TIME_REPEATS)
        scores[setting][name] = (error, seconds)
        print(f"    {name:<20}{estimates[0]:.6f}        rmse {error:.1e}        time {seconds:.3f} s")

# ── comparison ────────────────────────────────────────────────────────────────

compare_estimates(scores, FEATURES, MISSING_RATES, len(x_test_missing), f"estimate/{MODEL}", f"comparison{SUFFIX}")

# ── the variants of pcpg ──────────────────────────────────────────────────────

if PCPG_VARIANTS:

    # pcpg with quadrature or mc over the outer and inner integral, rmse against budget

    # the variants of pcpg at the end (off by default): at MAIN_FEATURES features,
    # the budgets of the mc variants, the nodes of quadrature pcpg (n_inner =
    # 4 n_outer), and n_inner / n_outer of pcpg with mc over both
    MAIN_FEATURES = 15
    BUDGETS       = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
    QUAD_NODES    = [(n, 4 * n) for n in (2, 3, 4, 6, 8, 11, 16, 23, 32)]
    PCPG_RATIO    = 5

    # the node and sample counts of pcpg with mc over both at a budget, with
    # n_inner / n_outer = PCPG_RATIO
    def full_mc_counts(budget):
        n_outer = max(1, round(np.sqrt(budget / PCPG_RATIO)))
        return {"n_outer": n_outer, "n_inner": max(1, budget // n_outer)}


    results = []
    for missing_rate in MISSING_RATES:
        x_train, _, y_train, _, x_test_missing, _ = load_data(MAIN_FEATURES, missing_rate, INFORMATIVE, SEED, device, standardize=NORM)
        px = fit_density(MODEL, x_train, f"cache/estimate/{MODEL}/{MAIN_FEATURES}features_{INFORMATIVE}informative{'_norm' if NORM else ''}.pt", device)
        w = logistic_regression(x_train, y_train, C)
        if NORM:
            w = w / w.norm()
        points = x_test_missing[:N_TEST]
        truths = MixtureAdaptiveQuad(px).estimate(w, points)

        variants = {
            "PCPG":          (PCPG(px, pg_method=OUTER_RULE), {o * i: {"n_outer": o, "n_inner": i} for o, i in QUAD_NODES}),
            "PCPG outer MC": (PCPG(px, outer="mc"),
                              {b: {"n_outer": max(1, b // N_INNER), "n_inner": N_INNER} for b in BUDGETS}),
            "PCPG inner MC": (PCPG(px, inner="mc", pg_method=OUTER_RULE),
                              {b: {"n_outer": N_OUTER, "n_inner": max(1, b // N_OUTER)} for b in BUDGETS}),
            "PCPG full MC":  (PCPG(px, outer="mc", inner="mc"), {b: full_mc_counts(b) for b in BUDGETS}),
        }
        result = {}
        for name, (estimator, counts) in variants.items():
            result[name] = (list(counts), [rmse(estimator, w, points, truths, **c) for c in counts.values()])
        results.append(result)
    save_figure(plot_pcpg_variants, f"estimate/{MODEL}", f"pcpg_variants{SUFFIX}", MISSING_RATES, results)
