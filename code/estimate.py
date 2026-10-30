# estimating the expected prediction E[sigmoid(w^T x) | x_obs] of a logistic
# regression when features of the test points are missing
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

# standardize the features and add stron L2 penalty
# if false both make z larger and pcpg's performance very much worse
NORM = True

# pcpg's nodes: the outer rule for gamma ~ PG(1, 0) and the inner gauss-hermite rule. mc gets the same budget
N_OUTER, N_INNER = 20, 60
BUDGET           = N_OUTER * N_INNER

SUFFIX = "_" + str(INFORMATIVE) + "_" + str(N_OUTER) + "_" + str(N_INNER) + ("_norm" if NORM else "")

# the l2 penalty of the logistic regression. decreasing it makes magnitude of w 
# smaller which makes pcpg's better
C = 0.001 if NORM else 1.0

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
    # if NORM:
    #     w = w / w.norm()

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

    MAIN_FEATURES = 15
    BUDGETS       = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
    QUAD_NODES    = [(n, 4 * n) for n in (2, 3, 4, 6, 8, 11, 16, 23, 32)]
    PCPG_RATIO    = 5  # n_inner / n_outer of pcpg with mc over both

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
            "PCPG":          (PCPG(px), {o * i: {"n_outer": o, "n_inner": i} for o, i in QUAD_NODES}),
            "PCPG outer MC": (PCPG(px, outer="mc"),
                              {b: {"n_outer": max(1, b // N_INNER), "n_inner": N_INNER} for b in BUDGETS}),
            "PCPG inner MC": (PCPG(px, inner="mc"),
                              {b: {"n_outer": N_OUTER, "n_inner": max(1, b // N_OUTER)} for b in BUDGETS}),
            "PCPG full MC":  (PCPG(px, outer="mc", inner="mc"), {b: full_mc_counts(b) for b in BUDGETS}),
        }
        result = {}
        for name, (estimator, counts) in variants.items():
            result[name] = (list(counts), [rmse(estimator, w, points, truths, **c) for c in counts.values()])
        results.append(result)
    save_figure(plot_pcpg_variants, f"estimate/{MODEL}", f"pcpg_variants{SUFFIX}", MISSING_RATES, results)
