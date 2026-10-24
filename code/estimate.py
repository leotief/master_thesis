# estimating the expected prediction E[sigmoid(w^T x) | x_obs] of a logistic
# regression when features are missing
#
# step by step, for one number of features:
#   1. data        training and test points
#   2. models      a density model p(x) (MODEL: "gmm", "pc" a small rat-spn-like
#                  circuit written from scratch, or "spflow" the rat-spn of the
#                  spflow library) and a logistic regression w (sklearn, C = 1,
#                  no intercept: the estimators only use w^T x), both fit on the
#                  complete training points
#   3. missing     missing_rate of the features of every test point are hidden
#   4. estimation  every estimator averages the prediction over the missing
#                  features under p(x_mis | x_obs):
#                    PCPG             pcpg.py, the method of the thesis.
#                                     polya-gamma and hubbard-stratonovich turn
#                                     the sigmoid into a double integral over
#                                     the model's characteristic function, done
#                                     by quadrature (deterministic) or by mc
#                    MC               the missing features sampled from the
#                                     model, the predictions averaged
#                    Mean imputation  sigmoid(w^T x) with the gaps filled by the
#                                     training means
#   5. error       the rmse over N_TEST test points against the exact value:
#                  z = w^T x given x_obs is a mixture of gaussians
#                  (model.z_moments), and MixtureAdaptiveQuad integrates the
#                  sigmoid against every component, exact up to ~1e-12. for the
#                  pcs every induced tree is a component, so only small circuits
# "budget" is the number of cf evaluations (samples for mc), so the methods
# are compared at the same cost.
#
# experiments, every one saves its figure in figures/estimate/<MODEL>/ and
# thesis/figures/estimate/<MODEL>/:
#   comparison             the estimators against the number of features, one
#                          column per missing rate: the error, and below it the
#                          time of one call on the whole test set
#   comparison_fixed_norm  the same with w rescaled to norm 1. the logistic
#                          regression overfits many features and w grows, which
#                          makes z larger and pcpg's fixed nodes worse, this
#                          takes that away
#   pcpg_variants          the error of the four pcpg variants (quadrature or mc
#                          over the outer and the inner integral) against the
#                          budget, at MAIN_FEATURES features
# the plotting is in side/utils/plots.py

import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from pcpg import PCPG
from side.utils.analyses import rmse, timed_rmse
from side.utils.baselines import MCBaseline, MeanImputation, MixtureAdaptiveQuad
from side.utils.data import DataManager
from side.utils.gmm import GMM
from side.utils.pc import PC
from side.utils.spflow_pc import SPFlowPC
from side.utils.plots import plot_estimate_comparison, plot_pcpg_variants, save_figure

CACHE_DIR = "cache/estimate"

# ── config ────────────────────────────────────────────────────────────────────

# the density model, "gmm", "pc" or "spflow"
MODEL = "gmm"

SEED   = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# numbers of features of the comparison, evenly spaced on a log axis (a factor
# sqrt(10) apart, 5 to 5000), and a low / mid / high missing rate. more missing
# features means the cf is damped more, which makes pcpg easier
FEATURES      = [round(5 * 10 ** (k / 2)) for k in range(7)]
MISSING_RATES = [0.3, 0.5, 0.8]
MAIN_FEATURES = 15
N_TEST        = 50       # test points with an exact value per setting
# the fraction of informative features, the others are noise
INFORMATIVE   = 0.5

# the density models: K gmm components, and the circuit of both pcs
K              = 4
PC_CHANNELS    = 4
PC_REPETITIONS = 3
PC_DEPTH       = 2

# nodes of deterministic pcpg: the outer golub-welsch pg rule (only reliable up
# to ~19) and the inner gauss-hermite rule. mc gets the same budget
N_OUTER, N_INNER = 19, 64
BUDGET           = N_OUTER * N_INNER

# pcpg_variants: the budgets of the mc variants, the nodes of deterministic
# pcpg (n_inner = 4 n_outer), and n_inner / n_outer of pcpg with mc over both
BUDGETS    = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
QUAD_NODES = [(n, 4 * n) for n in (2, 3, 4, 6, 8, 11, 15, 19)]
PCPG_RATIO = 5

# the time of the comparison is the median of TIME_REPEATS calls on the whole test set
TIME_REPEATS = 3

# ── 1. data ───────────────────────────────────────────────────────────────────

# the training points and labels, and the test points, as tensors
def load_data(n_features):
    dm = DataManager("sk_class", samples=10_000, features=n_features, test_size=0.2, random_state=SEED,
                     informative=INFORMATIVE)
    x_train, x_test, y_train, _ = dm.get_data()
    return (torch.tensor(x_train, device=device), torch.tensor(y_train, device=device),
            torch.tensor(x_test, device=device))

# ── 2. models ─────────────────────────────────────────────────────────────────

# the density model fit on the training points, saved and loaded from the cache
def fit_density(x_train):
    n_features = x_train.shape[1]
    path = f"{CACHE_DIR}/density/{MODEL}_{n_features}features.pt"
    torch.manual_seed(SEED)
    if MODEL == "gmm":
        px = GMM(device, K=K, means_init=x_train[:K], n_features=n_features).to(device)
    elif MODEL == "pc":
        px = PC(device, n_features, channels=PC_CHANNELS, repetitions=PC_REPETITIONS, depth=PC_DEPTH)
    else:
        px = SPFlowPC(device, n_features, channels=PC_CHANNELS, repetitions=PC_REPETITIONS, depth=PC_DEPTH)
    if os.path.exists(path):
        px.load_state_dict(torch.load(path, map_location=device))
    else:
        dataset = torch.utils.data.TensorDataset(x_train)
        px.fit(torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(px.state_dict(), path)
    return px

# the logistic regression on the training points, without an intercept
def fit_classifier(x_train, y_train):
    classifier = LogisticRegression(fit_intercept=False, max_iter=1000).fit(x_train.cpu(), y_train.cpu())
    return torch.tensor(classifier.coef_[0], dtype=torch.float32, device=device)

# ── 3. missing features ───────────────────────────────────────────────────────

# the test points with missing_rate of the features of every point set to nan,
# chosen at random for every point, the same for every method
def hide(x_test, missing_rate):
    rng = np.random.default_rng(SEED)
    x = x_test.clone()
    n_hidden = int(missing_rate * x.shape[1])
    for row in x:
        row[torch.from_numpy(rng.choice(len(row), n_hidden, replace=False))] = float("nan")
    return x

# ── 4. the estimators ─────────────────────────────────────────────────────────

# what is compared. every estimator takes w and a batch of incomplete points
# (B, D) and returns E[sigmoid(w^T x) | x_obs] for all of them, (B,)
def estimators(px, x_train):
    return {
        "Mean imputation": MeanImputation(x_train.mean(dim=0)),
        "MC":              MCBaseline(px, n_samples=BUDGET),
        "PCPG":            PCPG(px, n_outer=N_OUTER, n_inner=N_INNER),
    }

# the node or sample counts of pcpg with mc over both at a budget, with
# n_inner / n_outer = PCPG_RATIO
def full_mc_counts(budget):
    n_outer = max(1, round(np.sqrt(budget / PCPG_RATIO)))
    return {"n_outer": n_outer, "n_inner": max(1, budget // n_outer)}

# pcpg with quadrature or mc over the outer integral (gamma ~ PG(1, 0)) and the
# inner one (u ~ N(0, gamma)), each as (estimator, {budget: counts}). the
# quadrature part of a mixed variant keeps the nodes of the comparison, the mc
# part gets the rest of the budget
def pcpg_variants_of(px):
    return {
        "PCPG":          (PCPG(px), {o * i: {"n_outer": o, "n_inner": i} for o, i in QUAD_NODES}),
        "PCPG outer MC": (PCPG(px, outer="mc"),
                          {b: {"n_outer": max(1, b // N_INNER), "n_inner": N_INNER} for b in BUDGETS}),
        "PCPG inner MC": (PCPG(px, inner="mc"),
                          {b: {"n_outer": N_OUTER, "n_inner": max(1, b // N_OUTER)} for b in BUDGETS}),
        "PCPG full MC":  (PCPG(px, outer="mc", inner="mc"), {b: full_mc_counts(b) for b in BUDGETS}),
    }

# ── 5. error ──────────────────────────────────────────────────────────────────

# the exact values of the first N_TEST incomplete test points
def exact_values(px, w, points):
    return MixtureAdaptiveQuad(px)(w, points[:N_TEST])

# ── experiments ───────────────────────────────────────────────────────────────

# error and time of the estimators for every number of features and missing
# rate. the time is one batched call on the whole test set, the error is over
# its first N_TEST points. every setting is printed when it is done: the exact
# value and every estimate of the first test point, the rmse and the time.
# fixed_norm uses w rescaled to norm 1 and saves comparison_fixed_norm
def comparison(fixed_norm=False):
    figure = "comparison_fixed_norm" if fixed_norm else "comparison"
    errors = {rate: {} for rate in MISSING_RATES}
    times = {rate: {} for rate in MISSING_RATES}
    print(f"{figure} ({MODEL}, budget {BUDGET}, time for the whole test set)")
    for n_features in FEATURES:
        x_train, y_train, x_test = load_data(n_features)
        px = fit_density(x_train)
        w = fit_classifier(x_train, y_train)
        if fixed_norm:
            w = w / w.norm()
        for rate in MISSING_RATES:
            points = hide(x_test, rate)
            truths = exact_values(px, w, points)
            print(f"\n{n_features} features, {rate:.0%} missing")
            print(f"    {'Reference':<20}{truths[0]:.6f}")
            for name, estimator in estimators(px, x_train).items():
                error, seconds, estimates = timed_rmse(estimator, w, points, truths, repeats=TIME_REPEATS)
                errors[rate].setdefault(name, []).append(error)
                times[rate].setdefault(name, []).append(seconds)
                print(f"    {name:<20}{estimates[0]:.6f}        rmse {error:.1e}        time {seconds:.3f} s")
    save_figure(plot_estimate_comparison, f"estimate/{MODEL}", figure,
                FEATURES, MISSING_RATES, errors, times, len(x_test))

# error against budget of the four pcpg variants, one panel per missing rate
def pcpg_variants():
    x_train, y_train, x_test = load_data(MAIN_FEATURES)
    px = fit_density(x_train)
    w = fit_classifier(x_train, y_train)
    results = []
    for rate in MISSING_RATES:
        points = hide(x_test, rate)[:N_TEST]
        truths = exact_values(px, w, points)
        result = {}
        for name, (estimator, counts) in pcpg_variants_of(px).items():
            result[name] = (list(counts), [rmse(estimator, w, points, truths, **c) for c in counts.values()])
        results.append(result)
    save_figure(plot_pcpg_variants, f"estimate/{MODEL}", "pcpg_variants", MISSING_RATES, results)

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    comparison()
    comparison(fixed_norm=True)
    # pcpg_variants()
