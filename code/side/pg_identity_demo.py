# the polya-gamma identity: how it works and how well the methods approximate it
#
#   sigmoid(z) = 1/2 e^{z/2} E[e^{-gamma z^2 / 2}],  gamma ~ PG(1, 0)
#
# this is the outer integral of pcpg in one dimension. the expectation is
# approximated by picking a few points gamma_j with weights w_j (random samples
# or a quadrature rule from utils/quadrature.py, it works the same for both):
#   1. pick points gamma_j on the PG(1, 0) distribution, with weights w_j
#   2. every point gives a bell e^{-gamma_j z^2 / 2} in z. small gamma_j gives a
#      wide bell, large gamma_j a narrow one
#   3. the weighted sum of the bells approximates E[e^{-gamma z^2 / 2}], which
#      is exactly sech(z / 2), a bell itself
#   4. multiplying by 1/2 e^{z/2} tilts this bell into the sigmoid
# for large positive z the factor e^{z/2} is huge and amplifies the small error
# of the bell sum in its tail. pcpg avoids this with the sign flip
# sigmoid(z) = 1 - sigmoid(-z), see pg_estimate.
#
# the figures go to figures/pg_identity/:
#   steps.pdf     the four steps for STEPS_METHOD with STEPS_N points
#   points.pdf    which points the different methods pick
# and the accuracy plots, once with the flip in flip/ and once without in no_flip/:
#   mc_check.pdf              the identity with mc for growing sample sizes
#   <group>_comparison.pdf    the group at a fixed number of points, |z| <= Z_MAX
#   key_best_comparison.pdf   the key group, every method in its best setup (BEST)
#   convergence.pdf           mean absolute error vs number of points over
#                             |z| <= ERR_MAX, all methods and the key group
# the plotting itself is in utils/plots.py

import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from polyagamma import random_polyagamma

from utils.pg_density import pg_pdf
from utils.quadrature import pg_quadrature
from utils.plots import (blues, plot_error_vs_nodes,
                         plot_function_and_error, plot_nodes_on_density,
                         plot_pg_identity_steps)

OUT_DIR = "figures/pg_identity"

# ── config ────────────────────────────────────────────────────────────────────

# method -> label, colour, linestyle. "mc" draws random PG samples, the others
# are the rules from utils/quadrature.py. colours from the palette in
# utils/plots.py (see code/PLOTTING.md): greys for mc and the plain trapezoid,
# one colour per family, dashed for the log variant, pinks/greens for the fits
METHODS = {
    "mc":                 ("MC",                 "0.55",    ":"),
    "trapezoid":          ("trapezoid",          "0.15",    "-."),
    "log_trapezoid":      ("log-trapezoid",      "#0072B2", "-"),
    "gauss_legendre":     ("Gauss-Legendre",     "#E69F00", "-"),
    "log_gauss_legendre": ("log-Gauss-Legendre", "#E69F00", "--"),
    "golub_welsch":       ("Golub-Welsch",       "#D55E00", "-"),
    "log_golub_welsch":   ("log-Golub-Welsch",   "#D55E00", "--"),
    "sigmoid_fit_l2":     ("sigmoid fit (L2)",   "#CC79A7", "--"),
    "sigmoid_fit_l1":     ("sigmoid fit (L1)",   "#56B4E9", "--"),
    "sigmoid_fit_max":    ("sigmoid fit (max)",  "#009E73", "--"),
}

# steps.pdf and points.pdf: method and number of points for the steps, and the
# number of points every method gets in points.pdf
STEPS_METHOD = "golub_welsch"
STEPS_N      = 20

# layout of points.pdf: the three rules as columns, spaced in gamma in the
# first rule row and in log(gamma) right below, then the three sigmoid fits. mc
# sits on top as the random reference, None leaves a cell empty.
# why the rules spaced in gamma put their points further right:
#   - they are built on the plain gamma axis. trapezoid and gauss-legendre
#     spread their points evenly (or polynomial-like) over [0, 5], so only a few
#     land where the mass is, around 0.1 to 0.5. the log axis of the plot
#     stretches small gamma, so evenly spread points look bunched to the right
#   - golub-welsch is exact for the moments E[gamma^k]. high moments are
#     dominated by large gamma, so some points go out into the right tail. on
#     the left the density is extremely flat (like e^{-1/(8 gamma)}), there is
#     almost no mass below ~0.03 and nothing in the moments asks for points there
#   - the log rules spread their points over log(gamma) instead, so the left
#     side gets as many as the right. that is where e^{-gamma z^2 / 2} lives for
#     large |z|, which is why the log rules also work without the sign flip
ROWS = [
    ([None, "mc", None], "random"),
    (["trapezoid", "gauss_legendre", "golub_welsch"], "spaced in $\\gamma$"),
    (["log_trapezoid", "log_gauss_legendre", "log_golub_welsch"], "spaced in $\\log \\gamma$"),
    (["sigmoid_fit_l2", "sigmoid_fit_l1", "sigmoid_fit_max"], "fitted"),
]

# accuracy plots. z range of the mc check and the comparison plots
Z_MAX = 30
# mc_check: sample sizes, one set of curves each
N_SAMPLES = [10, 100, 1000, 10000]
# comparison plots: number of points every quadrature gets (golub-welsch max
# ~19, the sigmoid fits ignore it and pick their own). mc gets N_MC samples
N_NODES = 15
N_MC    = 1000
# convergence plot: point counts on the x axis. mc uses the same numbers as
# samples (one run each), golub-welsch skips the ones above 19, the sigmoid fits
# are single points. the error is the mean absolute error over |z| <= ERR_MAX
N_RANGE = [3, 5, 10, 15, 19, 30, 50]
ERR_MAX = 10

# group -> (methods, title), one comparison plot each
GROUPS = {
    "trapezoid":   (["trapezoid", "log_trapezoid"], "Trapezoid: linear vs log spacing"),
    "sigmoid_fit": (["sigmoid_fit_l2", "sigmoid_fit_l1", "sigmoid_fit_max"], "Sigmoid fit: loss"),
    "gauss":       (["gauss_legendre", "log_gauss_legendre", "golub_welsch", "log_golub_welsch"], "Gauss rules"),
    "key":         (["mc", "log_trapezoid", "sigmoid_fit_l2", "golub_welsch"], "Best of each family"),
    "all":         (list(METHODS), "All methods"),
}

# key_best_comparison: number of points (samples for mc) per method, the most
# each one gets in this demo. golub-welsch is capped at 19, the sigmoid fit
# picks its own (None)
BEST = {
    "mc":             100_000,
    "log_trapezoid":  50,
    "golub_welsch":   19,
    "sigmoid_fit_l2": None,
}

# gamma axes. the steps show step 1 on a plain axis with the PG(1, 0) density
# (almost all of its mass is below 3) and on a log axis, which shows the small
# points too (some methods go down to 1e-3). on a log axis the density of
# log(gamma) is p(gamma) * gamma, which is what is drawn, so the area under the
# curve is still the probability. points.pdf only uses the log axis
GAMMA       = np.linspace(0, 3, 600)
DENSITY     = pg_pdf(GAMMA)
GAMMA_LOG   = np.logspace(-3, 1, 600)
DENSITY_LOG = pg_pdf(GAMMA_LOG) * GAMMA_LOG

# ── core functions ────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# deterministic rules are cached, golub-welsch takes a few seconds per n
@lru_cache(maxsize=None)
def quadrature_nodes(method, n):
    return pg_quadrature(n, method)

# points and weights of any method, the weights sum to one. mc is n random
# samples with equal weights
def get_nodes(method, n, rng=None):
    if method == "mc":
        rng = rng if rng is not None else np.random.default_rng(0)
        return random_polyagamma(1.0, np.zeros(n), random_state=rng), np.full(n, 1.0 / n)
    return quadrature_nodes(method, n)

def fits_own_n(method):
    return method.startswith("sigmoid_fit")

# sigmoid(z) = 1/2 e^{z/2} E[e^{-gamma z^2 / 2}], with the expectation as a
# weighted sum over the points.
# with flip it uses sigmoid(z) = 1 - sigmoid(-z): the estimate is only hard for
# z > 0, where e^{z/2} amplifies errors, so it evaluates at -|z| and mirrors.
# without it shows how the methods behave on the right side
def pg_estimate(z, nodes, weights, flip):
    a = -np.abs(z) if flip else z
    bells = np.exp(-np.outer(nodes, a**2) / 2)              # (n, len(z))
    s = 0.5 * np.exp(a / 2) * (weights[:, None] * bells).sum(axis=0)
    return np.where(z > 0, 1 - s, s) if flip else s

def mean_abs_error(z, nodes, weights, flip):
    return np.abs(pg_estimate(z, nodes, weights, flip) - sigmoid(z)).mean()

# one folder per flip setting for the accuracy plots, same file names in both
def out_dir(flip=None):
    path = OUT_DIR if flip is None else f"{OUT_DIR}/{'flip' if flip else 'no_flip'}"
    os.makedirs(path, exist_ok=True)
    return path

# ── how the identity works ────────────────────────────────────────────────────

# the four steps for one method, without the flip
def plot_steps():
    z = np.linspace(-10, 10, 600)
    nodes, weights = get_nodes(STEPS_METHOD, STEPS_N)
    bells    = weights[:, None] * np.exp(-np.outer(nodes, z**2 / 2))   # step 2, one row per point
    estimate = 0.5 * np.exp(z / 2) * bells.sum(axis=0)                  # steps 3 and 4
    plot_pg_identity_steps(GAMMA, DENSITY, GAMMA_LOG, DENSITY_LOG, nodes, weights, z, bells,
                           exact_avg=1 / np.cosh(z / 2), estimate=estimate,
                           out_path=f"{out_dir()}/steps.pdf",
                           title=f"PG identity step by step ({METHODS[STEPS_METHOD][0]}, {len(nodes)} points)")

def plot_points():
    rows = [([(METHODS[m][0], *get_nodes(m, STEPS_N)) if m else None for m in methods], row_label)
            for methods, row_label in ROWS]
    plot_nodes_on_density(GAMMA_LOG, DENSITY_LOG, rows, f"{out_dir()}/points.pdf",
                          title=f"Which points the methods pick ($N = {STEPS_N}$, height = weight)")

# ── how well the methods do ───────────────────────────────────────────────────

# does the identity hold? mc with growing sample sizes, a few runs each, should
# approach the exact sigmoid. for large |z| mc gets much worse, because the
# factor e^{z/2} amplifies the noise in the average
def plot_mc_check(flip, n_runs=3):
    z   = np.linspace(-Z_MAX, Z_MAX, 1000)
    rng = np.random.default_rng(0)
    colors = blues(len(N_SAMPLES))

    curves = []
    for color, n in zip(colors, N_SAMPLES):
        for run in range(n_runs):
            label = f"MC $N={n}$" if run == 0 else None
            curves.append((pg_estimate(z, *get_nodes("mc", n, rng), flip), label, color, "-", 0.6))

    plot_function_and_error(z, sigmoid(z), curves, f"{out_dir(flip)}/mc_check.pdf",
                            title=f"PG-sigmoid identity with MC ({n_runs} runs each)")

# methods against the exact sigmoid on |z| <= Z_MAX. entries is a list of
# (method, n), n is the number of points (samples for mc, None for the sigmoid
# fits). for large |z| the bells e^{-gamma z^2 / 2} only reach that far for very
# small gamma, so this is where the methods differ most.
#
# what to expect for large z without the flip: the estimate is 1/2 e^{z/2} times
# a sum of bells, and staying at 1 needs the tail of that sum to be exactly
# 2 e^{-z/2}. only points at very small gamma reach that far out.
#   - trapezoid, gauss-legendre, golub-welsch: few or no points near 0, tail
#     too small, so the curve drops back to 0 (error at most 1)
#   - log rules: points near 0 with the right weights, stay at 1
#   - mc: depends on whether samples land near 0, so noisy
# with the flip all of them stay on the left side, where these errors are damped
def _comparison(entries, out_name, title, flip):
    z   = np.linspace(-Z_MAX, Z_MAX, 1000)
    rng = np.random.default_rng(0)

    curves = []
    for method, n in entries:
        label, color, ls = METHODS[method]
        nodes, weights = get_nodes(method, n, rng)
        unit = "samples" if method == "mc" else "points"
        curves.append((pg_estimate(z, nodes, weights, flip), f"{label} ({len(nodes)} {unit})", color, ls, 1.0))

    plot_function_and_error(z, sigmoid(z), curves, f"{out_dir(flip)}/{out_name}.pdf", title=title)

# a group with N_NODES points per quadrature and N_MC samples for mc
def plot_comparison(group, flip):
    methods, title = GROUPS[group]
    entries = [(m, N_MC if m == "mc" else None if fits_own_n(m) else N_NODES) for m in methods]
    _comparison(entries, f"{group}_comparison", f"{title}, $n={N_NODES}$ points", flip)

# the key group, every method with its point count from BEST
def plot_best_comparison(flip):
    _comparison(list(BEST.items()), "key_best_comparison", "Best of each family, best setup", flip)

# one line per method: mean absolute error over |z| <= ERR_MAX for every point
# count in N_RANGE (samples for mc, a single run each)
def _convergence_lines(methods, z, rng, flip):
    lines = []
    for method in methods:
        label, color, ls = METHODS[method]
        if fits_own_n(method):
            nodes, weights = get_nodes(method, None, rng)
            lines.append(([len(nodes)], [mean_abs_error(z, nodes, weights, flip)], label, color, ls))
            continue
        ns   = [n for n in N_RANGE if method != "golub_welsch" or n <= 19]
        errs = [mean_abs_error(z, *get_nodes(method, n, rng), flip) for n in ns]
        lines.append((ns, errs, label, color, ls))
    return lines

# convergence of the all group (left) and the key group (right) in one figure
def plot_convergence(flip):
    z   = np.linspace(-ERR_MAX, ERR_MAX, 1000)
    rng = np.random.default_rng(0)
    panels = [(GROUPS[g][1], _convergence_lines(GROUPS[g][0], z, rng, flip)) for g in ("all", "key")]
    plot_error_vs_nodes(panels, f"{out_dir(flip)}/convergence.pdf",
                        suptitle=rf"Mean absolute error over $|z| \leq {ERR_MAX}$", xticks=N_RANGE,
                        xlabel="number of points / samples", ylabel="mean absolute error")

if __name__ == "__main__":
    plot_steps()
    plot_points()
    for flip in (True, False):
        plot_mc_check(flip)
        for group in GROUPS:
            plot_comparison(group, flip)
        plot_best_comparison(flip)
        plot_convergence(flip)
