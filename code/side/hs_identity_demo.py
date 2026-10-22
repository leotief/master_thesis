# the hubbard-stratonovich identity: how it works and how well the methods
# approximate it
#
#   e^{-a h^2 / 2} = E[e^{-i u h}],  u ~ N(0, a)
#
# this is the inner integral of pcpg (with a = gamma). it turns a gaussian in h
# into an average of waves e^{-i u h}, which are linear in h in the exponent.
# with u = sqrt(a) u' and t = sqrt(a) h it is the same for every a,
#   e^{-t^2 / 2} = E[e^{-i u t}] = E[cos(t u)],  u ~ N(0, 1)
# (the sin parts cancel, N(0, 1) is symmetric), so everything here uses
# N(0, 1). the expectation is approximated by picking a few points u_j with
# weights w_j (random samples or a quadrature rule from utils/hs_quadrature.py):
#   1. pick points u_j on N(0, 1), with weights w_j
#   2. every point gives a wave cos(t u_j) in t. points far out give fast waves
#   3. the weighted sum of the waves approximates the bell e^{-t^2 / 2}: near
#      t = 0 all waves are in phase, further out they have to cancel
#   4. a finite sum of waves never decays for good, so beyond some t they stop
#      cancelling and the error jumps up. more points push that t further out
#
# what t means in pcpg: there h = z - 1/(2 gamma), so for small gamma
# t ~ 1/(2 sqrt(gamma)), e.g. t ~ 3.5 for gamma = 0.02 and t ~ 7 for
# gamma = 0.005. the result is then multiplied by e^{1/(8 gamma)} ~ e^{t^2 / 2},
# so what matters there is the error relative to e^{-t^2 / 2}, not the absolute
# one. that is why small gamma needs many inner points.
#
# the figures go to figures/hs_identity/:
#   steps.pdf         the four steps for STEPS_METHOD with STEPS_N points
#   points.pdf        which points the different methods pick
#   mc_check.pdf      the identity with mc for growing sample sizes
#   comparison.pdf    all methods at N_NODES points, |t| <= T_MAX
#   convergence.pdf   mean absolute error vs number of points over t <= ERR_T
# the plotting itself is in utils/plots.py

import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from utils.hs_quadrature import hs_quadrature
from utils.plots import (blues, plot_error_vs_nodes,
                         plot_function_and_error, plot_hs_identity_steps,
                         plot_nodes_on_density)

OUT_DIR = "figures/hs_identity"
os.makedirs(OUT_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

# method -> label, colour, linestyle. "mc" draws random N(0, 1) samples, the
# others are the rules from utils/hs_quadrature.py. colours from the palette in
# utils/plots.py (see code/PLOTTING.md), the method we use (Gauss-Hermite) in
# the accent colour
METHODS = {
    "mc":             ("MC",             "0.55",    ":"),
    "qmc":            ("QMC (Halton)",   "0.15",    "-."),
    "trapezoid":      ("trapezoid",      "#0072B2", "-"),
    "gauss_legendre": ("Gauss-Legendre", "#E69F00", "-"),
    "gauss_hermite":  ("Gauss-Hermite",  "#D55E00", "-"),
}

# steps.pdf: method and number of points. points.pdf: number of points every
# method gets
STEPS_METHOD = "gauss_hermite"
STEPS_N      = 10

# accuracy plots. t range of the mc check and the comparison
T_MAX = 8
# mc_check: sample sizes, one set of curves each
N_SAMPLES = [10, 100, 1000, 10000]
# comparison: number of points every quadrature gets, mc gets N_MC samples
N_NODES = 20
N_MC    = 1000
# convergence: point counts on the x axis. mc uses the same numbers as samples
# (one run each). the error is the mean absolute error over t <= ERR_T
N_RANGE = [3, 5, 10, 15, 20, 30, 50]
ERR_T   = 6

# u axis and the N(0, 1) density on it
U       = np.linspace(-5, 5, 600)
DENSITY = np.exp(-U**2 / 2) / np.sqrt(2 * np.pi)

# ── core functions ────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def quadrature_nodes(method, n):
    return hs_quadrature(n, method)

# points and weights of any method, the weights sum to one. mc is n random
# samples with equal weights
def get_nodes(method, n, rng=None):
    if method == "mc":
        rng = rng if rng is not None else np.random.default_rng(0)
        return rng.standard_normal(n), np.full(n, 1.0 / n)
    return quadrature_nodes(method, n)

# e^{-t^2/2} = E[cos(t u)], with the expectation as a weighted sum over the points
def hs_estimate(t, nodes, weights):
    return weights @ np.cos(np.outer(nodes, t))

def exact(t):
    return np.exp(-t**2 / 2)

def mean_abs_error(t, nodes, weights):
    return np.abs(hs_estimate(t, nodes, weights) - exact(t)).mean()

# ── how the identity works ────────────────────────────────────────────────────

def plot_steps():
    t = np.linspace(-T_MAX, T_MAX, 800)
    nodes, weights = get_nodes(STEPS_METHOD, STEPS_N)
    waves = weights[:, None] * np.cos(np.outer(nodes, t))     # step 2, one row per point
    plot_hs_identity_steps(U, DENSITY, nodes, weights, t, waves, exact(t),
                           out_path=f"{OUT_DIR}/steps.pdf",
                           title=f"HS identity step by step ({METHODS[STEPS_METHOD][0]}, {len(nodes)} points)")

def plot_points():
    panels = [(METHODS[m][0], *get_nodes(m, STEPS_N)) for m in METHODS]
    plot_nodes_on_density(U, DENSITY, [(panels, "")], f"{OUT_DIR}/points.pdf",
                          title=f"Which points the methods pick ($N = {STEPS_N}$, height = weight)",
                          log=False, xlabel="$u$")

# ── how well the methods do ───────────────────────────────────────────────────

# does the identity hold? mc with growing sample sizes, a few runs each
def plot_mc_check(n_runs=3):
    t   = np.linspace(-T_MAX, T_MAX, 1600)
    rng = np.random.default_rng(0)
    colors = blues(len(N_SAMPLES))

    curves = []
    for color, n in zip(colors, N_SAMPLES):
        for run in range(n_runs):
            label = f"MC $N={n}$" if run == 0 else None
            curves.append((hs_estimate(t, *get_nodes("mc", n, rng)), label, color, "-", 0.6))

    plot_function_and_error(t, exact(t), curves, f"{OUT_DIR}/mc_check.pdf",
                            title=f"HS identity with MC ({n_runs} runs each)",
                            truth_label=r"exact $e^{-t^2/2}$", ylim=(-0.3, 1.1), xlabel="$t$")

# all methods with N_NODES points (mc with N_MC samples)
def plot_comparison():
    t   = np.linspace(-T_MAX, T_MAX, 1600)
    rng = np.random.default_rng(0)

    curves = []
    for method, (label, color, ls) in METHODS.items():
        n = N_MC if method == "mc" else N_NODES
        nodes, weights = get_nodes(method, n, rng)
        unit = "samples" if method == "mc" else "points"
        curves.append((hs_estimate(t, nodes, weights), f"{label} ({len(nodes)} {unit})", color, ls, 1.0))

    plot_function_and_error(t, exact(t), curves, f"{OUT_DIR}/comparison.pdf",
                            title=f"HS identity, $n={N_NODES}$ points",
                            truth_label=r"exact $e^{-t^2/2}$", ylim=(-0.3, 1.1), xlabel="$t$")

# mean absolute error over t <= ERR_T against the number of points
def plot_convergence():
    t   = np.linspace(0, ERR_T, 800)
    rng = np.random.default_rng(0)

    lines = []
    for method, (label, color, ls) in METHODS.items():
        errs = [mean_abs_error(t, *get_nodes(method, n, rng)) for n in N_RANGE]
        lines.append((N_RANGE, errs, label, color, ls))

    plot_error_vs_nodes([("All methods", lines)], f"{OUT_DIR}/convergence.pdf",
                        suptitle=rf"Mean absolute error over $t \leq {ERR_T}$", xticks=N_RANGE,
                        xlabel="number of points / samples", ylabel="mean absolute error")

if __name__ == "__main__":
    plot_steps()
    plot_points()
    plot_mc_check()
    plot_comparison()
    plot_convergence()
