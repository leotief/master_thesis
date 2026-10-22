# quadrature rules for the inner integral of pcpg, over u ~ N(0, 1)
#
# hs_quadrature(n, method) returns n nodes and weights (weights sum to one) so
# that E[f(u)] ~ sum_j weights_j f(u_j) for u ~ N(0, 1). for u ~ N(0, a) the
# nodes are scaled by sqrt(a). methods:
#   "qmc"             quasi monte carlo: halton points through the normal
#                     inverse cdf, equal weights
#   "trapezoid"       equally spaced on the real line, weighted by the density
#   "gauss_legendre"  gauss-legendre on [-L, L], the density in the weights
#   "gauss_hermite"   gauss quadrature for the normal distribution
#
# in pcpg the function that gets averaged is a wave, e^{-itu} (only its real
# part cos(tu) is left, the rules are symmetric). the higher t, the faster it
# oscillates in u, so all rules need more nodes for larger t.

import numpy as np
from scipy.special import ndtri, roots_hermite, roots_legendre
from scipy.stats import qmc

# quasi monte carlo: instead of random uniforms, the halton sequence (in 1d the
# van der corput sequence), which fills [0, 1] evenly, pushed through the normal
# inverse cdf. equal weights like mc, but the error shrinks like ~1/n instead
# of 1/sqrt(n). scrambled with a fixed seed, so it is reproducible
def _qmc(n):
    p = qmc.Halton(d=1, scramble=True, seed=0).random(n)[:, 0]
    return ndtri(p), np.full(n, 1.0 / n)

# trapezoid rule on the whole real line: equally spaced nodes with spacing h,
# weighted by the normal density. for smooth integrands that decay like a
# gaussian this converges exponentially. two errors have to be balanced: the
# spacing (~ e^{-2 pi^2 / h^2}) and cutting off the range at +- n h / 2
# (~ e^{-(n h / 2)^2 / 2}). they are equal for h = sqrt(4 pi / n)
def _trapezoid(n):
    h = np.sqrt(4 * np.pi / n)
    nodes = (np.arange(n) - (n - 1) / 2) * h
    weights = np.exp(-nodes**2 / 2)
    return nodes, weights / weights.sum()

# gauss-legendre on [-L, L]: the standard nodes on [-1, 1] scaled by L, the
# normal density goes into the weights. the generic textbook rule, it knows
# nothing about the normal distribution (like gauss-legendre for pg). L = 6
# cuts off a mass of ~2e-9
_L = 6.0

def _gauss_legendre(n):
    x, w    = roots_legendre(n)
    nodes   = _L * x
    weights = w * np.exp(-nodes**2 / 2)
    return nodes, weights / weights.sum()

# gauss-hermite: exact for polynomials in u up to degree 2n - 1. the standard
# nodes are for the weight e^{-x^2}, u = sqrt(2) x turns them into nodes for
# N(0, 1), and dividing by sqrt(pi) normalises the weights
def _gauss_hermite(n):
    x, w = roots_hermite(n)
    return np.sqrt(2) * x, w / np.sqrt(np.pi)

def hs_quadrature(n, method="gauss_hermite"):
    if method == "qmc":
        return _qmc(n)
    if method == "trapezoid":
        return _trapezoid(n)
    if method == "gauss_legendre":
        return _gauss_legendre(n)
    if method == "gauss_hermite":
        return _gauss_hermite(n)
    raise ValueError(f"unknown method: {method}")
