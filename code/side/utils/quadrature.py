# quadrature rules for the two integrals of pcpg: pg_quadrature for the outer
# one over omega ~ PG(1, 0), hs_quadrature for the inner one over u ~ N(0, 1)
#
# ── outer integral over omega ~ PG(1, 0) ──
#
# pg_quadrature(n, method) returns n nodes and weights (weights sum to one) so
# that E[f(omega)] ~ sum_j weights_j f(nodes_j). methods:
#   "trapezoid"       composite trapezoid rule on [0, T], the simple baseline
#   "log_trapezoid"   trapezoid rule in log omega
#   "gauss_legendre"  gauss-legendre on [0, T]
#   "log_gauss_legendre"  gauss-legendre in log omega
#   "golub_welsch"    gauss quadrature with PG(1, 0) itself as the weight, via
#                     golub-welsch, n <= ~19
#   "log_golub_welsch"  the same for log(omega), with the distribution of
#                     log(omega) as the weight
#   "sigmoid_fit_l2"  weights fitted so that the identity gives the sigmoid on
#   "sigmoid_fit_l1"  z <= 5, for use with the sign flip, ignores n. the suffix
#   "sigmoid_fit_max" is the norm of the error curve that is minimised
#
# all rules except golub_welsch know nothing about PG and need its density
# pointwise. golub_welsch only needs its moments. pg_reference(f) is not a rule
# but adaptive gauss-kronrod (scipy quad) on the density, for reference values
# of E[f(omega)]
#
# in pcpg the function of omega that gets integrated is (a mixture of)
# exp(-omega z^2 / 2). for large |z| this is a sharp peak at small omega, so the
# rules differ mostly in how well they cover that region.
#
# what this means for the sigmoid: the estimate is 1/2 e^{z/2} times a sum of
# bells e^{-omega_j z^2 / 2}, and it only stays at 1 for large z if the tail of
# that sum is exactly 2 e^{-z/2}. only nodes at very small omega reach that far.
#   - trapezoid, gauss_legendre, golub_welsch: few or no nodes near 0,
#     tail too small, the estimate drops back to 0 for large z (error at most 1)
#   - log_trapezoid: nodes near 0 with the right weights, stays accurate
# a rule that puts more weight at tiny omega than PG really has there does the
# opposite: the tail is too heavy and the estimate blows up like e^{z/2}.
#
# all of this goes away with sigmoid(z) = 1 - sigmoid(-z): evaluating at -|z| and
# mirroring keeps every rule on the left side, where e^{z/2} shrinks errors
# instead of amplifying them. the estimators in pcpg.py and the demo do that
#
# ── inner integral over u ~ N(0, 1) ──
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

from functools import lru_cache

import mpmath
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog, nnls
from scipy.sparse import bmat, csr_matrix, identity
from scipy.special import ndtri, roots_hermite, roots_legendre
from scipy.stats import qmc

from .pg_density import pg_pdf

# ═══ outer: PG(1, 0) ══════════════════════════════════════════════════════════

# ── trapezoid rules ───────────────────────────────────────────────────────────

# right end of the truncated interval for trapezoid and gauss_legendre. the
# mass above it is about 4/pi e^{-pi^2 T / 2} ~ 2e-11
_PG_T = 5.0

# composite trapezoid rule on [0, T] with n equally spaced nodes:
#   E[f(omega)] ~ sum_j h c_j p(omega_j) f(omega_j),  c_j = 1/2 at the ends, else 1
# the simplest rule there is. the first node is at 0 and gets weight 0, and with
# spacing T / (n - 1) only few nodes fall where the mass is (around 0.25)
def _pg_trapezoid(n):
    nodes, h = np.linspace(0, _PG_T, n, retstep=True)
    weights  = pg_pdf(nodes) * h
    weights[[0, -1]] /= 2
    return nodes, weights / weights.sum()

# the same rule in s = log(omega): with omega = exp(s),
#   E[f(omega)] = integral f(exp(s)) p(exp(s)) exp(s) ds
# so equally spaced s nodes get the weight p(exp(s)) exp(s) h. this puts many
# nodes at small omega, where exp(-omega z^2 / 2) lives for large |z|, and for
# smooth integrands that vanish at both ends it converges exponentially.
# [1e-3, 10] covers all the mass: p decays like exp(-1/(8 omega)) towards 0
# and like exp(-pi^2 omega / 2) towards infinity. the ends get no factor 1/2,
# their weights are negligible anyway
_LOG_RANGE = (np.log(1e-3), np.log(10.0))

def _pg_log_trapezoid(n):
    s, h    = np.linspace(*_LOG_RANGE, n, retstep=True)
    nodes   = np.exp(s)
    weights = pg_pdf(nodes) * nodes * h
    return nodes, weights / weights.sum()

# ── gauss rules ───────────────────────────────────────────────────────────────

# gauss-legendre on [0, T]: the standard nodes x_j on [-1, 1] mapped to
# omega_j = T (x_j + 1) / 2, the density goes into the weights. exact for
# polynomials times the density, which is not what we integrate, so it does not
# get the gauss rate for E[f(omega)]
def _pg_gauss_legendre(n):
    x, w    = roots_legendre(n)
    nodes   = _PG_T * (x + 1) / 2
    weights = w * _PG_T / 2 * pg_pdf(nodes)
    return nodes, weights / weights.sum()

# the same in s = log(omega), on the same range as log_trapezoid: legendre nodes
# mapped to [log 1e-3, log 10], weights times p(exp(s)) exp(s) like there
def _pg_log_gauss_legendre(n):
    x, w    = roots_legendre(n)
    lo, hi  = _LOG_RANGE
    nodes   = np.exp(lo + (hi - lo) * (x + 1) / 2)
    weights = w * (hi - lo) / 2 * pg_pdf(nodes) * nodes
    return nodes, weights / weights.sum()

# gauss quadrature for PG(1, 0): exact for polynomials in omega up to degree 2n - 1.
# there is no standard rule for the pg distribution, so it is built from its
# moments with the golub-welsch algorithm:
#   moments -> hankel matrix -> cholesky -> jacobi matrix -> eigen decomposition
# the moments come from the laplace transform L(t) = E[exp(-t omega)] = sech(sqrt(t/2)).
# runs in mpmath with 200 digits because the hankel matrix is extremely ill
# conditioned (float64 breaks down around n = 12). takes a few seconds, so
# every n is only computed once
@lru_cache(maxsize=None)
def _pg_golub_welsch(n):
    mpmath.mp.dps = 200

    def laplace_transform(t):
        return 1 / mpmath.cosh(mpmath.sqrt(t / 2))

    # E[omega^k] = (-1)^k L^(k)(0), 2n moments are needed for an n point rule
    moments = [mpmath.re((-1) ** k * mpmath.diff(laplace_transform, mpmath.mpf(0), k)) for k in range(2 * n)]

    # H[i, j] = E[omega^(i+j)] and H1[i, j] = E[omega^(i+j+1)]
    H  = mpmath.matrix([[moments[i + j]     for j in range(n)] for i in range(n)])
    H1 = mpmath.matrix([[moments[i + j + 1] for j in range(n)] for i in range(n)])

    # jacobi matrix of the orthogonal polynomials, by changing H1 to the
    # orthonormal basis given by the cholesky factor of H. its eigenvalues are
    # the nodes, the squared first entries of the eigenvectors the weights
    Rinv = mpmath.inverse(mpmath.cholesky(H).T)
    J = Rinv.T * H1 * Rinv
    J = (J + J.T) / 2                                  # clean up rounding

    eigenvalues, eigenvectors = mpmath.eigh(J)
    nodes   = np.array([float(v) for v in eigenvalues])
    weights = np.array([float(eigenvectors[0, k] ** 2) for k in range(n)])
    return nodes, weights

# gauss quadrature for log(omega): the same idea as golub_welsch, but in
# s = log(omega), with the distribution of s as the weight. exact for
# polynomials in s up to degree 2n - 1. the moments of s have no closed form, so
# the jacobi matrix comes from the lanczos (stieltjes) procedure on a fine
# discretisation of that distribution instead: 4000 trapezoid points in s, the
# same discretisation as log_trapezoid with many nodes. lanczos builds the
# orthonormal polynomials point by point (with reorthogonalisation, so it stays
# stable), and their recurrence coefficients are the jacobi matrix. the last
# step, nodes and weights from its eigendecomposition, is the same as golub_welsch
def _pg_log_golub_welsch(n):
    s, h = np.linspace(np.log(1e-4), np.log(20.0), 4000, retstep=True)
    c = pg_pdf(np.exp(s)) * np.exp(s) * h
    c /= c.sum()

    # lanczos on the vectors sqrt(c) * p_k(s), so plain dot products are the
    # inner products of the discretised distribution
    Q = np.zeros((n, len(s)))
    q, q_prev, b_prev = np.sqrt(c), 0.0, 0.0
    a, b = np.zeros(n), np.zeros(n)
    for k in range(n):
        Q[k] = q
        v = s * q - b_prev * q_prev
        a[k] = q @ v
        v -= a[k] * q
        v -= Q[:k + 1].T @ (Q[:k + 1] @ v)             # reorthogonalise
        b[k] = np.linalg.norm(v)
        q_prev, q, b_prev = q, v / b[k], b[k]

    J = np.diag(a) + np.diag(b[:-1], 1) + np.diag(b[:-1], -1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    return np.exp(eigenvalues), eigenvectors[0] ** 2

# ── fitted rule ───────────────────────────────────────────────────────────────

# fits nonnegative weights w that sum to one so that A w ~ b, where row i of A
# is the approximation at one grid point z_i and b the target there. loss is the
# norm of the error curve A w - b that gets minimised:
#   "l2"   sum of squared errors, nonnegative least squares. the sum to one
#          constraint is an extra row with weight 10 (much larger and nnls
#          does not converge, the fit is close to summing to one anyway)
#   "l1"   sum of absolute errors, i.e. the area under the error curve
#   "max"  largest absolute error, makes the error curve equally high everywhere
# l1 and max are linear programs: with slacks e_i >= |(A w - b)_i| (one per row
# for l1, a single shared one for max) minimise sum e_i or e.
# all three keep only a few columns, so the number of nodes is not chosen
# directly
def fit_nonnegative(A, b, loss="l2"):
    n_rows, n_cols = A.shape
    if loss == "l2":
        big = 10.0
        w, _ = nnls(np.vstack([A, big * np.ones(n_cols)]), np.append(b, big), maxiter=20_000)
    elif loss in ("l1", "max"):
        n_slack = n_rows if loss == "l1" else 1
        slack   = -identity(n_rows) if loss == "l1" else csr_matrix(-np.ones((n_rows, 1)))
        # A w - b <= e  and  b - A w <= e
        A_ub = bmat([[csr_matrix(A), slack], [csr_matrix(-A), slack]])
        b_ub = np.concatenate([b, -b])
        c    = np.concatenate([np.zeros(n_cols), np.ones(n_slack)])
        A_eq = np.concatenate([np.ones(n_cols), np.zeros(n_slack)])[None, :]
        res  = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=(0, None), method="highs-ds")
        w = res.x[:n_cols]
    else:
        raise ValueError(f"unknown loss: {loss}")
    return np.maximum(w, 0)

# not a quadrature for PG in the usual sense: the nodes and weights are fitted so
# that the pg identity reproduces the sigmoid itself, i.e. nonnegative weights with
#   1/2 e^{z/2} sum_j w_j e^{-omega_j z^2 / 2} ~ sigmoid(z)
# for z in [-30, 5], on a log spaced grid of candidate nodes (see fit_nonnegative
# for the losses). in pcpg the outer rule enters linearly, so the estimator
# error is this approximation error averaged over z, which makes this the
# quantity worth optimising. meant to be used with the sign flip
# sigmoid(z) = 1 - sigmoid(-z), which keeps z mostly <= 0. the 5 is a margin for
# the part of z that still ends up on the right. the grid starts at 0.02 so that
# e^{1/(8 omega)} in the hermite based estimators stays small (<= ~500).
# the sum to one constraint makes the estimate exactly 1/2 at z = 0
def _pg_sigmoid_fit(loss):
    z     = np.linspace(-30, 5, 1500)
    omega = np.logspace(np.log10(0.02), np.log10(8.0), 400)
    A = 0.5 * np.exp(z / 2)[:, None] * np.exp(-np.outer(z**2 / 2, omega))
    weights = fit_nonnegative(A, 1 / (1 + np.exp(-z)), loss)
    used = weights > 1e-14
    return omega[used], weights[used] / weights[used].sum()

# ── dispatch and reference ────────────────────────────────────────────────────

def pg_quadrature(n, method="golub_welsch"):
    if method == "trapezoid":
        return _pg_trapezoid(n)
    if method == "log_trapezoid":
        return _pg_log_trapezoid(n)
    if method == "gauss_legendre":
        return _pg_gauss_legendre(n)
    if method == "log_gauss_legendre":
        return _pg_log_gauss_legendre(n)
    if method == "golub_welsch":
        return _pg_golub_welsch(n)
    if method == "log_golub_welsch":
        return _pg_log_golub_welsch(n)
    if method.startswith("sigmoid_fit_"):
        return _pg_sigmoid_fit(method.removeprefix("sigmoid_fit_"))
    raise ValueError(f"unknown method: {method}")

# E[f(omega)] by adaptive gauss-kronrod (scipy quad, i.e. quadpack). not a fixed
# rule: the nodes depend on f and there are usually hundreds of evaluations, so
# it is only used for reference values. f takes a scalar omega. split at 1
# because [1, inf) is handled by quad with a variable transformation
def pg_reference(f, epsrel=1e-12):
    def integrand(o):
        return f(o) * float(pg_pdf(o))
    left,  _ = quad(integrand, 0, 1,      epsabs=0, epsrel=epsrel, limit=200)
    right, _ = quad(integrand, 1, np.inf, epsabs=0, epsrel=epsrel, limit=200)
    return left + right

# ═══ inner: N(0, 1) ═══════════════════════════════════════════════════════════

# quasi monte carlo: instead of random uniforms, the halton sequence (in 1d the
# van der corput sequence), which fills [0, 1] evenly, pushed through the normal
# inverse cdf. equal weights like mc, but the error shrinks like ~1/n instead
# of 1/sqrt(n). scrambled with a fixed seed, so it is reproducible
def _hs_qmc(n):
    p = qmc.Halton(d=1, scramble=True, seed=0).random(n)[:, 0]
    return ndtri(p), np.full(n, 1.0 / n)

# trapezoid rule on the whole real line: equally spaced nodes with spacing h,
# weighted by the normal density. for smooth integrands that decay like a
# gaussian this converges exponentially. two errors have to be balanced: the
# spacing (~ e^{-2 pi^2 / h^2}) and cutting off the range at +- n h / 2
# (~ e^{-(n h / 2)^2 / 2}). they are equal for h = sqrt(4 pi / n)
def _hs_trapezoid(n):
    h = np.sqrt(4 * np.pi / n)
    nodes = (np.arange(n) - (n - 1) / 2) * h
    weights = np.exp(-nodes**2 / 2)
    return nodes, weights / weights.sum()

# gauss-legendre on [-L, L]: the standard nodes on [-1, 1] scaled by L, the
# normal density goes into the weights. the generic textbook rule, it knows
# nothing about the normal distribution (like gauss-legendre for pg). L = 6
# cuts off a mass of ~2e-9
_HS_L = 6.0

def _hs_gauss_legendre(n):
    x, w    = roots_legendre(n)
    nodes   = _HS_L * x
    weights = w * np.exp(-nodes**2 / 2)
    return nodes, weights / weights.sum()

# gauss-hermite: exact for polynomials in u up to degree 2n - 1. the standard
# nodes are for the weight e^{-x^2}, u = sqrt(2) x turns them into nodes for
# N(0, 1), and dividing by sqrt(pi) normalises the weights
def _hs_gauss_hermite(n):
    x, w = roots_hermite(n)
    return np.sqrt(2) * x, w / np.sqrt(np.pi)

# ── dispatch ──────────────────────────────────────────────────────────────────

def hs_quadrature(n, method="gauss_hermite"):
    if method == "qmc":
        return _hs_qmc(n)
    if method == "trapezoid":
        return _hs_trapezoid(n)
    if method == "gauss_legendre":
        return _hs_gauss_legendre(n)
    if method == "gauss_hermite":
        return _hs_gauss_hermite(n)
    raise ValueError(f"unknown method: {method}")
