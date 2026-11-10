# the baselines: estimators (the prediction part, estimate.py) and models (training plus
# an estimator to predict with, train.py)

from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit

from experiment_utils import logistic_regression
from pclr import cfi, chunks, join

# ══ estimators: the prediction part (estimate.py) ════════════════════════════════
#
# like cfi, method(density, w, x_partial, **settings), w and the density given

# sample the missing features and average the predictions. the samples don't depend on
# w, so the gradient is the one of the average. density needs sample
def monte_carlo(density, w, x_partial, n_samples=1000):
    results = []
    for part in chunks(x_partial, per_point=n_samples * len(w)):
        x_samples = density.sample(n_samples, x_obs=part)
        results.append(torch.sigmoid(x_samples @ w).mean(dim=-1))
    return join(results, x_partial)


# fill the gaps with fixed values (e.g. the training means) and predict once
def mean_imputation(means, w, x_partial):
    x = torch.where(torch.isnan(x_partial), means.to(x_partial), x_partial)
    return torch.sigmoid(x @ w)


# fill the gaps with the density's conditional mean and predict once
def conditional_mean(density, w, x_partial):
    return torch.sigmoid(density.conditional_mean(x_partial) @ w)


# gauss-hermite nodes, cached per count and device
@lru_cache(maxsize=None)
def hermite_nodes(n, device):
    return tuple(torch.tensor(a, device=device) for a in np.polynomial.hermite.hermgauss(n))

# the integral of every component by gauss-hermite. density needs z_mixture
def mixture_gauss_hermite(density, w, x_partial, n_nodes=50):
    nodes, weights = hermite_nodes(n_nodes, density.device)
    results = []
    for part in chunks(x_partial, per_point=n_nodes * len(w)):
        mix, m, v = density.z_mixture(w, part)
        nodes, weights = nodes.to(m.dtype), weights.to(m.dtype)
        # the floor only matters for the gradient, which is infinite at zero variance
        z = m[..., None] + torch.sqrt((2 * v).clamp_min(1e-30))[..., None] * nodes
        results.append((mix * (torch.sigmoid(z) @ weights)).sum(dim=-1) / np.sqrt(np.pi))
    return join(results, x_partial)


# the integral of every component by scipy's adaptive quadrature, the reference: accurate
# for any spread, but point by point and without gradients. density needs z_mixture
@torch.no_grad()
def mixture_adaptive_quad(density, w, x_partial, epsrel=1e-12):
    if x_partial.dim() == 2:
        return torch.stack([mixture_adaptive_quad(density, w, x, epsrel) for x in x_partial])
    mix, m, v = (t.double().cpu().numpy() for t in density.z_mixture(w, x_partial))
    per_component = []
    for m_k, s_k in zip(m, np.sqrt(v)):
        integrand = lambda t: expit(m_k + s_k * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
        # where the sigmoid switches from 0 to 1, so quad doesn't miss it for a wide spread
        step = np.clip(-m_k / s_k, -10, 10)
        per_component.append(quad(integrand, -10, 10, points=[step], epsabs=0, epsrel=epsrel, limit=200)[0])
    return torch.tensor(float(mix @ np.array(per_component)), dtype=torch.float64)


# ══ models: the whole classifier (train.py) ══════════════════════════════════════
#
# like PCLR, fit(x_train, y_train), w and predict(x_test) with one of the estimators

# lr on the complete training points, the upper bound
class CompleteData:
    def __init__(self, px, n_freq):
        self.px, self.n_freq = px, n_freq

    def fit(self, x_train, y_train):
        self.w = logistic_regression(x_train, y_train)
        return self

    def predict(self, x_test):
        return cfi(self.px, self.w, x_test, self.n_freq)

# lr on the gaps filled by the training means
class MeanImputation:
    def fit(self, x_train, y_train):
        self.means = x_train.nanmean(dim=0)
        self.w = logistic_regression(torch.where(torch.isnan(x_train), self.means, x_train), y_train)
        return self

    def predict(self, x_test):
        return mean_imputation(self.means, self.w, x_test)

# lr on the gaps filled by the density's conditional mean
class ConditionalMean:
    def __init__(self, px):
        self.px = px

    def fit(self, x_train, y_train):
        with torch.no_grad():
            self.w = logistic_regression(self.px.conditional_mean(x_train), y_train)
        return self

    def predict(self, x_test):
        return conditional_mean(self.px, self.w, x_test)
