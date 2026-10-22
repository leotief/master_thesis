# simple estimators of E[sigmoid(w^T x) | x_obs] to compare the pcpg ones against
#
# MCBaseline is the obvious approach: sample the missing features and average.
# it is unbiased, so with enough samples it also serves as the reference value
# (demo.py uses 10M samples for that). MeanImputation is the naive approach that
# just plugs in one value and ignores the uncertainty in the missing features.
# GMMGaussHermite and GMMAdaptiveQuad are the direct route for a gmm: a 1d
# integral per component, with fixed or adaptive quadrature.

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit

# plain monte carlo over the missing features
# the variance per sample is at most 0.25 since sigmoid is in [0, 1],
# so the error shrinks like 1/sqrt(n_samples)
class MCBaseline:
    def __init__(self, model):
        self.model = model   # needs model.sample(n, x_obs)

    @torch.no_grad()
    def __call__(self, w, x_partial, n_samples=1000):
        # full vectors with x_mis ~ p(x_mis | x_obs) and the observed part copied in
        x_samples = self.model.sample(n_samples, x_obs=x_partial)
        # average prediction over all samples
        return float(torch.sigmoid(x_samples @ w).mean())


# fills the gaps with a fixed vector (e.g. the training mean) and predicts once.
# this gives sigmoid(w^T x_filled) instead of E[sigmoid(w^T x)], so it tends to
# be too confident when the missing part is uncertain
class MeanImputation:
    def __init__(self, mean):
        self.mean = mean   # one value per feature

    @torch.no_grad()
    def __call__(self, w, x_partial):
        obs_mask = ~torch.isnan(x_partial)
        # keep observed values, use the mean everywhere else
        x = torch.where(obs_mask, x_partial, self.mean.to(x_partial))
        return float(torch.sigmoid((w * x).sum()))


# the direct way for a gmm. within component k, z = w^T x is gaussian,
# z | k ~ N(m_k, v_k), so the expectation is a 1d integral per component,
# done with gauss-hermite:
#   E[sigmoid(z) | x_obs] ~ sum_k p(k | x_obs) 1/sqrt(pi) sum_i h_i sigmoid(m_k + sqrt(2 v_k) t_i)
class GMMGaussHermite:
    def __init__(self, model, n_nodes=50):
        self.model = model   # needs model.z_moments(w, x_obs)
        self.nodes, self.weights = np.polynomial.hermite.hermgauss(n_nodes)

    @torch.no_grad()
    def __call__(self, w, x_partial):
        mix, m, v = (t.double().cpu().numpy() for t in self.model.z_moments(w, x_partial))
        z = m[:, None] + np.sqrt(2 * v)[:, None] * self.nodes      # (K, n_nodes)
        return float(mix @ (1 / (1 + np.exp(-z)) @ self.weights) / np.sqrt(np.pi))


# same 1d integral per component, but with adaptive quadrature (scipy quad):
# it keeps splitting the interval where the integrand changes quickly until the
# requested accuracy is reached, so it is exact for any spread v_k. integrated
# over the standardised t with z = m_k + sqrt(v_k) t. points tells quad where
# the sigmoid switches from 0 to 1, which matters when z is very spread out
class GMMAdaptiveQuad:
    def __init__(self, model, epsrel=1e-12):
        self.model = model   # needs model.z_moments(w, x_obs)
        self.epsrel = epsrel

    @torch.no_grad()
    def __call__(self, w, x_partial):
        mix, m, v = (t.double().cpu().numpy() for t in self.model.z_moments(w, x_partial))
        per_component = []
        for m_k, s_k in zip(m, np.sqrt(v)):
            integrand = lambda t: expit(m_k + s_k * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
            step = np.clip(-m_k / s_k, -10, 10)
            per_component.append(quad(integrand, -10, 10, points=[step], epsabs=0, epsrel=self.epsrel, limit=200)[0])
        return float(mix @ np.array(per_component))
