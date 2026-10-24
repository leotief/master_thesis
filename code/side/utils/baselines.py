# simple estimators of E[sigmoid(w^T x) | x_obs] to compare the pcpg ones against
#
# MCBaseline is the obvious approach: sample the missing features and average.
# it is unbiased, so with enough samples it also serves as the reference value
# (demo.py uses 10M samples for that). MeanImputation is the naive approach that
# just plugs in one value and ignores the uncertainty in the missing features.
# MixtureGaussHermite and MixtureAdaptiveQuad are the direct route when z = w^T x
# given x_obs is a mixture of gaussians: a 1d integral per component, with
# fixed or adaptive quadrature. they only need model.z_moments, which the gmm
# gives directly (K components) and the pc by listing all its induced trees
# (exponentially many, so only for small circuits).

from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit

from side.utils.batching import in_chunks

# every estimator takes one point x_partial (D,) and returns a float, or a
# batch (B, D) and returns an array (B,), see batching.py

# plain monte carlo over the missing features
# the variance per sample is at most 0.25 since sigmoid is in [0, 1],
# so the error shrinks like 1/sqrt(n_samples)
class MCBaseline:
    def __init__(self, model, n_samples=1000):
        self.model = model   # needs model.sample(n, x_obs)
        self.n_samples = n_samples

    # n_samples defaults to the one of the constructor
    @torch.no_grad()
    def __call__(self, w, x_partial, n_samples=None):
        n_samples = n_samples or self.n_samples
        return in_chunks(lambda x: self._estimate(w, x, n_samples), x_partial, per_point=n_samples * len(w))

    # the same as a tensor with gradients through w, for training (train.py).
    # the samples don't depend on w, so the gradient is the one of the average
    def differentiable(self, w, x_partial, n_samples=None):
        return self._estimate(w, x_partial, n_samples or self.n_samples)

    def _estimate(self, w, x_partial, n_samples):
        # full vectors with x_mis ~ p(x_mis | x_obs) and the observed part copied in
        x_samples = self.model.sample(n_samples, x_obs=x_partial)
        # average prediction over all samples
        return torch.sigmoid(x_samples @ w).mean(dim=-1)


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
        return in_chunks(lambda x: torch.sigmoid(x @ w), x, per_point=len(w))


# the direct way for a mixture. within component k, z = w^T x is gaussian,
# z | k ~ N(m_k, v_k), so the expectation is a 1d integral per component,
# done with gauss-hermite:
#   E[sigmoid(z) | x_obs] ~ sum_k p(k | x_obs) 1/sqrt(pi) sum_i h_i sigmoid(m_k + sqrt(2 v_k) t_i)
# in the precision of the model (float32 for the gmm, float64 for the pc)
class MixtureGaussHermite:
    def __init__(self, model, n_nodes=50):
        self.model = model   # needs model.z_moments(w, x_obs)
        self.n_nodes = n_nodes

    # nodes and weights for n nodes, cached
    @lru_cache(maxsize=None)
    def _nodes(self, n):
        return tuple(torch.tensor(a, device=self.model.device) for a in np.polynomial.hermite.hermgauss(n))

    # n_nodes defaults to the one of the constructor
    @torch.no_grad()
    def __call__(self, w, x_partial, n_nodes=None):
        nodes, weights = self._nodes(n_nodes or self.n_nodes)
        return in_chunks(lambda x: self._estimate(w, x, nodes, weights), x_partial, per_point=len(w))

    # the same as a tensor with gradients through w, for training (train.py)
    def differentiable(self, w, x_partial, n_nodes=None):
        return self._estimate(w, x_partial, *self._nodes(n_nodes or self.n_nodes))

    def _estimate(self, w, x_partial, nodes, weights):
        mix, m, v = self.model.z_moments(w, x_partial)
        nodes, weights = nodes.to(m.dtype), weights.to(m.dtype)
        # the floor only matters for gradients: the root has an infinite slope
        # at v = 0 (at w = 0, or with nothing missing)
        z = m[..., None] + torch.sqrt((2 * v).clamp_min(1e-30))[..., None] * nodes     # (..., K, n_nodes)
        return (mix * (torch.sigmoid(z) @ weights)).sum(dim=-1) / np.sqrt(np.pi)


# same 1d integral per component, but with adaptive quadrature (scipy quad):
# it keeps splitting the interval where the integrand changes quickly until the
# requested accuracy is reached, so it is exact for any spread v_k. integrated
# over the standardised t with z = m_k + sqrt(v_k) t. points tells quad where
# the sigmoid switches from 0 to 1, which matters when z is very spread out
class MixtureAdaptiveQuad:
    def __init__(self, model, epsrel=1e-12):
        self.model = model   # needs model.z_moments(w, x_obs)
        self.epsrel = epsrel

    # a batch is done point by point, this is only the reference
    @torch.no_grad()
    def __call__(self, w, x_partial):
        if x_partial.dim() == 2:
            return np.array([self(w, x) for x in x_partial])
        mix, m, v = (t.double().cpu().numpy() for t in self.model.z_moments(w, x_partial))
        per_component = []
        for m_k, s_k in zip(m, np.sqrt(v)):
            integrand = lambda t: expit(m_k + s_k * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
            step = np.clip(-m_k / s_k, -10, 10)
            per_component.append(quad(integrand, -10, 10, points=[step], epsabs=0, epsrel=self.epsrel, limit=200)[0])
        return float(mix @ np.array(per_component))
