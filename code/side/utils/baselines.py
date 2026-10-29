# simple estimators of E[sigmoid(w^T x) | x_obs] to compare the pcpg ones against
#
# MCBaseline is the obvious approach: sample the missing features and average.
# it is unbiased, so with enough samples it also serves as the reference value
# (demo.py uses 10M samples for that). MeanImputation is the naive approach that
# just plugs in one value and ignores the uncertainty in the missing features.
# MixtureGaussHermite and MixtureAdaptiveQuad are the direct route when z = w^T x
# given x_obs is a mixture of gaussians: a 1d integral per component, with
# fixed or adaptive quadrature. they only need model.z_mixture, which the gmm
# gives directly (K components) and the pc by listing all its induced trees
# (exponentially many, so only for small circuits).

from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit

# every estimator has the same two methods:
#   estimate(w, x_partial)       for evaluation, without gradients: one point
#                                x_partial (D,) gives a float, a batch (B, D)
#                                an array (B,)
#   estimate_diff(w, x_partial)  the same as a tensor with gradients through w,
#                                for training (train.py)
# MixtureAdaptiveQuad only has estimate, scipy's quad has no gradients.
# for estimate a batch is one call over all points, so the gpu does the work in
# a few large operations instead of many small ones. the intermediate tensors
# grow with the number of points, so the batch is split into chunks that keep
# them below max_elements

# fn on the points of x_partial, in chunks of points, each point costs
# per_point elements of the largest intermediate tensor. a float for one point,
# an array of shape (B,) for a batch
def in_chunks(fn, x_partial, per_point, max_elements=2**25):
    if x_partial.dim() == 1:
        return float(fn(x_partial))
    size = max(1, max_elements // per_point)
    return torch.cat([fn(part) for part in x_partial.split(size)]).cpu().numpy()

# plain monte carlo over the missing features
# the variance per sample is at most 0.25 since sigmoid is in [0, 1],
# so the error shrinks like 1/sqrt(n_samples)
class MCBaseline:
    def __init__(self, model, n_samples=1000):
        self.model = model   # needs model.sample(n, x_obs)
        self.n_samples = n_samples

    # n_samples defaults to the one of the constructor
    @torch.no_grad()
    def estimate(self, w, x_partial, n_samples=None):
        n_samples = n_samples or self.n_samples
        return in_chunks(lambda x: self.estimate_diff(w, x, n_samples), x_partial, per_point=n_samples * len(w))

    # the samples don't depend on w, so the gradient is the one of the average
    def estimate_diff(self, w, x_partial, n_samples=None):
        # full vectors with x_mis ~ p(x_mis | x_obs) and the observed part copied in
        x_samples = self.model.sample(n_samples or self.n_samples, x_obs=x_partial)
        # average prediction over all samples
        return torch.sigmoid(x_samples @ w).mean(dim=-1)


# fills the gaps with a fixed vector (e.g. the training mean) and predicts once.
# this gives sigmoid(w^T x_filled) instead of E[sigmoid(w^T x)], so it tends to
# be too confident when the missing part is uncertain
class MeanImputation:
    def __init__(self, mean):
        self.mean = mean   # one value per feature

    @torch.no_grad()
    def estimate(self, w, x_partial):
        return in_chunks(lambda x: self.estimate_diff(w, x), x_partial, per_point=len(w))

    def estimate_diff(self, w, x_partial):
        obs_mask = ~torch.isnan(x_partial)
        # keep observed values, use the mean everywhere else
        x = torch.where(obs_mask, x_partial, self.mean.to(x_partial))
        return torch.sigmoid(x @ w)


# the direct way for a mixture. within component k, z = w^T x is gaussian,
# z | k ~ N(m_k, v_k), so the expectation is a 1d integral per component,
# done with gauss-hermite:
#   E[sigmoid(z) | x_obs] ~ sum_k p(k | x_obs) 1/sqrt(pi) sum_i h_i sigmoid(m_k + sqrt(2 v_k) t_i)
# in the precision of the model (float32 for the gmm, float64 for the pc)
class MixtureGaussHermite:
    def __init__(self, model, n_nodes=50):
        self.model = model   # needs model.z_mixture(w, x_obs)
        self.n_nodes = n_nodes

    # nodes and weights for n nodes, cached
    @lru_cache(maxsize=None)
    def _nodes(self, n):
        return tuple(torch.tensor(a, device=self.model.device) for a in np.polynomial.hermite.hermgauss(n))

    # n_nodes defaults to the one of the constructor
    @torch.no_grad()
    def estimate(self, w, x_partial, n_nodes=None):
        return in_chunks(lambda x: self.estimate_diff(w, x, n_nodes), x_partial, per_point=len(w))

    def estimate_diff(self, w, x_partial, n_nodes=None):
        nodes, weights = self._nodes(n_nodes or self.n_nodes)
        mix, m, v = self.model.z_mixture(w, x_partial)
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
        self.model = model   # needs model.z_mixture(w, x_obs)
        self.epsrel = epsrel

    # a batch is done point by point, this is only the reference
    @torch.no_grad()
    def estimate(self, w, x_partial):
        if x_partial.dim() == 2:
            return np.array([self.estimate(w, x) for x in x_partial])
        mix, m, v = (t.double().cpu().numpy() for t in self.model.z_mixture(w, x_partial))
        per_component = []
        for m_k, s_k in zip(m, np.sqrt(v)):
            integrand = lambda t: expit(m_k + s_k * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
            step = np.clip(-m_k / s_k, -10, 10)
            per_component.append(quad(integrand, -10, 10, points=[step], epsabs=0, epsrel=self.epsrel, limit=200)[0])
        return float(mix @ np.array(per_component))
