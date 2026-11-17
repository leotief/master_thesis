# pclr: the joint model of a density (gmm or pc) and a logistic regression, trained
# jointly on incomplete data. the likelihood of a point splits into the density of its
# observed features and the expected prediction of its label, both differentiable.
#
# below the model is the method for the expected prediction, cfi. like the baselines in
# utils/baselines.py, it is a plain function method(density, w, x_partial, **settings),
# with nan at the missing features, and returns a tensor with gradients through w and the
# density

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLR(nn.Module):
    # n_freq are cfi's frequencies, lam the weight of the label term, C the l2 penalty (as
    # in sklearn), eps the clamp of the predictions in the loss, train_density whether the
    # density is trained together with w or kept as it is, max_iter the iterations of
    # l-bfgs, start where the trained density starts: an untrained one ("untrained", a gmm
    # of the same size) or the fitted density given to fit ("fitted"). with the density
    # fixed it is always the fitted one
    def __init__(self, n_freq=100, lam=1.0, C=1.0, eps=1e-6, train_density=True, max_iter=1000, start="untrained"):
        super().__init__()
        self.n_freq = n_freq
        self.lam, self.C, self.eps = lam, C, eps
        self.train_density, self.max_iter, self.start = train_density, max_iter, start

    # log-density of the observed features
    def log_density(self, x_partial):
        return self.density(x_partial)

    # expected prediction p(y=1 | x_obs), the missing features averaged out under the density
    def predict(self, x_partial):
        return cfi(self.density, self.w, x_partial, self.n_freq)

    # the two terms of the log-likelihood, each a mean over the points: the density of the
    # observed features, and the binary cross-entropy of the expected prediction
    def log_likelihood(self, x_partial, y):
        log_density = self.log_density(x_partial).mean()
        p = self.predict(x_partial).to(y.dtype).clamp(self.eps, 1 - self.eps)
        log_label = -F.binary_cross_entropy(p, y)
        return log_density, log_label

    # the training loss on a minibatch: negative log-likelihood plus the l2 penalty,
    # scaled to one of n_train points
    def objective(self, x_partial, y, n_train):
        log_density, log_label = self.log_likelihood(x_partial, y)
        return -(log_density + self.lam * log_label) + self.w.square().sum() / (2 * self.C * n_train)

    # the density and w jointly, or only w with the density kept as it is (train_density),
    # by l-bfgs on all training points at once until it converges, like sklearn for a
    # logistic regression. the objective is deterministic, so no minibatches are needed.
    # density is the fitted density: pclr trains a new untrained gmm of the same size, or its
    # own copy of the fitted one (start "fitted", or the density fixed). w starts at 0
    def fit(self, x_partial, y, density):
        if self.start == "untrained" and self.train_density:
            self.density = type(density)(x_partial, density.K, density.device)
        else:
            self.density = copy.deepcopy(density)
        for parameter in self.density.parameters():
            parameter.requires_grad_(True)
        self.w = nn.Parameter(torch.zeros(x_partial.shape[1], device=x_partial.device))

        parameters = list(self.parameters()) if self.train_density else [self.w]
        optimizer = torch.optim.LBFGS(parameters, max_iter=self.max_iter, tolerance_grad=1e-7, tolerance_change=1e-12,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = self.objective(x_partial, y, len(x_partial))
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


# ── cfi ───────────────────────────────────────────────────────────────────────
#
# characteristic-function inversion: the sigmoid written as a weighted sum of sine
# waves, so the expected prediction only needs the characteristic function of z = w^T x
# at a few frequencies. the integral over the frequencies is done by the trapezoid rule

# beyond this frequency the weights are below double precision
S_MAX = 12.0

# density needs char_func, z_mean and z_var
def cfi(density, w, x_partial, n_freq=200):
    results = []
    for part in chunks(x_partial, per_point=n_freq * len(w)):
        # the step per point: small enough that the periodic copies of the sigmoid in the
        # trapezoid sum don't overlap with where z lives, and large enough to reach S_MAX.
        # the value doesn't depend on it, so no gradient
        mean = density.z_mean(w, part)
        std = density.z_var(w, part).clamp(min=0).sqrt()
        h = torch.minimum(np.pi / (mean.abs() + 6 * std + 1), torch.full_like(mean, S_MAX / n_freq)).detach()
        s = h[..., None] * torch.arange(1, n_freq + 1, device=w.device, dtype=w.dtype)

        # characteristic function of z at all frequencies: the observed features only add
        # a phase, the missing ones come from the density
        mis_mask = torch.isnan(part)
        wo_xo = (w * torch.nan_to_num(part)).sum(dim=-1)[..., None]
        freqs = s[..., None] * (w * mis_mask).unsqueeze(-2)
        cf = torch.exp(torch.complex(torch.zeros_like(s), s * wo_xo)) * density.char_func(freqs, x_obs=part)

        # trapezoid rule, the term at frequency 0 has half the weight and is the mean of z
        integral = h * (mean / (2 * np.pi) + (cf.imag / torch.sinh(np.pi * s)).sum(dim=-1))
        results.append(0.5 + integral)
    return join(results, x_partial)

# ── batching ──────────────────────────────────────────────────────────────────
#
# the methods build tensors that grow with the number of points, so large batches are
# split into chunks that keep the largest tensor below MAX_ELEMENTS, and joined again
# (with their gradients). a single point is one chunk

MAX_ELEMENTS = 2**25

def chunks(x_partial, per_point):
    if x_partial.dim() == 1:
        return [x_partial]
    return x_partial.split(max(1, MAX_ELEMENTS // per_point))

def join(results, x_partial):
    if x_partial.dim() == 1:
        return results[0]
    return torch.cat(results)
