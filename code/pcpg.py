# pcpg estimators for E[sigmoid(w^T x) | x_obs]
#
# the problem: we want the expected prediction when some features are missing,
# averaged over p(x_mis | x_obs). the sigmoid makes this integral intractable,
# so instead of sampling x_mis directly (mc baseline) we rewrite the sigmoid in
# two steps until x_mis only shows up inside something we know in closed form.
#
# with z = w^T x:
#   1. polya-gamma identity
#        sigmoid(z) = 1/2 exp(z/2) E_gamma[exp(-gamma z^2 / 2)],  gamma ~ PG(1, 0)
#   2. hubbard-stratonovich, turns the gaussian in z into something linear in z
#        exp(-gamma z^2 / 2) = E_u[exp(-i u z)],  u ~ N(0, gamma)
#   3. now z only appears in an exponent linear in x, and taking the expectation
#      over x_mis gives the characteristic function of p(x_mis | x_obs)
#        E[exp(-i u w_mis^T x_mis) | x_obs] = phi(-u w_mis)
#      which the gmm gives us exactly.
#
# so what remains is a double integral over gamma (outer) and u (inner), each
# done by quadrature or by mc. the outer quadrature is a rule for PG(1, 0) from
# side/utils/quadrature.py (pg_method, golub-welsch by default), the inner
# one gauss-hermite:
#   PCPG(model)                          quadrature over both, deterministic
#   PCPG(model, outer="mc")              mc over gamma, gauss-hermite over u
#   PCPG(model, inner="mc")              golub-welsch over gamma, mc over u
#   PCPG(model, outer="mc", inner="mc")  mc over both

from functools import lru_cache

import numpy as np
import torch

from side.utils.batching import in_chunks
from side.utils.pg_density import sample_pg
from side.utils.quadrature import pg_quadrature

class PCPG:
    # model needs characteristic_function(t, x_obs) and z_mean(w, x_obs).
    # outer and inner are "quad" or "mc". n_outer and n_inner are the number of
    # nodes (quad) or samples (mc) of each integral, the defaults for __call__.
    # pg_method picks the outer quadrature rule from side/utils/quadrature.py
    # ("golub_welsch", "sigmoid_fit_l2", ...), the inner one is gauss-hermite.
    # cost is n_outer * n_inner cf evaluations per call.
    # the rng is created here, so create the estimator once and reuse it,
    # otherwise every call gets the same samples (same seed) and the variance
    # looks like 0.
    # golub-welsch only works up to n_outer ~ 19, "sigmoid_fit_*" ignore n_outer.
    # flip: use sigma(z) = 1 - sigma(-z) when z = w^T x is mostly positive, see
    # _expectation_diff
    def __init__(self, model, outer="quad", inner="quad", n_outer=19, n_inner=20, seed=42,
                 pg_method="golub_welsch", flip=True):
        assert outer in ("quad", "mc") and inner in ("quad", "mc")
        self.model = model
        self.outer, self.inner = outer, inner
        self.pg_method = pg_method
        self.n_outer, self.n_inner = n_outer, n_inner
        self.flip = flip
        self._rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=model.device).manual_seed(seed)

        # the nodes for the default n_outer and n_inner are already computed here and cached
        if outer == "quad":
            self._outer_nodes(n_outer)
        if inner == "quad":
            self._inner_nodes(n_inner)

    def _to_tensor(self, a):
        return torch.tensor(a, dtype=torch.float32, device=self.model.device)

    # quadrature nodes and weights of the outer integral, for gamma ~ PG(1, 0)
    @lru_cache(maxsize=None)
    def _outer_nodes(self, n):
        return tuple(map(self._to_tensor, pg_quadrature(n, self.pg_method)))

    # gauss-hermite nodes and weights of the inner integral, for the weight exp(-x^2)
    @lru_cache(maxsize=None)
    def _inner_nodes(self, n):
        return tuple(map(self._to_tensor, np.polynomial.hermite.hermgauss(n)))

    # the gamma values ((..., n_outer, 1)) and u values ((..., n_outer, n_inner))
    # with their weights, None means plain mc. batch are the leading dims of the
    # points: the random parts are drawn for every point on its own, the
    # quadrature nodes are the same for all
    def _points(self, n_outer, n_inner, batch=()):
        if self.outer == "mc":
            gamma = self._to_tensor(sample_pg(int(np.prod(batch)) * n_outer, self._rng)).view(*batch, n_outer)
            pg_weights = None
        else:
            gamma, pg_weights = self._outer_nodes(n_outer)
        gamma = gamma[..., None]

        if self.inner == "mc": # n_inner values of u for every gamma, u ~ N(0, gamma)
            shape = (*batch, n_outer, n_inner)
            eps = torch.randn(shape, device=self.model.device, generator=self._gen)
            u, h_weights = eps * gamma.sqrt(), None
        else: # u = sqrt(2 gamma) x turns the nodes for exp(-x^2) into nodes for N(0, gamma)
            nodes, h_weights = self._inner_nodes(n_inner)
            u = (2 * gamma).sqrt() * nodes
        return u, gamma, h_weights, pg_weights

    # E[sigma(w^T x) | x_obs] as a float for one point x_partial (D,), or as an
    # array (B,) for a batch (B, D). n_outer and n_inner default to the ones of
    # the constructor. with quadrature over both, it's deterministic
    def __call__(self, w, x_partial, n_outer=None, n_inner=None):
        n_outer, n_inner = n_outer or self.n_outer, n_inner or self.n_inner
        return in_chunks(lambda x: self._expectation(w, x, *self._points(n_outer, n_inner, x.shape[:-1])),
                         x_partial, per_point=n_outer * n_inner * len(w))

    # the same as a tensor with gradients through w (and the model), for
    # training on incomplete data (train.py). one call on the whole batch, no chunks
    def differentiable(self, w, x_partial, n_outer=None, n_inner=None):
        n_outer, n_inner = n_outer or self.n_outer, n_inner or self.n_inner
        return self._expectation_diff(w, x_partial, *self._points(n_outer, n_inner, x_partial.shape[:-1]))

    def _expectation_diff(self, w, x_partial, u, gamma, h_weights=None, pg_weights=None):
        """Compute E[σ(wᵀx) | x_obs], shape () or (B,); gradients flow through w."""
        # shapes: gamma is (..., n_pg, 1), u is (..., n_pg, n_u), one row of u
        # values per gamma. h_weights / pg_weights are quadrature weights, None
        # means plain mc. x_partial is one point (D,) or a batch (B, D),
        # everything below gets its leading dims (written ... in the shapes),
        # gamma and u have them when they are random
        #
        # sigma(z) = 1 - sigma(-z) also holds for the whole expectation. for z > 0
        # the factor e^{z/2} in the pg identity amplifies every approximation
        # error, for z < 0 it shrinks them. so if z is mostly positive, compute
        # E[sigma(-z)] with -w instead and mirror the result at the end. the
        # sign is decided per point, so w becomes one row per point
        flip = self.model.z_mean(w, x_partial) > 0
        if not self.flip:
            flip = torch.zeros_like(flip)
        w = torch.where(flip.unsqueeze(-1), -w, w)

        # split z = w_obs^T x_obs + w_mis^T x_mis. the observed part is just a
        # number and can be pulled out, only the missing part needs the cf
        mis_mask = torch.isnan(x_partial)
        wo_xo = (w * torch.nan_to_num(x_partial)).sum(dim=-1)[..., None, None]
        inv_gamma = 1.0 / gamma

        # the exp(z/2) in front of the pg identity is not gaussian in z, so it is
        # absorbed by completing the square before applying hubbard-stratonovich:
        #   exp(z/2 - gamma z^2/2) = exp(1/(8 gamma)) exp(-gamma (z - 1/(2 gamma))^2 / 2)
        # the constant exp(1/(8 gamma)) becomes the real part below, and the shift
        # 1/(2 gamma) ends up in the phase together with the observed part.
        # this factor is huge for small gamma, which is why the inner mc variance
        # is infinite in theory and pcpg-mc needs enough u samples per gamma
        im = u * (0.5 * inv_gamma - wo_xo)                 # (..., n_pg, n_u)
        real = (inv_gamma / 8).expand_as(im)
        factor = torch.exp(torch.complex(real, im))

        # expectation over x_mis: the cf of p(x_mis | x_obs) at s = -u w_mis.
        # all (gamma, u) pairs are flattened into one batch of n_pg * n_u
        # frequency vectors per point and reshaped back afterwards
        s = -u.flatten(start_dim=-2)[..., None] * (w * mis_mask).unsqueeze(-2)   # (..., n_pg * n_u, D)
        cf = self.model.characteristic_function(s, x_obs=x_partial)
        cf = cf.view_as(factor)

        integrand = factor * cf
        # inner integral over u: plain mean for mc, weighted sum for gauss-hermite.
        # the 1/sqrt(pi) is the normalisation of the hermite weight exp(-x^2)
        if h_weights is None:
            inner = integrand.mean(dim=-1)
        else:
            inner = (integrand * h_weights).sum(dim=-1) / np.sqrt(np.pi)

        # outer integral over gamma, same idea (pg weights already sum to one)
        if pg_weights is None:
            result = inner.mean(dim=-1)
        else:
            result = (inner * pg_weights).sum(dim=-1)
        # the exact result is real, the imaginary part of the estimate is only
        # noise and is dropped. the 1/2 is from the pg identity
        result = 0.5 * result.real
        return torch.where(flip, 1 - result, result)

    # same as above without gradients. used for evaluation
    @torch.no_grad()
    def _expectation(self, w, x_partial, u, gamma, h_weights=None, pg_weights=None):
        return self._expectation_diff(w, x_partial, u, gamma, h_weights, pg_weights)
