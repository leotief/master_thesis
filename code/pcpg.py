# pcpg: E[sigmoid(w^T x) | x_obs] through the cf of p(x_mis | x_obs), see the
# methods chapter of the thesis. with z = w^T x and c = 1/(2 gamma):
#   sigmoid(z)                   = 1/2 E_gamma[exp(z/2 - gamma z^2/2)],      gamma ~ PG(1, 0)
#   exp(z/2 - gamma z^2/2)       = exp(1/(8 gamma)) E_u[exp(-i u (z - c))],  u ~ N(0, gamma)
#   E[exp(-i u w_mis^T x_mis)]   = phi(-u w_mis)
# the integral over gamma (outer) and over u (inner) are each done by
# quadrature or mc:
#   PCPG(model)                          gauss-pg over gamma, gauss-hermite over u
#   PCPG(model, outer="mc")              mc over gamma, gauss-hermite over u
#   PCPG(model, inner="mc")              gauss-pg over gamma, mc over u
#   PCPG(model, outer="mc", inner="mc")  mc over both
#   PCPG(model, inner="exact")           gauss-pg over gamma, the inner integral
#                                        in closed form (needs z_mixture)

from functools import lru_cache

import numpy as np
import torch

from side.utils.pg_density import sample_pg
from side.utils.quadrature import pg_quadrature

# fn on the points of x_partial in chunks, so that the largest tensor (per_point
# elements per point) stays below max_elements. a float for one point (D,), an
# array (B,) for a batch (B, D)
def in_chunks(fn, x_partial, per_point, max_elements=2**25):
    if x_partial.dim() == 1:
        return float(fn(x_partial))
    size = max(1, max_elements // per_point)
    return torch.cat([fn(part) for part in x_partial.split(size)]).cpu().numpy()

class PCPG:
    # model needs characteristic_function and z_mean (z_mixture for inner="exact").
    # n_outer, n_inner: nodes (quad) or samples (mc) per integral, the defaults
    # of estimate and estimate_diff. pg_method: the outer rule from
    # side/utils/quadrature.py. flip: sigma(z) = 1 - sigma(-z) when z is mostly
    # positive. create it once and reuse it, the rng is seeded here
    def __init__(self, model, outer="quad", inner="quad", n_outer=19, n_inner=20, seed=42,
                 pg_method="golub_welsch", flip=True):
        assert outer in ("quad", "mc") and inner in ("quad", "mc", "exact")
        self.model = model
        self.outer, self.inner = outer, inner
        self.pg_method = pg_method
        self.n_outer, self.n_inner = n_outer, n_inner
        self.flip = flip
        self._rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=model.device).manual_seed(seed)

        # the nodes of the default counts, computed once
        if outer == "quad":
            self._outer_nodes(n_outer)
        if inner == "quad":
            self._inner_nodes(n_inner)

    def _to_tensor(self, a):
        return torch.tensor(a, dtype=torch.float32, device=self.model.device)

    # nodes and weights for gamma ~ PG(1, 0)
    @lru_cache(maxsize=None)
    def _outer_nodes(self, n):
        return tuple(map(self._to_tensor, pg_quadrature(n, self.pg_method)))

    # gauss-hermite nodes and weights, for the weight exp(-x^2)
    @lru_cache(maxsize=None)
    def _inner_nodes(self, n):
        return tuple(map(self._to_tensor, np.polynomial.hermite.hermgauss(n)))

    # gamma (..., n_outer, 1) and u (..., n_outer, n_inner) with their weights,
    # None for mc. random values are drawn per point (batch = leading dims)
    def _points(self, n_outer, n_inner, batch=()):
        if self.outer == "mc":
            gamma = self._to_tensor(sample_pg(int(np.prod(batch)) * n_outer, self._rng)).view(*batch, n_outer)
            pg_weights = None
        else:
            gamma, pg_weights = self._outer_nodes(n_outer)
        gamma = gamma[..., None]

        if self.inner == "mc": # u ~ N(0, gamma)
            shape = (*batch, n_outer, n_inner)
            eps = torch.randn(shape, device=self.model.device, generator=self._gen)
            u, h_weights = eps * gamma.sqrt(), None
        elif self.inner == "exact":
            u, h_weights = None, None
        else: # gauss-hermite nodes scaled to N(0, gamma)
            nodes, h_weights = self._inner_nodes(n_inner)
            u = (2 * gamma).sqrt() * nodes
        return u, gamma, h_weights, pg_weights

    # for evaluation: no gradients, in chunks. a float for one point (D,), an
    # array (B,) for a batch (B, D)
    @torch.no_grad()
    def estimate(self, w, x_partial, n_outer=None, n_inner=None):
        n_outer, n_inner = n_outer or self.n_outer, n_inner or self.n_inner
        return in_chunks(lambda x: self.estimate_diff(w, x, n_outer, n_inner), x_partial,
                         per_point=n_outer * n_inner * len(w))

    # for training: a tensor () or (B,) with gradients through w, no chunks
    def estimate_diff(self, w, x_partial, n_outer=None, n_inner=None):
        n_outer, n_inner = n_outer or self.n_outer, n_inner or self.n_inner
        u, gamma, h_weights, pg_weights = self._points(n_outer, n_inner, x_partial.shape[:-1])

        # sign flip: where z is mostly positive, compute E[sigma(-z)] instead and mirror at the end
        flip = self.model.z_mean(w, x_partial) > 0
        if not self.flip:
            flip = torch.zeros_like(flip)

        if self.inner == "exact":
            # E[exp(z/2 - gamma z^2/2)] per gaussian component N(m_k, v_k) of z,
            # the flip only changes the sign of m_k
            mix, m, v = self.model.z_mixture(w, x_partial)
            m = torch.where(flip.unsqueeze(-1), -m, m)
            mix, m, v = mix.unsqueeze(-2), m.unsqueeze(-2), v.unsqueeze(-2)          # (..., 1, K)
            per_component = (1 + gamma * v) ** -0.5 * torch.exp((v / 4 + m - gamma * m**2) / (2 * (1 + gamma * v)))
            inner = (mix * per_component).sum(dim=-1)                                # (..., n_pg)
        else:
            w = torch.where(flip.unsqueeze(-1), -w, w)

            # exp(1/(8 gamma)) exp(i u (c - w_obs^T x_obs)), the part without x_mis
            mis_mask = torch.isnan(x_partial)
            wo_xo = (w * torch.nan_to_num(x_partial)).sum(dim=-1)[..., None, None]
            inv_gamma = 1.0 / gamma
            im = u * (0.5 * inv_gamma - wo_xo)                 # (..., n_pg, n_u)
            real = (inv_gamma / 8).expand_as(im)
            factor = torch.exp(torch.complex(real, im))

            # the cf at s = -u w_mis, for all (gamma, u) pairs at once
            s = -u.flatten(start_dim=-2)[..., None] * (w * mis_mask).unsqueeze(-2)   # (..., n_pg * n_u, D)
            cf = self.model.characteristic_function(s, x_obs=x_partial)
            cf = cf.view_as(factor)

            integrand = factor * cf

            # inner integral over u
            if h_weights is None:
                inner = integrand.mean(dim=-1)
            else:
                inner = (integrand * h_weights).sum(dim=-1) / np.sqrt(np.pi)

        # outer integral over gamma
        if pg_weights is None:
            result = inner.mean(dim=-1)
        else:
            result = (inner * pg_weights).sum(dim=-1)

        # the result is real, the imaginary part is noise
        result = 0.5 * result.real
        return torch.where(flip, 1 - result, result)