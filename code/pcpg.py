# pcpg estimators for E[sigmoid(w^T x) | x_obs]
#
# the problem: we want the expected prediction when some features are missing,
# averaged over p(x_mis | x_obs). the sigmoid makes this integral intractable,
# so instead of sampling x_mis directly (mc baseline) we rewrite the sigmoid in
# two steps until x_mis only shows up inside something we know in closed form.
#
# with z = w^T x:
#   1. polya-gamma identity
#        sigmoid(z) = 1/2 exp(z/2) E_omega[exp(-omega z^2 / 2)],  omega ~ PG(1, 0)
#   2. hubbard-stratonovich, turns the gaussian in z into something linear in z
#        exp(-omega z^2 / 2) = E_u[exp(-i u z)],  u ~ N(0, omega)
#   3. now z only appears in an exponent linear in x, and taking the expectation
#      over x_mis gives the characteristic function of p(x_mis | x_obs)
#        E[exp(-i u w_mis^T x_mis) | x_obs] = phi(-u w_mis)
#      which the gmm gives us exactly.
#
# so what remains is a double integral over omega and u. the classes below
# only differ in how those two integrals are approximated:
#   PCPGMC          mc over omega, mc over u
#   PCPGMCHermite   mc over omega, gauss-hermite over u
#   PCPGDeterministic  quadrature over both, fully deterministic
# in the code gamma is the same thing as omega. the quadrature rules for omega
# live in helpers/pg_quadrature.py.

from polyagamma import random_polyagamma
import numpy as np
import torch

from side.utils.pg_quadrature import pg_quadrature

# shared code. subclasses only decide which omega (gamma) and u values to use
# and with which weights, then call _expectation which does the actual math
class _PCPGBase:
    # model needs characteristic_function(t, x_obs) and z_mean(w, x_obs). the rng
    # is created here, so create the estimator once and reuse it, otherwise every
    # call gets the same pg samples (same seed) and the variance looks like 0.
    # flip: use sigma(z) = 1 - sigma(-z) when z = w^T x is mostly positive, see
    # _expectation_diff
    def __init__(self, model, seed=42, flip=True):
        self.model = model
        self._rng = np.random.default_rng(seed)
        self.flip = flip

    # omega ~ PG(1, 0) as a float32 tensor on the model device
    # sampling is done with the polyagamma package (numpy), then moved to torch
    def sample_pg(self, n_samples):
        omega = random_polyagamma(1.0, np.zeros(n_samples), random_state=self._rng)
        return torch.tensor(omega, dtype=torch.float32, device=self.model.device)

    def _expectation_diff(self, w, x_partial, u, gamma, h_weights=None, pg_weights=None):
        """Compute E[σ(wᵀx) | x_obs] as a scalar tensor; gradients flow through w."""
        # shapes: gamma is (n_pg, 1), u is (n_pg, n_u), one row of u values per
        # omega. h_weights / pg_weights are quadrature weights, None means plain mc
        #
        # sigma(z) = 1 - sigma(-z) also holds for the whole expectation. for z > 0
        # the factor e^{z/2} in the pg identity amplifies every approximation
        # error, for z < 0 it shrinks them. so if z is mostly positive, compute
        # E[sigma(-z)] with -w instead and mirror the result at the end
        flip = self.flip and bool(self.model.z_mean(w, x_partial) > 0)
        if flip:
            w = -w

        # split z = w_obs^T x_obs + w_mis^T x_mis. the observed part is just a
        # number and can be pulled out, only the missing part needs the cf
        obs_mask = ~torch.isnan(x_partial)
        wo, wm = w[obs_mask], w[~obs_mask]
        wo_xo = wo @ x_partial[obs_mask]
        inv_gamma = 1.0 / gamma

        # the exp(z/2) in front of the pg identity is not gaussian in z, so it is
        # absorbed by completing the square before applying hubbard-stratonovich:
        #   exp(z/2 - gamma z^2/2) = exp(1/(8 gamma)) exp(-gamma (z - 1/(2 gamma))^2 / 2)
        # the constant exp(1/(8 gamma)) becomes the real part below, and the shift
        # 1/(2 gamma) ends up in the phase together with the observed part.
        # this factor is huge for small gamma, which is why the inner mc variance
        # is infinite in theory and pcpg-mc needs enough u samples per omega
        real = (inv_gamma / 8).expand_as(u)
        im = u * (0.5 * inv_gamma - wo_xo)
        factor = torch.exp(torch.complex(real, im))

        # expectation over x_mis: the cf of p(x_mis | x_obs) at s = -u w_mis.
        # all (omega, u) pairs are flattened into one batch of n_pg * n_u
        # frequency vectors and reshaped back afterwards
        s = torch.outer(-u.flatten(), wm)
        cf = self.model.characteristic_function(s, x_obs=x_partial)
        cf = cf.view_as(factor)

        integrand = factor * cf
        # inner integral over u: plain mean for mc, weighted sum for gauss-hermite.
        # the 1/sqrt(pi) is the normalisation of the hermite weight exp(-x^2)
        if h_weights is None:
            inner = integrand.mean(dim=-1)
        else:
            inner = (integrand * h_weights).sum(dim=-1) / np.sqrt(np.pi)

        # outer integral over omega, same idea (pg weights already sum to one)
        if pg_weights is None:
            result = inner.mean()
        else:
            result = (inner * pg_weights).sum()
        # the exact result is real, the imaginary part of the estimate is only
        # noise and is dropped. the 1/2 is from the pg identity
        result = 0.5 * result.real
        return 1 - result if flip else result

    # same as above without gradients, returns a python float. used for evaluation
    @torch.no_grad()
    def _expectation(self, w, x_partial, u, gamma, h_weights=None, pg_weights=None):
        return float(self._expectation_diff(w, x_partial, u, gamma, h_weights, pg_weights))


class PCPGMC(_PCPGBase):
    """Both integrals (PG outer, Gaussian inner) estimated by Monte Carlo."""

    # total cost is n_pg * n_gauss cf evaluations. see demo.py for how to split
    # a fixed budget between the two (n_gauss around 4-5 times n_pg works well)
    def __call__(self, w, x_partial, n_pg=100, n_gauss=100):
        gamma = self.sample_pg(n_pg)[:, None]
        # n_gauss values of u for every omega, u ~ N(0, gamma)
        u = torch.randn(n_pg, n_gauss, device=self.model.device) * gamma.sqrt()
        return self._expectation(w, x_partial, u, gamma)


class PCPGMCHermite(_PCPGBase):
    """PG outer integral by MC; Gaussian inner integral by Gauss-Hermite quadrature."""

    # the inner integral is over a gaussian, which is exactly what gauss-hermite
    # is made for, so a few nodes already remove almost all of the inner variance.
    # nodes and weights don't depend on omega, so they are computed once here
    def __init__(self, model, n_gauss_quad=30, seed=42, flip=True):
        super().__init__(model, seed=seed, flip=flip)
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_gauss_quad)
        self.hermite_nodes   = torch.tensor(nodes_np,   dtype=torch.float32, device=model.device)
        self.hermite_weights = torch.tensor(weights_np, dtype=torch.float32, device=model.device)

    # cost is n_pg * n_gauss_quad cf evaluations, the only randomness left is omega
    def __call__(self, w, x_partial, n_pg=100):
        gamma = self.sample_pg(n_pg)[:, None]
        # hermite nodes are for the weight exp(-x^2), so u = sqrt(2 gamma) x
        # turns them into nodes for N(0, gamma)
        u = (2 * gamma).sqrt() * self.hermite_nodes
        return self._expectation(w, x_partial, u, gamma, h_weights=self.hermite_weights)

class PCPGDeterministic(_PCPGBase):
    # no sampling at all, so zero variance. the error is only the quadrature
    # error, controlled by n_pg_quad and n_gauss_quad.
    # cost is n_pg_quad * n_gauss_quad cf evaluations per call.
    # pg_method picks the outer rule from helpers/pg_quadrature.py ("golub_welsch" only
    # works up to n_pg_quad ~ 19, "sigmoid_fit_*" ignore n_pg_quad)

    def __init__(self, model, n_pg_quad=10, n_gauss_quad=30, seed=42, pg_method="golub_welsch", flip=True):
        super().__init__(model, seed=seed, flip=flip)
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_gauss_quad)
        self.hermite_nodes   = torch.tensor(nodes_np,   dtype=torch.float32, device=model.device)
        self.hermite_weights = torch.tensor(weights_np, dtype=torch.float32, device=model.device)

        # golub-welsch takes a few seconds, so only done once here
        pg_nodes_np, pg_weights_np = pg_quadrature(n_pg_quad, pg_method)
        self.pg_nodes   = torch.tensor(pg_nodes_np,   dtype=torch.float32, device=model.device)
        self.pg_weights = torch.tensor(pg_weights_np, dtype=torch.float32, device=model.device)

    def __call__(self, w, x_partial):
        # all nodes are fixed, so this returns the same value every time
        gamma = self.pg_nodes[:, None]
        u = (2 * gamma).sqrt() * self.hermite_nodes
        return self._expectation(w, x_partial, u, gamma, h_weights=self.hermite_weights, pg_weights=self.pg_weights)