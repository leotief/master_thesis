# probabilistic circuit as the density p(x), a rat-spn from the spflow library
# with gaussian leaves (pc.py is the same kind of circuit written from scratch),
# with the same interface as the gmm (gmm.py), so that pcpg, the mc baseline
# and the exact baselines work with it unchanged:
#   sample(n, x_obs)                   mc, spflow's conditional sampling
#   characteristic_function(t, x_obs)  pcpg, the cf of p(x_mis | x_obs)
#   z_mean(w, x_obs)                   pcpg's sign flip
#   z_moments(w, x_obs)                the exact baselines, p(z | x_obs) as a
#                                      mixture of gaussians
#
# a rat-spn is a stack of layers: gaussian leaves -> Factorize (products of
# leaves into regions) -> (SplitConsecutive -> ElementwiseProduct -> Sum) per
# depth -> RepetitionMixingLayer. spflow computes log-likelihoods through these
# layers, the two passes below go through the same layers with other values:
#
#   cf pass       the log of the cf in complex numbers. a missing leaf gives
#                 the log cf of its gaussian, i s mu - s^2 sigma^2 / 2, an
#                 observed one its log density. products add, sums are a
#                 logsumexp. divided by p(x_obs) (minus in log space) this is
#                 the cf of p(x_mis | x_obs). one pass per frequency, so it
#                 scales with the size of the circuit
#   mixture pass  the distribution of z = w^T x. given the observed features it
#                 is a mixture of gaussians, one component per induced tree of
#                 the circuit: every node holds the mixture of its part of z,
#                 products combine the components of their children pairwise,
#                 sums put them together. the number of components grows
#                 exponentially with the depth, so this only works for small
#                 circuits, as the exact reference
#
# both passes run in float64, the data and parameters are float32

import torch
import torch.nn as nn
from spflow.meta.data.scope import Scope
from spflow.modules.leaves import Normal
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.rat.factorize import Factorize
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.modules.sums.sum import Sum
from spflow.zoo.rat.rat_spn import RatSPN

class SPFlowPC(nn.Module):
    # channels: gaussians per feature and sum nodes per region, repetitions:
    # independent random structures mixed at the root, depth: number of
    # sum/product layers. the mixture pass has repetitions * channels^(2^depth - 1)
    # components
    def __init__(self, device, n_features, channels=4, repetitions=3, depth=2):
        super().__init__()
        self.device = device
        self.n_features = n_features
        leaf = Normal(scope=Scope(list(range(n_features))), out_channels=channels, num_repetitions=repetitions)
        self.spn = RatSPN(leaf_modules=[leaf], n_root_nodes=1, n_region_nodes=channels,
                          num_repetitions=repetitions, depth=depth).to(device)

    # log p(x_obs) for every row of x, nan = missing, shape (batch,)
    def forward(self, x):
        return self.spn.log_likelihood(x).flatten()

    # maximum likelihood by gradient descent, as for the gmm
    def fit(self, loader, epochs=30, lr=0.05):
        print("Fitting SPFlow PC...", end="", flush=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for batch in loader:
                if isinstance(batch, (tuple, list)): batch = batch[0]
                loss = -self(batch.to(self.device)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # the message disappears once the fit is done
        print("\r\033[K", end="", flush=True)

    # the conditional methods below take one point x_obs of shape (D,) or a
    # batch (B, D), each row with its own missing pattern (nan = missing), as
    # for the gmm

    # n samples of p(x) or p(x_mis | x_obs), full vectors with the observed
    # entries copied in, shape (n, D) or (B, n, D)
    @torch.no_grad()
    def sample(self, num_samples, x_obs=None):
        if x_obs is None:
            return self.spn.sample(num_samples=num_samples)
        batch = x_obs.shape[:-1]
        evidence = x_obs.unsqueeze(-2).expand(*batch, num_samples, -1).reshape(-1, self.n_features)
        return self.spn.sample_with_evidence(evidence.clone()).view(*batch, num_samples, self.n_features)

    # cf of p(x_mis | x_obs), phi(t) = E[exp(i t^T x_mis) | x_obs], as for the gmm.
    # t are N frequency vectors over all D features, (N, D) for one point or
    # (B, N, D) for a batch, the entries of observed features are ignored.
    # returns complex values of shape (N,) or (B, N)
    @torch.no_grad()
    def characteristic_function(self, t, x_obs):
        return torch.exp(self._log_cf(t, x_obs)).to(torch.complex64)

    # log of the cf above in float64. the circuit with the cf at the missing
    # leaves, minus log p(x_obs), which is the same pass at frequency 0, so
    # phi(0) = 1 exactly. every frequency is one row of the pass, together with
    # its point, in chunks of chunk rows, the pass keeps a few
    # (rows, D, channels, repetitions) tensors
    def _log_cf(self, t, x_obs, chunk=10_000):
        batch, n = x_obs.shape[:-1], t.shape[-2]
        mis_mask = torch.isnan(x_obs)
        # the frequency of every feature, 0 for the observed ones. the first
        # rows are the points at frequency 0, then every point with its N frequencies
        s = (t.double() * mis_mask.unsqueeze(-2)).reshape(-1, self.n_features)
        x_obs = x_obs.double().reshape(-1, self.n_features)
        x = torch.cat([x_obs, x_obs.repeat_interleave(n, dim=0)])
        s = torch.cat([torch.zeros_like(x_obs), s])
        log_values = torch.cat([_log_cf(self.spn.root_node, x_part, s_part).flatten()
                                for x_part, s_part in zip(x.split(chunk), s.split(chunk))])
        log_norm, log_values = log_values[:len(x_obs)], log_values[len(x_obs):]
        return (log_values.view(-1, n) - log_norm[:, None]).view(*batch, n)

    # E[z | x_obs] for z = w^T x, from the slope of the log cf at a small
    # frequency: log phi(eps w_mis) = i eps E[w_mis^T x_mis] + O(eps^2).
    # shape () or (B,)
    @torch.no_grad()
    def z_mean(self, w, x_obs, eps=1e-4):
        log_cf = self._log_cf(eps * w.expand_as(x_obs).unsqueeze(-2), x_obs)[..., 0]
        return torch.nan_to_num(x_obs) @ w + (log_cf.imag / eps).to(w.dtype)

    # distribution of z = w^T x given x_obs as a mixture of gaussians, one
    # component per induced tree: (p(k | x_obs), m, v), each of shape
    # (n_components,), or (B, n_components) for a batch
    @torch.no_grad()
    def z_moments(self, w, x_obs):
        squeeze = x_obs.dim() == 1
        x = x_obs.double().reshape(-1, self.n_features)
        log_weights, m, v = (part.flatten(start_dim=1) for part in _mixture(self.spn.root_node, w.double(), x))
        m = m + (torch.nan_to_num(x) @ w.double())[:, None]
        weights = torch.softmax(log_weights, dim=-1)
        return tuple(part[0] if squeeze else part for part in (weights, m, v))


# ── cf pass ───────────────────────────────────────────────────────────────────

# log of the circuit with the cf at the missing leaves, complex, for the rows of
# x (one per frequency) and the frequencies s, both (N, D). the shapes follow
# spflow: (N, features, channels, repetitions)
def _log_cf(module, x, s):
    if isinstance(module, Normal):
        x, s = x[:, list(module.scope.query)], s[:, list(module.scope.query)]
        loc = module.loc.double()
        scale = module.scale.double()
        mis_mask = torch.isnan(x)[:, :, None, None]
        log_density = torch.distributions.Normal(loc, scale).log_prob(torch.nan_to_num(x)[:, :, None, None])
        s = s[:, :, None, None]
        log_cf = torch.complex(-0.5 * (s * scale)**2, s * loc)
        return torch.where(mis_mask, log_cf, log_density.to(log_cf.dtype))
    if isinstance(module, Factorize):
        # the leaves of every region multiplied, i.e. summed in log space
        child = _log_cf(module.inputs[0], x, s)
        return torch.einsum("bicr,ior->bocr", child, module.indices.to(child.dtype))
    if isinstance(module, SplitConsecutive):
        child = _log_cf(module.inputs, x, s)
        return list(child.split(module.inputs.out_shape.features // module.num_splits, dim=module.dim))
    if isinstance(module, ElementwiseProduct):
        parts = _log_cf(module.inputs[0], x, s)
        channels = module.out_shape.channels
        total = sum(part.expand(-1, -1, channels, -1) for part in parts)
        return total.reshape(len(x), module.out_shape.features, channels, module.out_shape.repetitions)
    # RepetitionMixingLayer is a Sum, so it has to come first
    if isinstance(module, RepetitionMixingLayer):
        child = _log_cf(module.inputs, x, s)
        return torch.logsumexp(child + module.log_weights.double()[None], dim=3)[..., None]
    if isinstance(module, Sum):
        child = _log_cf(module.inputs, x, s)
        return torch.logsumexp(child[:, :, :, None, :] + module.log_weights.double()[None], dim=2)
    raise NotImplementedError(f"cf pass for {type(module).__name__}")


# ── mixture pass ──────────────────────────────────────────────────────────────

# the gaussian mixture of the part of z = w^T x in the scope of every node, as
# (log weights, means, variances), each (points, features, channels,
# repetitions, components), for the rows of x (B, D). the log weights include
# the densities of the observed leaves, so at the root they are the
# unnormalised posterior p(k, x_obs). the observed part of z is added at the end
def _mixture(module, w, x):
    if isinstance(module, Normal):
        w, x = w[list(module.scope.query)], x[:, list(module.scope.query)]
        loc, scale = module.loc.double(), module.scale.double()
        mis = torch.isnan(x)[:, :, None, None]
        log_density = torch.distributions.Normal(loc, scale).log_prob(torch.nan_to_num(x)[:, :, None, None])
        w = w[:, None, None]
        log_weights = torch.where(mis, torch.zeros_like(log_density), log_density)
        m = torch.where(mis, w * loc, torch.zeros_like(loc))
        v = torch.where(mis, (w * scale)**2, torch.zeros_like(scale))
        return log_weights[..., None], m[..., None], v[..., None]
    if isinstance(module, Factorize):
        # one component per leaf, so the product of a region is one component
        # too, with everything summed
        parts = _mixture(module.inputs[0], w, x)
        indices = module.indices.double()
        return tuple(torch.einsum("bicrk,ior->bocrk", part, indices) for part in parts)
    if isinstance(module, SplitConsecutive):
        parts = _mixture(module.inputs, w, x)
        size = module.inputs.out_shape.features // module.num_splits
        return [tuple(part.split(size, dim=1)[i] for part in parts) for i in range(module.num_splits)]
    if isinstance(module, ElementwiseProduct):
        # every component of one child with every component of the other
        (lw1, m1, v1), (lw2, m2, v2) = _mixture(module.inputs[0], w, x)
        pair = lambda a, b: (a[..., :, None] + b[..., None, :]).flatten(start_dim=-2)
        return pair(lw1, lw2), pair(m1, m2), pair(v1, v2)
    if isinstance(module, RepetitionMixingLayer):
        lw, m, v = _mixture(module.inputs, w, x)
        lw = lw + module.log_weights.double()[..., None]
        return tuple(part.flatten(start_dim=3)[:, :, :, None, :] for part in (lw, m, v))
    if isinstance(module, Sum):
        # the components of all children, weighted: (b, f, ci, r, k) -> (b, f, co, r, ci * k)
        lw, m, v = _mixture(module.inputs, w, x)
        co = module.out_shape.channels
        lw = lw[:, :, :, None] + module.log_weights.double()[..., None]
        m, v = (part[:, :, :, None].expand(-1, -1, -1, co, -1, -1) for part in (m, v))
        return tuple(part.permute(0, 1, 3, 4, 2, 5).flatten(start_dim=4) for part in (lw, m, v))
    raise NotImplementedError(f"mixture pass for {type(module).__name__}")
