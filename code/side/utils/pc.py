# a small probabilistic circuit as the density p(x), written from scratch with
# only what the experiments need. the same kind of circuit as the rat-spn of
# spflow_pc.py, with the same interface as the gmm (gmm.py):
#   forward(x), fit(loader)            training, log p(x_obs) with nan = missing
#   sample(n, x_obs)                   mc, samples of p(x_mis | x_obs)
#   characteristic_function(t, x_obs)  pcpg, the cf of p(x_mis | x_obs)
#   z_mean(w, x_obs)                   pcpg's sign flip
#   z_moments(w, x_obs)                the exact baselines, p(z | x_obs) as a
#                                      mixture of gaussians
#
# the structure, for every one of R repetitions:
#   leaves   C gaussians per feature
#   regions  the features in a random order, cut into G = 2^depth regions. a
#            region multiplies the leaves of its features, channel c with c
#   levels   depth times: neighbouring regions multiplied (channel c with c),
#            then a sum layer with C outputs (1 in the last level)
#   root     a mixture over the repetitions
#
# all passes go through _circuit with other values at the leaves: the log
# density (log-likelihood), or at the missing features the log cf of the leaf
# gaussian (cf, in complex numbers). products add logs, sums are a logsumexp,
# both work for complex numbers. the mixture pass (z_moments) follows the same
# structure with gaussian mixtures as values, one component per induced tree,
# R * C^(2^depth - 1) of them, so only for small circuits

import torch
import torch.nn as nn

class PC(nn.Module):
    def __init__(self, device, n_features, channels=4, repetitions=3, depth=2):
        super().__init__()
        self.device = device
        self.n_features = n_features
        D, C, R, G = n_features, channels, repetitions, 2**depth
        assert D >= G, f"needs at least {G} features for depth {depth}"

        # region of every feature, per repetition (R, D), and as a one-hot (D, G, R)
        order = torch.stack([torch.randperm(D) for _ in range(R)])
        region = torch.empty(R, D, dtype=torch.long)
        region.scatter_(1, order, (torch.arange(D) * G // D).expand(R, -1))
        self.register_buffer("region", region.to(device))
        self.register_buffer("one_hot", nn.functional.one_hot(region, G).permute(1, 2, 0).float().to(device))

        self.loc = nn.Parameter(torch.randn(D, C, R, device=device))
        self.log_scale = nn.Parameter(torch.zeros(D, C, R, device=device))
        # sum weights (regions, C in, C out, R) of every level, as logits
        self.sum_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(G >> (l + 1), C, C if l < depth - 1 else 1, R, device=device))
             for l in range(depth)])
        self.mix_logits = nn.Parameter(torch.zeros(R, device=device))

    # log densities of the leaves (N, D, C, R) for the rows of x (N, D), 0 where missing
    def _leaf_log_density(self, x):
        log_density = torch.distributions.Normal(self.loc, self.log_scale.exp()).log_prob(
            torch.nan_to_num(x)[:, :, None, None])
        return torch.where(torch.isnan(x)[:, :, None, None], 0.0, log_density)

    # the circuit on log values at the leaves (N, D, C, R), real or complex,
    # returns the log value of the root (N,). products, the values before
    # every sum layer (N, regions, C, R), are appended to it if given
    def _circuit(self, leaves, products=None):
        real = leaves.real.dtype
        x = torch.einsum("ndcr,dgr->ngcr", leaves, self.one_hot.to(leaves.dtype))
        for logits in self.sum_logits:
            x = x[:, 0::2] + x[:, 1::2]
            if products is not None:
                products.append(x)
            log_weights = torch.log_softmax(logits, dim=1).to(real)
            x = torch.logsumexp(x[:, :, :, None] + log_weights, dim=2)
        return torch.logsumexp(x[:, 0, 0] + torch.log_softmax(self.mix_logits, dim=0).to(real), dim=-1)

    # log p(x_obs) for every row of x, nan = missing, shape (batch,)
    def forward(self, x):
        return self._circuit(self._leaf_log_density(x))

    # maximum likelihood by gradient descent, as for the gmm
    def fit(self, loader, epochs=30, lr=0.05):
        print("Fitting PC...", end="", flush=True)
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

    # n samples of p(x_mis | x_obs), full vectors with the observed entries
    # copied in, shape (n, D) or (B, n, D). the pass up once per point, then
    # top-down for every sample: a repetition, and at every sum node a child,
    # with the posterior weights (weight times the value of the child given
    # x_obs), then the missing features from the chosen leaves
    @torch.no_grad()
    def sample(self, num_samples, x_obs=None):
        if x_obs is None:
            x_obs = torch.full((self.n_features,), torch.nan, device=self.device)
        batch = x_obs.shape[:-1]
        x = x_obs.reshape(-1, self.n_features)
        products = []
        self._circuit(self._leaf_log_density(x), products)
        # one row per sample, the point it belongs to
        point = torch.arange(len(x), device=self.device).repeat_interleave(num_samples)
        # the repetition: weight times the value of the repetition at the root
        top = torch.logsumexp(products[-1][:, :, :, None] + torch.log_softmax(self.sum_logits[-1], dim=1),
                              dim=2)[:, 0, 0]
        r = torch.distributions.Categorical(logits=top + torch.log_softmax(self.mix_logits, dim=0)).sample(
            (num_samples,)).T.reshape(-1)
        # the chosen output channel of every region, from the top level down
        channel = torch.zeros(len(r), 1, dtype=torch.long, device=self.device)
        for logits, values in zip(reversed(self.sum_logits), reversed(products)):
            log_weights = torch.log_softmax(logits, dim=1).permute(3, 0, 2, 1)[r]       # (S, regions, C out, C in)
            log_weights = log_weights.gather(2, channel[:, :, None, None].expand(-1, -1, 1, log_weights.shape[-1]))
            values = values[point, :, :, r]                                            # (S, regions, C in)
            channel = torch.distributions.Categorical(logits=log_weights[:, :, 0] + values).sample()
            # both regions of a product get the same channel
            channel = channel.repeat_interleave(2, dim=1)
        # the leaf of every feature: the channel of its region
        leaf = channel.gather(1, self.region[r])                                       # (S, D)
        features = torch.arange(self.n_features, device=self.device)
        loc = self.loc[features, leaf, r[:, None]]
        scale = self.log_scale.exp()[features, leaf, r[:, None]]
        samples = loc + scale * torch.randn_like(loc)
        samples = torch.where(torch.isnan(x[point]), samples, x[point])
        return samples.view(*batch, num_samples, self.n_features)

    # cf of p(x_mis | x_obs), phi(t) = E[exp(i t^T x_mis) | x_obs], as for the gmm.
    # t are N frequency vectors over all D features, (N, D) for one point or
    # (B, N, D) for a batch, the entries of observed features are ignored.
    # returns complex values of shape (N,) or (B, N)
    @torch.no_grad()
    def characteristic_function(self, t, x_obs):
        return torch.exp(self._log_cf(t, x_obs)).to(torch.complex64)

    # log of the cf above in float64: the circuit with the log cf of the leaf
    # gaussian, i s mu - s^2 sigma^2 / 2, at the missing features, minus log
    # p(x_obs), which is the same pass at frequency 0, so phi(0) = 1 exactly.
    # every frequency is one row of the pass, together with its point, in
    # chunks that keep the leaves (rows, D, C, R) below max_elements
    def _log_cf(self, t, x_obs, max_elements=2**24):
        batch, n = x_obs.shape[:-1], t.shape[-2]
        mis = torch.isnan(x_obs).reshape(-1, 1, self.n_features)
        x = x_obs.double().reshape(-1, 1, self.n_features)
        # the first frequency of every point is 0
        s = torch.cat([torch.zeros_like(x), t.double().reshape(len(x), n, -1)], dim=1) * mis
        x, s = x.expand(-1, n + 1, -1).reshape(-1, self.n_features), s.reshape(-1, self.n_features)
        loc, scale = self.loc.double(), self.log_scale.exp().double()
        chunk = max(1, max_elements // self.loc.numel())
        log_values = []
        for x_part, s_part in zip(x.split(chunk), s.split(chunk)):
            s_part = s_part[:, :, None, None]
            log_cf = torch.complex(-0.5 * (s_part * scale)**2, s_part * loc)
            log_density = torch.distributions.Normal(loc, scale).log_prob(torch.nan_to_num(x_part)[:, :, None, None])
            leaves = torch.where(torch.isnan(x_part)[:, :, None, None], log_cf, log_density.to(log_cf.dtype))
            log_values.append(self._circuit(leaves))
        log_values = torch.cat(log_values).view(-1, n + 1)
        return (log_values[:, 1:] - log_values[:, :1]).view(*batch, n)

    # E[z | x_obs] for z = w^T x, from the slope of the log cf at a small
    # frequency: log phi(eps w_mis) = i eps E[w_mis^T x_mis] + O(eps^2).
    # shape () or (B,)
    @torch.no_grad()
    def z_mean(self, w, x_obs, eps=1e-4):
        log_cf = self._log_cf(eps * w.expand_as(x_obs).unsqueeze(-2), x_obs)[..., 0]
        return torch.nan_to_num(x_obs) @ w + (log_cf.imag / eps).to(w.dtype)

    # distribution of z = w^T x given x_obs as a mixture of gaussians, one
    # component per induced tree: (p(k | x_obs), m, v), each of shape
    # (n_components,), or (B, n_components) for a batch. every node holds the
    # mixture of its part of z as (log weights, means, variances), each
    # (B, regions, C, R, components): a missing leaf is the gaussian w_d x_d,
    # an observed one weighs its density and adds nothing (its part of z is
    # added at the end), products combine the components of the two regions
    # pairwise, sums put the components of their children together
    @torch.no_grad()
    def z_moments(self, w, x_obs):
        squeeze = x_obs.dim() == 1
        x = x_obs.double().reshape(-1, self.n_features)
        mis = torch.isnan(x)[:, :, None, None]
        loc, scale, w = self.loc.double(), self.log_scale.exp().double(), w.double()[:, None, None]
        log_density = torch.distributions.Normal(loc, scale).log_prob(torch.nan_to_num(x)[:, :, None, None])
        parts = (torch.where(mis, 0.0, log_density), torch.where(mis, w * loc, 0.0), torch.where(mis, (w * scale)**2, 0.0))
        lw, m, v = (torch.einsum("bdcr,dgr->bgcr", part, self.one_hot.double())[..., None] for part in parts)
        pair = lambda a: (a[:, 0::2, ..., :, None] + a[:, 1::2, ..., None, :]).flatten(start_dim=-2)
        for logits in self.sum_logits:
            lw, m, v = pair(lw), pair(m), pair(v)
            # (B, regions, C in, R, k) -> (B, regions, C out, R, C in * k)
            lw = lw[:, :, :, None] + torch.log_softmax(logits, dim=1).double()[..., None]
            m, v = (part[:, :, :, None].expand(-1, -1, -1, lw.shape[3], -1, -1) for part in (m, v))
            lw, m, v = (part.permute(0, 1, 3, 4, 2, 5).flatten(start_dim=4) for part in (lw, m, v))
        # the root: one region with one channel, mixed over the repetitions
        lw = lw[:, 0, 0] + torch.log_softmax(self.mix_logits, dim=0).double()[:, None]
        lw, m, v = (part.flatten(start_dim=1) for part in (lw, m[:, 0, 0], v[:, 0, 0]))
        m = m + (torch.nan_to_num(x) @ w[:, 0, 0])[:, None]
        weights = torch.softmax(lw, dim=-1)
        return tuple(part[0] if squeeze else part for part in (weights, m, v))
