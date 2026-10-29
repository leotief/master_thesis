# gaussian mixture with diagonal covariances, used as the density p(x)
#
# p(x) = sum_k pi_k N(x; mu_k, diag(var_k))
#
# the diagonal covariance is what makes missing data easy here. within one
# component all features are independent, so
#   - marginalising out missing features just means dropping those dims
#   - the conditional p(x_mis | x_obs) is again a mixture with the same means
#     and variances (restricted to the missing dims), only the weights change
#     to the posterior p(k | x_obs)
# both the mc baseline (sample) and pcpg (characteristic_function) use this.

import torch
import torch.nn as nn
import torch.distributions as dist

class GMM(nn.Module):
    # pi are logits of the mixture weights (softmax is applied later), so they
    # don't need to be constrained during gradient descent. same reason for storing
    # log_vars instead of the variances. means_init is usually the first K data points
    def __init__(self, device, K, means_init, random_weights=False, n_features=2):
        super(GMM, self).__init__()
        self.device = device
        self.K = K
        self.n_features = n_features

        if random_weights:
            pi_init = torch.rand(K, dtype=torch.float32)  
            self.pi = nn.Parameter(pi_init)
        else:
            self.pi = nn.Parameter(torch.ones(K))

        self.means = nn.Parameter(means_init.clone().detach())
        self.log_vars = nn.Parameter(torch.zeros(K, n_features))

    # log-likelihood log p(x_obs) for every row of x, shape (batch,)
    # works for complete rows and for rows with nan, each row can have its own pattern
    def forward(self, x):
        weights_log = torch.log_softmax(self.pi, dim=-1)
        variances   = torch.exp(self.log_vars) + 1e-6   # small floor so no variance collapses to 0

        # fill with zeros so log_prob doesn't return nan, masked out below anyway
        x_filled = x.clone()
        x_filled[torch.isnan(x)] = 0.0

        # log density of every feature under every component, shape (batch, K, D)
        log_p_feat = dist.Normal(self.means, variances.sqrt()).log_prob(x_filled.unsqueeze(1))
        # setting missing dims to 0 in log space is the same as integrating them out
        log_p_feat = log_p_feat * (~torch.isnan(x)).float().unsqueeze(1)

        # sum over features gives log p(x | k), then log sum_k pi_k p(x | k)
        return torch.logsumexp(weights_log + log_p_feat.sum(dim=-1), dim=-1)

    # the conditional methods below work on one point x_obs of shape (D,) or a
    # batch (B, D), each row with its own missing pattern (nan = missing). as in
    # forward(), the masks are multiplied in instead of used for indexing, so
    # all rows keep the full D features

    def _component_weights(self, x_obs=None):
        # Prior softmax(pi) if nothing observed, else posterior p(k | x_obs), shape (..., K).
        # the posterior is how the observed features change the mixture: components
        # that explain x_obs well get more weight for predicting x_mis
        if x_obs is None:
            return torch.softmax(self.pi, dim=-1)

        obs_mask = ~torch.isnan(x_obs)
        # likelihood of the observed part under each component, the missing
        # features add 0 in log space
        stds     = torch.sqrt(torch.exp(self.log_vars) + 1e-6)
        log_p    = dist.Normal(self.means, stds).log_prob(torch.nan_to_num(x_obs).unsqueeze(-2))   # (..., K, D)
        log_lik  = (log_p * obs_mask.unsqueeze(-2)).sum(dim=-1)

        # bayes rule p(k | x_obs) ~ pi_k p(x_obs | k), softmax does the normalising
        return torch.softmax(torch.log_softmax(self.pi, dim=-1) + log_lik, dim=-1)

    # samples from p(x), or from p(x_mis | x_obs) if x_obs is given.
    # always returns full vectors, with the observed entries copied in, so they
    # can go straight into w^T x: shape (num_samples, D), or (B, num_samples, D)
    # for a batch of points
    def sample(self, num_samples, x_obs=None):
        with torch.no_grad():
            weights = self._component_weights(x_obs)
            # pick a component per sample (with the posterior weights), then sample
            # from that gaussian. sampling all dims is simpler than only the missing
            # ones, the observed ones get overwritten below anyway
            indices = dist.Categorical(weights).sample((num_samples,))
            if indices.dim() == 2:
                indices = indices.T                  # (B, num_samples)

            means = self.means[indices]
            stds  = torch.sqrt(torch.exp(self.log_vars[indices]) + 1e-6)
            eps   = torch.randn_like(means)
            samples = means + eps * stds

            if x_obs is not None:
                # observed values overwrite the sampled ones
                obs_mask = ~torch.isnan(x_obs)
                samples = torch.where(obs_mask.unsqueeze(-2), x_obs.unsqueeze(-2), samples)
            return samples

    # characteristic function of p(x_mis | x_obs), i.e. phi(t) = E[exp(i t^T x_mis) | x_obs]
    #
    # this is the only thing pcpg needs from the density. for a gaussian the cf is
    # exp(i t^T mu - t^T Sigma t / 2), and for the mixture it is the weighted sum
    #   phi(t) = sum_k p(k | x_obs) exp(i t^T mu_k - sum_d t_d^2 var_kd / 2)
    # with mu_k and var_k restricted to the missing dims.
    # t are N frequency vectors over all D features, (N, D) for one point or
    # (B, N, D) for a batch, the entries of observed features are ignored.
    # returns complex values of shape (N,) or (B, N)
    def characteristic_function(self, t, x_obs=None):
        weights   = self._component_weights(x_obs)
        variances = torch.exp(self.log_vars) + 1e-6
        # only the missing dims matter, without x_obs this is the cf of the full p(x)
        if x_obs is not None:
            t = t * torch.isnan(x_obs).unsqueeze(-2)

        t_dot_mu = t @ self.means.T                        # (..., N, K)
        t_sq_var = t**2 @ variances.T                      # (..., N, K)

        # log of each component's cf. the real part damps it (large t or large
        # variance means small |phi|), the imaginary part is the phase from the mean
        log_cf = torch.complex(-0.5 * t_sq_var, t_dot_mu)
        # weighted sum over components
        return (weights.unsqueeze(-2) * torch.exp(log_cf)).sum(dim=-1)

    # distribution of z = w^T x within each component, given x_obs.
    # inside component k the missing features are independent gaussians, so z is
    # gaussian too: z | k ~ N(m_k, v_k) with
    #   m_k = w_obs^T x_obs + w_mis^T mu_k,mis
    #   v_k = sum over missing d of w_d^2 var_kd
    # returns (p(k | x_obs), m, v), each of shape (K,), or (B, K) for a batch
    def z_mixture(self, w, x_obs):
        weights  = self._component_weights(x_obs)
        mis_mask = torch.isnan(x_obs)
        w_mis    = w * mis_mask

        variances = torch.exp(self.log_vars) + 1e-6
        m = (torch.nan_to_num(x_obs) @ w).unsqueeze(-1) + w_mis @ self.means.T
        v = w_mis**2 @ variances.T
        return weights, m, v

    # mean of z = w^T x given x_obs, i.e. sum_k p(k | x_obs) m_k, shape () or (B,).
    # the pcpg estimators use its sign to decide whether to flip w (sigma(z) = 1 - sigma(-z))
    def z_mean(self, w, x_obs):
        weights, m, _ = self.z_mixture(w, x_obs)
        return (weights * m).sum(dim=-1)

    # maximum likelihood by gradient descent (adam) instead of em, mainly because
    # forward() already handles nan, so the same code works on incomplete data.
    # loss is the negative mean log-likelihood of each batch
    def fit(self, loader, epochs=50, lr=0.01):
        print("Fitting GMM...", end="", flush=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for _ in range(epochs):
            for batch in loader:
                # loaders may return (x, y), only x is needed
                if isinstance(batch, (tuple, list)): batch = batch[0]
                batch = batch.to(self.device)

                loss = -self(batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # the message disappears once the fit is done
        print("\r\033[K", end="", flush=True)