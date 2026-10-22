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

import time
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

    def _component_weights(self, x_obs=None):
        # Prior softmax(pi) if nothing observed, else posterior p(k | x_obs).
        # the posterior is how the observed features change the mixture: components
        # that explain x_obs well get more weight for predicting x_mis
        if x_obs is None:
            return torch.softmax(self.pi, dim=-1)

        obs_mask = ~torch.isnan(x_obs)
        # likelihood of the observed part under each component
        stds_obs = torch.sqrt(torch.exp(self.log_vars[:, obs_mask]) + 1e-6)
        log_lik  = dist.Normal(self.means[:, obs_mask], stds_obs).log_prob(x_obs[obs_mask]).sum(dim=-1)

        # bayes rule p(k | x_obs) ~ pi_k p(x_obs | k), softmax does the normalising
        return torch.softmax(torch.log_softmax(self.pi, dim=-1) + log_lik, dim=-1)

    # samples from p(x), or from p(x_mis | x_obs) if x_obs is given.
    # always returns full vectors of shape (num_samples, n_features), with the
    # observed entries copied in, so they can go straight into w^T x
    def sample(self, num_samples, x_obs=None):
        with torch.no_grad():
            weights = self._component_weights(x_obs)
            # pick a component per sample (with the posterior weights), then sample
            # from that gaussian. sampling all dims is simpler than only the missing
            # ones, the observed ones get overwritten below anyway
            indices = dist.Categorical(weights).sample((num_samples,))

            means = self.means[indices]
            stds  = torch.sqrt(torch.exp(self.log_vars[indices]) + 1e-6)
            eps   = torch.randn(num_samples, self.n_features, device=self.device)
            samples = means + eps * stds

            if x_obs is not None:
                obs_mask = ~torch.isnan(x_obs)
                # observed values overwrite the sampled ones
                samples[:, obs_mask] = x_obs[obs_mask]
            return samples

    # characteristic function of p(x_mis | x_obs), i.e. phi(t) = E[exp(i t^T x_mis) | x_obs]
    #
    # this is the only thing pcpg needs from the density. for a gaussian the cf is
    # exp(i t^T mu - t^T Sigma t / 2), and for the mixture it is the weighted sum
    #   phi(t) = sum_k p(k | x_obs) exp(i t^T mu_k - sum_d t_d^2 var_kd / 2)
    # with mu_k and var_k restricted to the missing dims.
    # t has one entry per missing feature, either a single vector (D_mis,) or a
    # batch (N, D_mis). returns complex values of shape () or (N,)
    def characteristic_function(self, t, x_obs=None):
        weights = self._component_weights(x_obs)
        # without x_obs this is just the cf of the full p(x)
        if x_obs is None:
            means, variances = self.means, torch.exp(self.log_vars) + 1e-6
        else:
            # only the missing dims matter here
            mis_mask  = torch.isnan(x_obs)
            means     = self.means[:, mis_mask]
            variances = torch.exp(self.log_vars[:, mis_mask]) + 1e-6

        # treat a single t as a batch of one and undo that at the end
        squeeze = t.dim() == 1
        if squeeze:
            t = t.unsqueeze(0)

        t_exp    = t.unsqueeze(1)                          # (N, 1, D_eff)
        t_dot_mu = (t_exp * means).sum(dim=-1)             # (N, K)
        t_sq_var = (t_exp**2 * variances).sum(dim=-1)      # (N, K)

        # log of each component's cf. the real part damps it (large t or large
        # variance means small |phi|), the imaginary part is the phase from the mean
        log_cf = torch.complex(-0.5 * t_sq_var, t_dot_mu)
        # weighted sum over components, shape (N,)
        cf = (weights * torch.exp(log_cf)).sum(dim=-1)
        return cf.squeeze(0) if squeeze else cf

    # distribution of z = w^T x within each component, given x_obs.
    # inside component k the missing features are independent gaussians, so z is
    # gaussian too: z | k ~ N(m_k, v_k) with
    #   m_k = w_obs^T x_obs + w_mis^T mu_k,mis
    #   v_k = sum over missing d of w_d^2 var_kd
    # returns (p(k | x_obs), m, v), each of shape (K,). used by PCPGClosedForm
    def z_moments(self, w, x_obs):
        weights  = self._component_weights(x_obs)
        obs_mask = ~torch.isnan(x_obs)
        mis_mask = ~obs_mask

        variances = torch.exp(self.log_vars[:, mis_mask]) + 1e-6
        m = w[obs_mask] @ x_obs[obs_mask] + self.means[:, mis_mask] @ w[mis_mask]
        v = variances @ w[mis_mask] ** 2
        return weights, m, v

    # mean of z = w^T x given x_obs, i.e. sum_k p(k | x_obs) m_k. the pcpg
    # estimators use its sign to decide whether to flip w (sigma(z) = 1 - sigma(-z))
    def z_mean(self, w, x_obs):
        weights, m, _ = self.z_moments(w, x_obs)
        return weights @ m

    # maximum likelihood by gradient descent (adam) instead of em, mainly because
    # forward() already handles nan, so the same code works on incomplete data.
    # loss is the negative mean log-likelihood of each batch
    def fit(self, loader, epochs=50, lr=0.01):
        print("Fitting GMM...", end=" ", flush=True)
        t0        = time.perf_counter()
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

        print(f"done ({time.perf_counter() - t0:.1f}s)")