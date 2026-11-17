# a gaussian mixture, the density model of the features. every component treats the
# features as independent, which makes missing features easy: they are simply left out,
# and what is left is again a mixture, only with new weights. missing entries are nan

import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.cluster import KMeans

# the smallest variance a component may have. without the floor a component
# shrinks onto a value many points share (the zeros of pima or spambase) and the
# likelihood grows without bound
MIN_VAR = 1e-2

class GMM(nn.Module):
    def __init__(self, x, K, device):
        super().__init__()
        self.device = device
        self.K = K

        # init means, variances and weights with k-means, otherwise one component can take
        # over all the data. k-means needs complete points, so the gaps get the feature means
        filled = torch.where(torch.isnan(x), x.nanmean(dim=0), x).cpu().numpy().astype(np.float64)
        kmeans = KMeans(K, n_init=4, random_state=0).fit(filled)
        # a cluster with a single point has no spread, it gets the spread of all points
        variances = np.stack([filled[kmeans.labels_ == k].var(axis=0) if (kmeans.labels_ == k).sum() > 1 else filled.var(axis=0) for k in range(K)])
        weights = np.bincount(kmeans.labels_, minlength=K) / len(filled)

        self.means = nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        self.log_vars = nn.Parameter(torch.log(torch.tensor(variances, dtype=torch.float32).clamp_min(MIN_VAR)))
        self.pi = nn.Parameter(torch.log(torch.tensor(weights, dtype=torch.float32).clamp_min(1e-6)))
        self.to(device)

    def variances(self):
        return torch.exp(self.log_vars).clamp_min(MIN_VAR)

    # the log-likelihood of the observed features of every point, what fit maximises
    def forward(self, x):
        return torch.logsumexp(self.component_log_probs(x), dim=-1)

    # one number per point and component: how likely it is that the point came from that
    # component and has these observed features, as a log. forward sums it over the
    # components, posterior_weights normalises it
    def component_log_probs(self, x):
        observed = (~torch.isnan(x)).float()
        x = torch.nan_to_num(x)
        variances = self.variances()
        precisions = 1 / variances
        # distances of the points to the means over the observed features, multiplied out
        # into matrix products so no point × component × feature tensor is built
        distances = (observed * x**2) @ precisions.T - 2 * (observed * x) @ (self.means * precisions).T + observed @ (self.means**2 * precisions).T
        normalisation = observed @ torch.log(2 * math.pi * variances).T
        return torch.log_softmax(self.pi, dim=-1) - 0.5 * (distances + normalisation)

    
    # how much each component counts for a point once its observed features are known.
    # components that explain them well get more weight. every conditional below uses it
    def posterior_weights(self, x_obs=None):
        if x_obs is None:
            return torch.softmax(self.pi, dim=-1)
        return torch.softmax(self.component_log_probs(x_obs), dim=-1)

    # samples of the missing features given the observed ones, as complete points with the
    # observed values kept. the monte carlo baseline averages the prediction over them
    def sample(self, num_samples, x_obs=None):
        with torch.no_grad():
            # pick a component for every sample, then draw all features from it
            weights = self.posterior_weights(x_obs)
            indices = dist.Categorical(weights).sample((num_samples,))
            if indices.dim() == 2:
                indices = indices.T

            means = self.means[indices]
            stds  = torch.sqrt(self.variances()[indices])
            samples = means + torch.randn_like(means) * stds

            if x_obs is not None:
                obs_mask = ~torch.isnan(x_obs)
                samples = torch.where(obs_mask.unsqueeze(-2), x_obs.unsqueeze(-2), samples)
            return samples

    # fits the mixture to the points of the loader by maximising the likelihood with adam.
    # gradient descent instead of em, because the likelihood already handles missing
    # features, so incomplete data needs no extra code
    def fit(self, loader, epochs=50, lr=0.01):
        print("Fitting GMM...", end="", flush=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for _ in range(epochs):
            for batch in loader:
                if isinstance(batch, (tuple, list)): batch = batch[0]
                batch = batch.to(self.device)

                loss = -self(batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # clears the message again
        print("\r\033[K", end="", flush=True)


    # ---- stuff that's needed for PCLR ----

    # the characteristic function of the missing features given the observed ones, at the
    # frequencies t. it is all cfi needs from the density
    def char_func(self, t, x_obs=None):
        weights   = self.posterior_weights(x_obs)
        variances = self.variances()
        # only the missing features vary, so the frequencies of the observed ones are dropped
        if x_obs is not None:
            t = t * torch.isnan(x_obs).unsqueeze(-2)

        # each component's value: the spread damps it, the mean turns its phase
        log_cf = torch.complex(-0.5 * (t**2 @ variances.T), t @ self.means.T)
        return (weights.unsqueeze(-2) * torch.exp(log_cf)).sum(dim=-1)

    # the score of the logistic regression given the observed features is again a mixture,
    # one gaussian per component. returns the weights, means and variances of that mixture,
    # used by the quadrature baselines and by z_mean and z_var
    def z_mixture(self, w, x_obs):
        weights  = self.posterior_weights(x_obs)
        w_mis    = w * torch.isnan(x_obs)
        # the observed features give a fixed part of the score, the missing ones add each
        # component's mean and spread
        m = (torch.nan_to_num(x_obs) @ w).unsqueeze(-1) + w_mis @ self.means.T
        v = w_mis**2 @ self.variances().T
        return weights, m, v

    # the expected score given the observed features. cfi chooses its step with it
    def z_mean(self, w, x_obs):
        weights, m, _ = self.z_mixture(w, x_obs)
        return (weights * m).sum(dim=-1)

    # the variance of the score given the observed features. cfi chooses its step with it
    def z_var(self, w, x_obs):
        weights, m, v = self.z_mixture(w, x_obs)
        mean = (weights * m).sum(dim=-1, keepdim=True)
        # the spread inside the components plus how far apart their means are
        return (weights * (v + (m - mean)**2)).sum(dim=-1)
    

    # ---- stuff for baselines ----

    # the points with every missing feature replaced by its expected value given the
    # observed ones, for the conditional mean imputation baseline
    def conditional_mean(self, x_obs):
        weights = self.posterior_weights(x_obs)
        return torch.where(torch.isnan(x_obs), weights @ self.means, x_obs)