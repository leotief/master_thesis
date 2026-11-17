# the baselines: estimators (the prediction part, estimate.py) and models (training plus
# an estimator to predict with, train.py)

import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import quad
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401, makes IterativeImputer importable
from sklearn.impute import IterativeImputer

from pclr import chunks, join
from utils.gmm import GMM
from utils.experiments import logistic_regression

# ══ estimators: the prediction part (estimate.py) ════════════════════════════════
#
# like cfi, method(density, w, x_partial, **settings), w and the density given

# sample the missing features and average the predictions. the samples don't depend on
# w, so the gradient is the one of the average. density needs sample
def conditional_sampling(density, w, x_partial, n_samples=1000):
    results = []
    for part in chunks(x_partial, per_point=n_samples * len(w)):
        x_samples = density.sample(n_samples, x_obs=part)
        results.append(torch.sigmoid(x_samples @ w).mean(dim=-1))
    return join(results, x_partial)


# fill the gaps with fixed values (e.g. the training means) and predict once
def mean_imputation(means, w, x_partial):
    x = torch.where(torch.isnan(x_partial), means.to(x_partial), x_partial)
    return torch.sigmoid(x @ w)


# fill the gaps with the density's conditional mean and predict once
def conditional_mean_imputation(density, w, x_partial):
    return torch.sigmoid(density.conditional_mean(x_partial) @ w)


# gauss-hermite nodes, cached per count and device
@lru_cache(maxsize=None)
def hermite_nodes(n, device):
    return tuple(torch.tensor(a, device=device) for a in np.polynomial.hermite.hermgauss(n))

# the integral of every component by gauss-hermite. density needs z_mixture
def mixture_gauss_hermite(density, w, x_partial, n_nodes=50):
    nodes, weights = hermite_nodes(n_nodes, density.device)
    results = []
    for part in chunks(x_partial, per_point=n_nodes * len(w)):
        mix, m, v = density.z_mixture(w, part)
        nodes, weights = nodes.to(m.dtype), weights.to(m.dtype)
        # the floor only matters for the gradient, which is infinite at zero variance
        z = m[..., None] + torch.sqrt((2 * v).clamp_min(1e-30))[..., None] * nodes
        results.append((mix * (torch.sigmoid(z) @ weights)).sum(dim=-1) / np.sqrt(np.pi))
    return join(results, x_partial)


# the integral of every component by scipy's adaptive quadrature, the reference: accurate
# for any spread, but point by point and without gradients. density needs z_mixture
@torch.no_grad()
def mixture_adaptive_quad(density, w, x_partial, epsrel=1e-12):
    if x_partial.dim() == 2:
        return torch.stack([mixture_adaptive_quad(density, w, x, epsrel) for x in x_partial])
    mix, m, v = (t.double().cpu().numpy() for t in density.z_mixture(w, x_partial))
    per_component = []
    for m_k, s_k in zip(m, np.sqrt(v)):
        # nothing missing: z is known
        if s_k == 0:
            per_component.append(expit(m_k))
            continue
        integrand = lambda t: expit(m_k + s_k * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
        # where the sigmoid switches from 0 to 1, so quad doesn't miss it for a wide spread
        step = np.clip(-m_k / s_k, -10, 10)
        per_component.append(quad(integrand, -10, 10, points=[step], epsabs=0, epsrel=epsrel, limit=200)[0])
    return torch.tensor(float(mix @ np.array(per_component)), dtype=torch.float64)


# ══ models: the whole classifier (train.py, real_data.py) ═══════════════════════
#
# like PCLR, fit(x_train, y_train, density) with the incomplete training points and the
# density fit on them (not every model uses it), w, and predict(x_test) with one of the
# estimators

# lr without missing values: trained on the complete training points and predicting
# complete test points, the reference
class CompleteData:
    def fit(self, x_train, y_train, density=None):
        self.w = logistic_regression(x_train, y_train)
        return self

    def predict(self, x_test):
        return torch.sigmoid(x_test @ self.w)

# lr on the gaps filled by the training means
class MeanImputation:
    def fit(self, x_train, y_train, density=None):
        self.means = x_train.nanmean(dim=0)
        self.w = logistic_regression(torch.where(torch.isnan(x_train), self.means, x_train), y_train)
        return self

    def predict(self, x_test):
        return mean_imputation(self.means, self.w, x_test)

# lr on the gaps filled by the density's conditional mean
class ConditionalMeanImputation:
    def fit(self, x_train, y_train, density):
        self.density = density
        with torch.no_grad():
            self.w = logistic_regression(density.conditional_mean(x_train), y_train)
        return self

    def predict(self, x_test):
        return conditional_mean_imputation(self.density, self.w, x_test)

# lr on the gaps filled by multiple chained regressions (mice, sklearn's IterativeImputer,
# every feature regressed on the others in turns), fit on the training points
class MICE:
    def fit(self, x_train, y_train, density=None):
        # 10 rounds are the default; it often stops before its tolerance, which only warns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.imputer = IterativeImputer(max_iter=10, random_state=0).fit(x_train.cpu().numpy())
        self.w = logistic_regression(self.impute(x_train), y_train)
        return self

    def impute(self, x):
        return torch.tensor(self.imputer.transform(x.cpu().numpy()), dtype=x.dtype, device=x.device)

    def predict(self, x_test):
        return torch.sigmoid(self.impute(x_test) @ self.w)

# the generative classifier: one joint model of the features and the label, a gmm with K
# components where every component also has a bernoulli leaf for the label, p(x, y) =
# sum_k pi_k p(x | k) p(y | k). the same kind of density as pclr's with K label
# probabilities more, so the same budget. fit on log p(x_obs, y) of the incomplete training
# points by adam (as GMM.fit), and the prediction by bayes' rule, the missing features
# marginalised: p(y=1 | x_obs) = sum_k p(k | x_obs) p(y=1 | k). no lr, no w
class GenerativeClassifier:
    def __init__(self, K, epochs=50, lr=0.01):
        self.K, self.epochs, self.lr = K, epochs, lr

    def fit(self, x_train, y_train, density=None):
        torch.manual_seed(0)
        self.px = GMM(x_train, self.K, x_train.device)
        self.label_logits = torch.nn.Parameter(torch.zeros(self.K, device=x_train.device))
        optimizer = torch.optim.Adam(list(self.px.parameters()) + [self.label_logits], lr=self.lr)
        data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train.float()), batch_size=64, shuffle=True)
        for _ in range(self.epochs):
            for x, y in data:
                # log p(x_obs, y) = log sum_k pi_k p(x_obs | k) p(y | k)
                log_label = y[:, None] * F.logsigmoid(self.label_logits) + (1 - y[:, None]) * F.logsigmoid(-self.label_logits)
                loss = -torch.logsumexp(self.px.component_log_probs(x) + log_label, dim=-1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x_test):
        return torch.softmax(self.px.component_log_probs(x_test), dim=-1) @ torch.sigmoid(self.label_logits)
