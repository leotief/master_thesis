import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# the predictions in the loss are kept this far from 0 and 1, so the log stays finite
EPS = 1e-6

# a density (gmm or pc) and a logistic regression as one model, trained jointly on potentially incomplete data
class PCLR(nn.Module):
    def __init__(self, n_freq=100, C=1.0, density_mode="untrained"):
        super().__init__()
        self.n_freq = n_freq                # cfi's frequencies
        self.C = C                          # the l2 penalty, as in sklearn
        self.density_mode = density_mode    # "untrained": a new density, trained with w. "fitted": the one given to fit, trained with w. "fixed": the one given to fit, not trained
        self.density = None                 # the density, set by fit
        self.w = None                       # the weights of the logistic regression, set by fit

    # the probability of label 1 with the missing features averaged out. used for the
    # loss and by the experiments to score the model
    def predict(self, x_partial):
        return cfi(self.density, self.w, x_partial, self.n_freq)

    # the two parts of the log-likelihood, each averaged over the points: how likely the
    # observed features are under the density, and how well the prediction fits the label
    def log_likelihood(self, x_partial, y):
        log_density = self.density(x_partial).mean()
        p = self.predict(x_partial).to(y.dtype).clamp(EPS, 1 - EPS)
        log_label = -F.binary_cross_entropy(p, y)
        return log_density, log_label

    # the training loss: the negative log-likelihood plus the l2 penalty
    def objective(self, x_partial, y):
        log_density, log_label = self.log_likelihood(x_partial, y)
        l2_penalty = self.w.square().sum() / (2 * self.C * len(x_partial))
        return -(log_density + log_label) + l2_penalty

    # trains the model on the incomplete training points until it converges, like sklearn
    # does for a logistic regression. density is the one fitted on these points, the
    # experiments give it to every model
    def fit(self, x_partial, y, density, max_iter=1000):
        # the density to train: a new untrained gmm of the same size, or a copy of the
        # fitted one, so the given density is never changed
        if self.density_mode == "untrained":
            self.density = type(density)(x_partial, density.K, density.device)
        else:
            self.density = copy.deepcopy(density)

        for parameter in self.density.parameters():
            parameter.requires_grad_(True)

        self.w = nn.Parameter(torch.zeros(x_partial.shape[1], device=x_partial.device))

        # l-bfgs on all points at once: the loss has no randomness, so no minibatches
        parameters = [self.w] if self.density_mode == "fixed" else list(self.parameters())
        optimizer = torch.optim.LBFGS(parameters, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = self.objective(x_partial, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


# ── cfi ───────────────────────────────────────────────────────────────────────

# beyond this frequency the weights are below double precision
S_MAX = 12.0

# the density model needs to expose char_func, z_mean and z_var
def cfi(density, w, x_partial, n_freq=200):
    results = []
    for part in chunks(x_partial, per_point=n_freq * len(w)):

        mis_mask = torch.isnan(part)
        mean = density.z_mean(w, part)
        std = density.z_var(w, part).clamp(min=0).sqrt()

        # spacing of the frequencies, the finer of two: h_reach keeps every z inside the
        # stretch where the result is correct, h_default lets the frequencies reach S_MAX
        z_reach = np.pi / (mean.abs() + 6 * std)
        h_default = torch.full_like(mean, S_MAX / n_freq)
        h = torch.minimum(z_reach, h_default).detach()

        # the frequencies, evenly spaced by h, for z (s) and for the missing features of x (t)
        s = h[..., None] * torch.arange(1, n_freq + 1, device=w.device, dtype=w.dtype)
        # the frequencies times the missing weight
        swm = s[..., None] * (w * mis_mask).unsqueeze(-2)

        # the observed part
        wo_xo = (w * torch.nan_to_num(part)).sum(dim=-1)[..., None]
        # the cf of z (cf of observed part times conditional cf of missing part from the density model)  
        cfz = torch.exp(torch.complex(torch.zeros_like(s), s * wo_xo)) * density.char_func(swm, x_obs=part)

        # the integrand 
        integrand = cfz.imag / torch.sinh(np.pi * s)
        # trapezoid rule: the sum times the spacing, plus half the value at frequency 0 (mean / pi)
        integral = h * (mean / (2 * np.pi) + integrand.sum(dim=-1))

        results.append(0.5 + integral)
        
    return join(results, x_partial)

# ── batching ──────────────────────────────────────────────────────────────────
#
# cfi and the baselines build tensors that grow with the number of points, so large
# batches are split into chunks small enough for memory and joined again afterwards,
# with their gradients. a single point is one chunk

MAX_ELEMENTS = 2**25

def chunks(x_partial, per_point):
    if x_partial.dim() == 1:
        return [x_partial]
    return x_partial.split(max(1, MAX_ELEMENTS // per_point))

def join(results, x_partial):
    if x_partial.dim() == 1:
        return results[0]
    return torch.cat(results)
