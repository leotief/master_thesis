# helpers of the experiment scripts (train.py, estimate.py) that are not part
# of their story: fitting the density model, the logistic regression, finding the optimum of an objective, the
# accuracy, the errors of the estimators, and the comparisons of all settings

import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from side.utils.data import loader
from side.utils.plots import (plot_estimate_comparison, plot_train_accuracy_missing, plot_train_difference,
                              plot_train_accuracy, save_figure)
from side.utils.gmm import GMM
from side.utils.spflow_pc import SPFlowPC

# the density models: K gmm components, and the circuit of both pcs (gaussians
# per feature, sum nodes per region, random structures)
K              = 4
PC_CHANNELS    = 4
PC_REPETITIONS = 3
PC_DEPTH       = 2


# the density model ("gmm", "pc" or "spflow") fit on the points x (nan =
# missing, the likelihood leaves those features out), saved as path and
# loaded from there if it was fit before, then frozen
def fit_density(model, x, path, device):
    n_features = x.shape[1]
    torch.manual_seed(0)
    if model == "gmm":
        # the means start at the first K points, their gaps filled with the means
        first = torch.where(torch.isnan(x[:K]), x.nanmean(dim=0), x[:K])
        px = GMM(device, K=K, means_init=first, n_features=n_features).to(device)
    elif model == "pc":
        px = SPFlowPC(device, n_features, channels=PC_CHANNELS, repetitions=PC_REPETITIONS, depth=PC_DEPTH)

    if os.path.exists(path):
        px.load_state_dict(torch.load(path, map_location=device))
    else:
        px.fit(loader(x))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(px.state_dict(), path)

    for parameter in px.parameters():
        parameter.requires_grad_(False)
    return px


# the w that minimises objective(predict, w, x, y, n_train) on all points x,
# by full batch l-bfgs from w = 0. predict has to be smooth and deterministic
def optimum(objective, predict, x, y, n_train, device):
    w = torch.zeros(x.shape[1], device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w], max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = objective(predict, w, x, y, n_train)
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach()


# sklearn's logistic regression on the complete points x with labels y, without
# an intercept, as a tensor w
def logistic_regression(x, y, C):
    classifier = LogisticRegression(C=C, fit_intercept=False, max_iter=1000).fit(x.cpu(), y.cpu())
    return torch.tensor(classifier.coef_[0], dtype=torch.float32, device=x.device)


# the fraction of the predictions p(y=1) that are on the right side of 0.5
def accuracy(p, y):
    return ((p > 0.5).float() == y).float().mean().item()


# the scores of train.py as series for the figures: score -> missing rate ->
# model -> one value per number of features
def _series(scores, features, missing_rates):
    results = {}
    for score in scores[(features[0], missing_rates[0])]["Exact"]:
        results[score] = {}
        for rate in missing_rates:
            results[score][rate] = {}
            for name in scores[(features[0], rate)]:
                results[score][rate][name] = [scores[(n, rate)][name][score] for n in features]
    return results


# the comparison of train.py: scores[(n_features, missing_rate)][model][score]
# holds the scores of every model in every setting of one configuration (suffix
# ends in _norm for the standardized data). saved as cache/<folder>/scores<suffix>.json,
# and drawn against the number of features, a column per missing rate, in
# figures/<folder>/ and the thesis: the accuracy on the incomplete test points
# of this configuration (comparison<suffix>_accuracy_missing), and once both
# configurations are saved, for both: the difference to exact, the accuracy on
# the complete test points, and on the incomplete ones with every model's own
# rule against the complete-data classifier with the exact expected prediction
# (comparison<suffix without _norm>_difference, _accuracy_complete, _accuracy_own)
def compare(scores, features, missing_rates, folder, suffix):
    os.makedirs(f"cache/{folder}", exist_ok=True)
    with open(f"cache/{folder}/scores{suffix}.json", "w") as file:
        json.dump({f"{n}_{rate}": scores[(n, rate)] for n, rate in scores}, file)
    save_figure(plot_train_accuracy_missing, folder, f"comparison{suffix}_accuracy_missing", features,
                missing_rates, _series(scores, features, missing_rates))

    base = suffix.removesuffix("_norm")
    paths = {"raw": f"cache/{folder}/scores{base}.json", "standardized": f"cache/{folder}/scores{base}_norm.json"}
    if not all(os.path.exists(path) for path in paths.values()):
        return
    both = {}
    for config, path in paths.items():
        with open(path) as file:
            saved = json.load(file)
        both[config] = _series({(n, rate): saved[f"{n}_{rate}"] for n in features for rate in missing_rates},
                               features, missing_rates)
    save_figure(plot_train_difference, folder, f"comparison{base}_difference", features, missing_rates, both)
    save_figure(plot_train_accuracy, folder, f"comparison{base}_accuracy_complete", features, missing_rates,
                both, "accuracy_complete", None)
    save_figure(plot_train_accuracy, folder, f"comparison{base}_accuracy_own", features, missing_rates,
                both, "accuracy_own", "accuracy_exact")


# the errors of estimate.py. every estimator is called as estimator.estimate(w,
# x_partial, **kwargs), on a batch of test points (B, D) it returns an array
# (B,), the kwargs are the swept counts (n_samples, n_outer, n_inner, n_nodes)

# root mean squared error over the test points (B, D), against their exact
# values (B,). for the random methods every test point gets one run, so the
# noise is part of the error
def rmse(estimator, w, points, truths, **kwargs):
    return float(np.sqrt(np.mean(np.square(estimator.estimate(w, points, **kwargs) - truths))))

# rmse and the wall clock time of one batched call over all points, and the
# estimates of that call. the truths belong to the first len(truths) points,
# the error is over those, the time is for all of them. after one call to warm
# up (gpu kernels, cached nodes), the call is repeated `repeats` times and the
# time is the median. the estimators return arrays on the cpu, so the gpu has
# finished when the clock stops
def timed_rmse(estimator, w, points, truths, repeats=3, **kwargs):
    estimator.estimate(w, points, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        estimates = estimator.estimate(w, points, **kwargs)
        times.append(time.perf_counter() - t0)
        if len(times) == 1:
            first = estimates
            error = float(np.sqrt(np.mean(np.square(estimates[:len(truths)] - truths))))
    return error, float(np.median(times)), first


# the comparison of estimate.py: scores[(n_features, missing_rate)][estimator]
# holds the rmse and the time of every estimator in every setting. one figure,
# against the number of features, one column per missing rate: the rmse on
# top, the time for all n_points test points below, as figures/<folder>/<name>
# and in the thesis
def compare_estimates(scores, features, missing_rates, n_points, folder, name):
    errors, times = {}, {}
    for rate in missing_rates:
        errors[rate], times[rate] = {}, {}
        for estimator in scores[(features[0], rate)]:
            errors[rate][estimator] = [scores[(n, rate)][estimator][0] for n in features]
            times[rate][estimator] = [scores[(n, rate)][estimator][1] for n in features]
    save_figure(plot_estimate_comparison, folder, name, features, missing_rates, errors, times, n_points)
