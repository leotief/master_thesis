# helpers of the experiment scripts (train.py, estimate.py) that are not part
# of their story: fitting the density model, the logistic regression, finding the optimum of an objective, the
# accuracy, the errors of the estimators, and the comparisons of all settings

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from side.utils.data import loader
from side.utils.plots import (plot_convergence, plot_estimate_comparison, plot_training_comparison,
                              plot_weight_error, save_figure)
from side.utils.gmm import GMM

# the number of components of the gmm
K = 4


# the density model (so far only "gmm") fit on the points x (nan =
# missing, the likelihood leaves those features out), saved as path and
# loaded from there if it was fit before, then frozen
def fit_density(model, x, path, device):
    n_features = x.shape[1]
    torch.manual_seed(0)
    if model == "gmm":
        # the means start at the first K points, their gaps filled with the means
        first = torch.where(torch.isnan(x[:K]), x.nanmean(dim=0), x[:K])
        px = GMM(device, K=K, means_init=first, n_features=n_features).to(device)

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
# an intercept and with the l2 penalty C (sklearn's default 1), as a tensor w
def logistic_regression(x, y, C=1.0):
    classifier = LogisticRegression(C=C, fit_intercept=False, max_iter=1000).fit(x.cpu(), y.cpu())
    return torch.tensor(classifier.coef_[0], dtype=torch.float32, device=x.device)


# the fraction of the predictions p(y=1) that are on the right side of 0.5
def accuracy(p, y):
    return ((p > 0.5).float() == y).float().mean().item()


# the errors of estimate.py. every method is a function method(first, w, x_partial,
# **settings) (pclr.py, side/utils/baselines.py), first is the density (the means for
# mean imputation), the settings are its counts (n_freq, n_samples, n_outer, ...).
# on a batch of test points (B, D) it returns a tensor (B,)

# the predictions of a method on the points, without gradients, as a numpy array
def predictions(method, first, w, points, **settings):
    with torch.no_grad():
        return method(first, w, points, **settings).cpu().numpy()

# mean kl divergence KL(p || p_hat) between the bernoulli of the exact value p and
# of the estimate p_hat over the test points: the extra log-loss of predicting
# with p_hat instead of p. unlike the rmse it also sees errors close to 0 and 1.
# both are clamped to [eps, 1 - eps] as in the loss of train.py, since the
# quadrature can return values slightly outside [0, 1]
KL_EPS = 1e-6

def kl(estimates, truths):
    # in float64: the estimates are float32, and 1 - p_hat in float32 is too coarse close to 1
    p     = np.clip(np.asarray(truths, dtype=np.float64), KL_EPS, 1 - KL_EPS)
    p_hat = np.clip(np.asarray(estimates, dtype=np.float64), KL_EPS, 1 - KL_EPS)
    return float(np.mean(p * np.log(p / p_hat) + (1 - p) * np.log((1 - p) / (1 - p_hat))))

# kl and the wall clock time of one batched call over all points, and the
# estimates of that call. the truths belong to the first len(truths) points,
# the error is over those, the time is for all of them. after one call to warm
# up (gpu kernels, cached nodes), the call is repeated `repeats` times and the
# time is the median. the predictions are numpy arrays on the cpu, so the gpu has
# finished when the clock stops
def timed_kl(method, first, w, points, truths, repeats=3, **settings):
    predictions(method, first, w, points, **settings)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        estimates = predictions(method, first, w, points, **settings)
        times.append(time.perf_counter() - t0)
        if len(times) == 1:
            first_estimates = estimates
            error = kl(estimates[:len(truths)], truths)
    return error, float(np.median(times)), first_estimates


# the comparison of estimate.py: scores[(n_features, missing_rate)][estimator]
# holds the kl and the time of every estimator in every setting. one figure,
# against the number of features, one column per missing rate: the kl on
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


# ── the training comparison of train.py ───────────────────────────────────────

# the scores of one trained model: the log-loss and the accuracy on the complete test points
# (sigmoid(w^T x)) and on the incomplete ones (its own predictions p_missing), and the
# relative error |w - w_true| / |w_true| of its weights. predictions clamped to [eps, 1 - eps]
def training_scores(w, p_missing, x_test, y_test, w_true, eps=1e-6):
    p_complete = torch.sigmoid(x_test @ w)
    return {
        "log_loss_complete": F.binary_cross_entropy(p_complete.clamp(eps, 1 - eps), y_test).item(),
        "accuracy_complete": accuracy(p_complete, y_test),
        "log_loss_incomplete": F.binary_cross_entropy(p_missing.clamp(eps, 1 - eps), y_test).item(),
        "accuracy_incomplete": accuracy(p_missing, y_test),
        "w_error": ((w - w_true).norm() / w_true.norm()).item(),
    }

# the scores of every model in one setting as a table
def print_training_scores(n_features, missing_rate, scores):
    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'':<22}{'complete test':>22}{'incomplete test':>22}{'w error':>10}")
    for name, score in scores.items():
        print(f"    {name:<22}{score['log_loss_complete']:>11.4f}{score['accuracy_complete']:>11.4f}"
              f"{score['log_loss_incomplete']:>11.4f}{score['accuracy_incomplete']:>11.4f}{score['w_error']:>10.3f}")

# the comparison of train.py: scores[(n_features, rate)][model][score] of every setting,
# saved as cache/<folder>/training_scores<suffix>.json, and drawn against the number of
# features: on complete and on incomplete test points (training_<test>_test<suffix>, a
# column per missing rate, log-loss and accuracy), and the error of w (weights<suffix>)
def compare_training(scores, features, missing_rates, folder, suffix):
    os.makedirs(f"cache/{folder}", exist_ok=True)
    with open(f"cache/{folder}/training_scores{suffix}.json", "w") as file:
        json.dump({f"{n}_{rate}": scores[(n, rate)] for n, rate in scores}, file, indent=1)

    names = list(scores[(features[0], missing_rates[0])])
    for test in ("complete", "incomplete"):
        results = {}
        for score in ("log_loss", "accuracy"):
            results[score] = {rate: {name: [scores[(n, rate)][name][f"{score}_{test}"] for n in features]
                                     for name in names} for rate in missing_rates}
        save_figure(plot_training_comparison, folder, f"training_{test}_test{suffix}", features, missing_rates, results)

    errors = {rate: {name: [scores[(n, rate)][name]["w_error"] for n in features] for name in names}
              for rate in missing_rates}
    save_figure(plot_weight_error, folder, f"weights{suffix}", features, missing_rates, errors)

# the convergence of train.py: errors[model] = the relative error of w per number of
# training points, saved as cache/<folder>/convergence_errors<suffix>.json and drawn
def compare_convergence(n_trains, errors, n_features, missing_rate, folder, suffix):
    os.makedirs(f"cache/{folder}", exist_ok=True)
    with open(f"cache/{folder}/convergence_errors{suffix}.json", "w") as file:
        json.dump({"n_trains": n_trains, "errors": errors}, file, indent=1)
    save_figure(plot_convergence, folder, f"convergence{suffix}", n_trains, errors, n_features, missing_rate)
