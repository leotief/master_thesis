# helpers of the experiment scripts (train.py, estimate.py, real_data.py) that are not part
# of their story: fitting the density model, the logistic regression, the
# accuracy, the errors of the estimators, and the comparisons of all settings

import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from utils.data import loader
from utils.plots import plot_learning_curve, plot_training_comparison, save_figure
from utils.gmm import GMM

# the density model (so far only "gmm", with K components) fit on the points x (nan =
# missing, the likelihood leaves those features out), saved as path and loaded from there
# if it was fit before (no cache if path is None), then frozen
def fit_density(model, x, path, device, K):
    torch.manual_seed(0)
    if model == "gmm":
        px = GMM(x, K, device)

    if path is not None and os.path.exists(path):
        px.load_state_dict(torch.load(path, map_location=device))
    else:
        px.fit(loader(x))
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(px.state_dict(), path)

    for parameter in px.parameters():
        parameter.requires_grad_(False)
    return px


# sklearn's logistic regression on the complete points x with labels y, without
# an intercept and with the l2 penalty C (sklearn's default 1), as a tensor w
def logistic_regression(x, y, C=1.0):
    classifier = LogisticRegression(C=C, fit_intercept=False, max_iter=1000).fit(x.cpu(), y.cpu())
    return torch.tensor(classifier.coef_[0], dtype=torch.float32, device=x.device)


# the fraction of the predictions p(y=1) that are on the right side of 0.5
def accuracy(p, y):
    return ((p > 0.5).float() == y).float().mean().item()


# the errors of estimate.py. every method is a function method(first, w, x_partial,
# **settings) (pclr.py, utils/baselines.py), first is the density (the means for
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
# estimates. the truths belong to the first len(truths) points, the error is over
# those, the time is for all of them. the first call also warms up (gpu kernels,
# cached nodes) and gives the estimates, then the call is repeated `repeats` times
# and the time is the median. the predictions are numpy arrays on the cpu, so the
# gpu has finished when the clock stops
def timed_kl(method, first, w, points, truths, repeats=3, **settings):
    estimates = predictions(method, first, w, points, **settings)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predictions(method, first, w, points, **settings)
        times.append(time.perf_counter() - t0)
    return kl(estimates[:len(truths)], truths), float(np.median(times)), estimates


# every method of estimate.py on all points: its kl against the truths (the first
# len(truths) points) and the time of the whole call, printed with its estimate of the
# first point. methods: name -> (method, first, settings). returns name -> (kl, time)
def score_methods(methods, w, points, truths, repeats):
    scores = {}
    for name, (method, first, settings) in methods.items():
        error, seconds, estimates = timed_kl(method, first, w, points, truths, repeats=repeats, **settings)
        scores[name] = (error, seconds)
        print(f"    {str(name):<24}{estimates[0]:.6f}        kl {error:.1e}        time {seconds:.3f} s")
    return scores


# the figures of estimate.py: scores[(n_features, missing_rate)][key] = (kl, time) of every
# key (an estimator, or a number of cfi frequencies) in every setting, drawn by plot
# (plot_estimate_comparison or plot_cfi_frequencies) against the number of features, one
# column per missing rate: the kl on top, the time for all n_points test points below, as
# figures/<folder>/<name> and in the thesis and the paper. the scores are saved as
# cache/<folder>/<name>.json
def compare_estimates(plot, scores, features, missing_rates, n_points, folder, name):
    os.makedirs(f"cache/{folder}", exist_ok=True)
    with open(f"cache/{folder}/{name}.json", "w") as file:
        json.dump({f"{n}_{rate}": {str(key): value for key, value in scores[(n, rate)].items()} for n, rate in scores},
                  file, indent=1)
    errors, times = {}, {}
    for rate in missing_rates:
        errors[rate], times[rate] = {}, {}
        for key in scores[(features[0], rate)]:
            errors[rate][key] = [scores[(n, rate)][key][0] for n in features]
            times[rate][key] = [scores[(n, rate)][key][1] for n in features]
    save_figure(plot, folder, name, features, missing_rates, errors, times, n_points)


# ── the training comparison of train.py ───────────────────────────────────────

# the log-loss and the accuracy of the predictions p(y=1) of the points with labels y,
# clamped to [eps, 1 - eps]
def prediction_scores(p, y, eps=1e-6):
    return {"log_loss": F.binary_cross_entropy(p.clamp(eps, 1 - eps), y).item(), "accuracy": accuracy(p, y)}

# the scores of one trained model: the log-loss and the accuracy on the complete test points
# (sigmoid(w^T x)) and on the incomplete ones (its own predictions p_missing)
def training_scores(w, p_missing, x_test, y_test):
    scores = {}
    for test, p in (("complete", torch.sigmoid(x_test @ w)), ("incomplete", p_missing)):
        scores.update({f"{score}_{test}": value for score, value in prediction_scores(p, y_test).items()})
    return scores

# the scores of every model in one setting as a table
def print_training_scores(n_features, missing_rate, scores):
    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'':<22}{'complete test':>22}{'incomplete test':>22}")
    for name, score in scores.items():
        print(f"    {name:<22}{score['log_loss_complete']:>11.4f}{score['accuracy_complete']:>11.4f}"
              f"{score['log_loss_incomplete']:>11.4f}{score['accuracy_incomplete']:>11.4f}")

# the comparison of train.py: scores[(n_features, rate)][model][score] of every setting,
# saved as cache/<folder>/<prefix>scores<suffix>.json, and drawn against the number of
# features, on complete and on incomplete test points (<prefix><test>_test<suffix>, a
# column per missing rate, log-loss and accuracy)
def compare_training(scores, features, missing_rates, folder, prefix, suffix, latex=True):
    os.makedirs(f"cache/{folder}", exist_ok=True)
    with open(f"cache/{folder}/{prefix}scores{suffix}.json", "w") as file:
        json.dump({f"{n}_{rate}": scores[(n, rate)] for n, rate in scores}, file, indent=1)

    # the lr without missing values is only the reference on complete test points
    all_names = list(scores[(features[0], missing_rates[0])])
    for test in ("complete", "incomplete"):
        names = all_names if test == "complete" else [name for name in all_names if name != "Complete data"]
        results = {}
        for score in ("log_loss", "accuracy"):
            results[score] = {rate: {name: [scores[(n, rate)][name][f"{score}_{test}"] for n in features]
                                     for name in names} for rate in missing_rates}
        save_figure(plot_training_comparison, folder, f"{prefix}{test}_test{suffix}", features, missing_rates, results,
                    latex=latex)

# the learning curve of train.py: curve[model][score] = the log-loss / accuracy on the
# incomplete test points per number of training points, saved as
# cache/<folder>/<prefix>learning_curve<suffix>.json and drawn as <prefix>learning_curve<suffix>
def compare_learning_curve(n_trains, curve, n_features, missing_rate, folder, prefix, suffix, latex=True):
    os.makedirs(f"cache/{folder}", exist_ok=True)
    with open(f"cache/{folder}/{prefix}learning_curve{suffix}.json", "w") as file:
        json.dump({"n_trains": n_trains, "curve": curve}, file, indent=1)
    # incomplete test points only, so without the lr without missing values
    curve = {name: values for name, values in curve.items() if name != "Complete data"}
    save_figure(plot_learning_curve, folder, f"{prefix}learning_curve{suffix}", n_trains, curve, n_features, missing_rate, latex=latex)


# ── the real-data table of real_data.py ───────────────────────────────────────

# the tables of real_data.py: results[(dataset, missing_rate)][model][score] = one value per
# seed. for every missing rate and score (log_loss, accuracy) one table, a row per dataset,
# a column per model (headers: model -> short column name), mean ± std over the seeds, the
# best model per row in bold (not counting the reference). printed, and written as a latex
# tabular (booktabs) to <folder>/<prefix>_<score>_<percent missing>.tex
def real_data_tables(results, datasets, missing_rates, headers, reference, folder, prefix="real_data"):
    os.makedirs(folder, exist_ok=True)
    plain = [header.replace("\\ ", " ") for header in headers.values()]
    for rate, (score, best) in itertools.product(missing_rates, (("log_loss", min), ("accuracy", max))):
        print(f"\n{score.replace('_', '-')}, {rate:.0%} missing (mean ± std over the seeds)")
        print(f"    {'':<14}" + "".join(f"{header:>18}" for header in plain))
        rows = []
        for dataset in datasets:
            values = results[(dataset, rate)]
            means = {name: float(np.mean(values[name][score])) for name in headers}
            stds = {name: float(np.std(values[name][score])) for name in headers}
            winner = best((name for name in headers if name != reference), key=lambda name: means[name])
            print(f"    {dataset:<14}" + "".join(f"{means[n]:>11.4f} ± {stds[n]:.3f}" for n in headers))
            cells = []
            for name in headers:
                cell = f"{means[name]:.3f}" + r"{\scriptsize$\pm$" + f"{stds[name]:.3f}" + "}"
                cells.append(r"\textbf{" + cell + "}" if name == winner else cell)
            rows.append(f"{dataset} & " + " & ".join(cells) + r" \\")
        lines = [r"\begin{tabular}{l" + "c" * len(headers) + "}", r"\toprule",
                 "Dataset & " + " & ".join(headers.values()) + r" \\", r"\midrule", *rows,
                 r"\bottomrule", r"\end{tabular}"]
        with open(f"{folder}/{prefix}_{score}_{round(100 * rate)}.tex", "w") as file:
            file.write("\n".join(lines) + "\n")
