# helpers of the experiment scripts that are not part of their story: loading
# the data with missing features, fitting the density model, finding the
# optimum of an objective, the accuracy, and the comparison of all settings

import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from side.utils.data import DataManager
from side.utils.plots import plot_train_features, plot_train_sklearn, save_figure
from side.utils.gmm import GMM
from side.utils.pc import PC
from side.utils.spflow_pc import SPFlowPC

# the density models: K gmm components, and the circuit of both pcs (gaussians
# per feature, sum nodes per region, random structures)
K              = 4
PC_CHANNELS    = 4
PC_REPETITIONS = 3
PC_DEPTH       = 2


# a copy of the points x (numpy) with missing_rate of the features of every
# point set to nan, chosen at random for every point
def hide(x, missing_rate, rng):
    x = x.copy()
    n_hidden = int(missing_rate * x.shape[1])
    for row in x:
        row[rng.choice(len(row), n_hidden, replace=False)] = np.nan
    return x


# make_classification with n_features features, informative of them
# informative and the others noise, 8000 training and 2000 test points. the
# points are the same for every seed, which features are hidden is not.
# returns the training points (complete and incomplete), their labels, the test
# points (complete and incomplete) and their labels, as tensors
def load_data(n_features, missing_rate, informative, seed, device):
    dm = DataManager("sk_class", samples=10_000, features=n_features, test_size=0.2, random_state=42,
                     informative=informative)
    x_train, x_test, y_train, y_test = dm.get_data()
    rng = np.random.default_rng(seed)
    x_missing = hide(x_train, missing_rate, rng)
    x_test_missing = hide(x_test, missing_rate, rng)
    arrays = (x_train, x_missing, y_train, x_test, x_test_missing, y_test)
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]


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
        px = PC(device, n_features, channels=PC_CHANNELS, repetitions=PC_REPETITIONS, depth=PC_DEPTH)
    else:
        px = SPFlowPC(device, n_features, channels=PC_CHANNELS, repetitions=PC_REPETITIONS, depth=PC_DEPTH)

    if os.path.exists(path):
        px.load_state_dict(torch.load(path, map_location=device))
    else:
        dataset = torch.utils.data.TensorDataset(x)
        px.fit(torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True))
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


# the comparison of train.py: scores[(n_features, missing_rate)][model] holds
# the scores of every model in every setting. draws them against the number
# of features, one column per missing rate, in figures/<folder>/ and the
# thesis: the objective gap with exact as the target, the test accuracies with
# complete data as the target (exact left out), complete data and mean
# imputation fit by sklearn. and in comparison_sklearn these two by sklearn
# against the same by adam, all three scores
def compare(scores, features, missing_rates, folder):
    results = {}
    for score in ("gap", "accuracy_missing", "accuracy_complete"):
        results[score] = {}
        for rate in missing_rates:
            results[score][rate] = {}
            for name in scores[(features[0], rate)]:
                results[score][rate][name] = [scores[(n, rate)][name][score] for n in features]

    # the main figures: the baselines by sklearn, under their plain names
    main = {"Complete data": "Complete data, sklearn", "Mean imputation": "Mean imputation, sklearn",
            "MC": "MC", "PCPG": "PCPG", "Exact": "Exact"}
    for score, ylabel, log, target in (("gap", "objective gap", True, "Exact"),
                                       ("accuracy_missing", "test accuracy, missing features", False, "Complete data"),
                                       ("accuracy_complete", "test accuracy, complete points", False, "Complete data")):
        panels = {}
        for rate in missing_rates:
            panels[rate] = {}
            for label, name in main.items():
                if not (name == "Exact" and target == "Complete data"):
                    panels[rate][label] = results[score][rate][name]
        save_figure(plot_train_features, folder, f"comparison_{score}",
                    features, missing_rates, panels, ylabel, log, target)

    # sklearn against adam for the two logistic regressions
    names = ["Complete data, sklearn", "Complete data", "Mean imputation, sklearn", "Mean imputation"]
    sklearn = {}
    for score in results:
        sklearn[score] = {}
        for rate in missing_rates:
            sklearn[score][rate] = {}
            for name in names:
                sklearn[score][rate][name] = results[score][rate][name]
    save_figure(plot_train_sklearn, folder, "comparison_sklearn", features, missing_rates, sklearn)
