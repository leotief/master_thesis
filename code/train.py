# training a logistic regression when the training points have missing
# features
#
# the story, for every setting (number of features, missing rate):
#   setup       the training and test points with some features hidden, and a
#               density model p(x) fit on the incomplete training points (its
#               likelihood leaves the missing features out exactly, so this
#               needs no pcpg). then the models, every model is a set of
#               training points and a prediction
#   training    every model is trained the same way, adam on minibatches of the
#               binary cross-entropy of its predictions plus the l2 penalty of
#               sklearn's LogisticRegression (C = 1). no intercept, as in
#               estimate.py
#   evaluation  how close every model gets to the optimum of the exact
#               training objective, and its accuracy on the test points
# and at the end the comparison of all settings, as three figures.
#
# the models differ only in the training points they see and in how they
# predict:
#   Complete data    sigmoid(w^T x) on the complete training points. an oracle,
#                    this data is not available when features are missing
#   Mean imputation  sigmoid(w^T x) on the training points with the gaps filled
#                    by the means
#   MC, PCPG, Exact  the expected prediction p(y=1 | x_obs) = E[sigmoid(w^T x) | x_obs],
#                    the average over the missing features under p(x_mis | x_obs),
#                    on the incomplete training points. its loss is the marginal
#                    likelihood -log p(y | x_obs), and the gradient flows through
#                    the estimate of the expectation:
#                      MC     BUDGET samples of the missing features. by jensen
#                             -log of the average is too large on average, by
#                             about var / (2 BUDGET p^2), which shrinks w
#                      PCPG   deterministic pcpg, N_OUTER x N_INNER nodes, from
#                             the model's characteristic function
#                      Exact  through the mixture of z (model.z_mixture),
#                             gauss-hermite per component. only for the gmm and
#                             small pcs, the target of mc and pcpg
# the helpers are in experiment_utils.py

import itertools

import torch
import torch.nn.functional as F

from experiment_utils import accuracy, compare, fit_density, logistic_regression, optimum
from pcpg import PCPG
from side.utils.baselines import MCBaseline, MixtureGaussHermite
from side.utils.data import load_data

# ── config ────────────────────────────────────────────────────────────────────

# the density model, "gmm", "pc" or "spflow"
MODEL = "gmm"

# the settings: every number of features (INFORMATIVE of them informative, the
# others noise) with every fraction of missing features per point. SEED fixes
# which features are hidden, the minibatches and mc's samples
FEATURES      = [5, 10, 20, 50, 100]
MISSING_RATES = [0.3, 0.5, 0.8]
INFORMATIVE   = 0.5
SEED          = 0

# pcpg's nodes, mc gets the same budget. gauss-hermite nodes per mixture
# component of the exact expectation
N_OUTER, N_INNER = 10, 32
BUDGET           = N_OUTER * N_INNER
EXACT_NODES      = 64

# the l2 penalty and adam
C      = 1.0
EPOCHS = 100
BATCH  = 1024
LR     = 0.01
# predictions are clamped to [EPS, 1 - EPS] in the loss, the quadrature error
# of pcpg can put them slightly outside [0, 1]
EPS    = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# the prediction of a logistic regression on complete points
def sigmoid_prediction(w, x):
    return torch.sigmoid(x @ w)


# the training objective: the mean binary cross-entropy of the predictions of
# the points x with labels y, plus the l2 penalty, scaled to one of n_train
# training points
def objective(predict, w, x, y, n_train):
    p = predict(w, x).to(y.dtype).clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(p, y) + w.square().sum() / (2 * C * n_train)


# the scores of every model in every setting, scores[setting][model]
scores = {}

for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):

    # ── setup ─────────────────────────────────────────────────────────────────

    # the training and test points: complete, and with missing_rate of the
    # features of every point hidden (nan)
    x_train, x_missing, y_train, x_test, x_test_missing, y_test = load_data(
        n_features, missing_rate, INFORMATIVE, SEED, device)
    n_train = len(x_train)

    # the training points with the gaps filled by the means, for mean imputation
    x_filled = torch.where(torch.isnan(x_missing), x_missing.nanmean(dim=0), x_missing)

    # the density model p(x), fit on the incomplete training points
    px = fit_density(MODEL, x_missing, f"cache/train/{MODEL}_{n_features}features_{missing_rate}missing.pt", device)

    # the three estimators of the expected prediction E[sigmoid(w^T x) | x_obs].
    # their estimate_diff(w, x) returns it for the points x as a tensor, with
    # gradients through w
    pcpg  = PCPG(px, n_outer=N_OUTER, n_inner=N_INNER)
    mc    = MCBaseline(px, n_samples=BUDGET)
    exact = MixtureGaussHermite(px, n_nodes=EXACT_NODES)

    # the models: the training points every model sees, and its prediction
    models = {
        "Complete data":   (x_train,   sigmoid_prediction),
        "Mean imputation": (x_filled,  sigmoid_prediction),
        "MC":              (x_missing, mc.estimate_diff),
        "PCPG":            (x_missing, pcpg.estimate_diff),
        "Exact":           (x_missing, exact.estimate_diff),
    }

    # ── training ──────────────────────────────────────────────────────────────

    # every model from w = 0, by adam on minibatches of its training points
    weights = {}
    for name, (x, predict) in models.items():
        torch.manual_seed(SEED)
        w = torch.zeros(n_features, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=LR)
        for epoch in range(EPOCHS):
            for batch in torch.randperm(n_train, device=device).split(BATCH):
                optimizer.zero_grad()
                loss = objective(predict, w, x[batch], y_train[batch], n_train)
                loss.backward()
                optimizer.step()
        weights[name] = w.detach()

    # the two logistic regressions on complete points also the normal way, by
    # sklearn (the same objective, solved exactly). the comparison uses these,
    # and compares them with the ones trained by adam above
    weights["Complete data, sklearn"] = logistic_regression(x_train, y_train, C)
    weights["Mean imputation, sklearn"] = logistic_regression(x_filled, y_train, C)

    # ── evaluation ────────────────────────────────────────────────────────────

    # the exact training objective J(w), and its optimum w* by full batch l-bfgs
    w_star = optimum(objective, exact.estimate_diff, x_missing, y_train, n_train, device)
    best = objective(exact.estimate_diff, w_star, x_missing, y_train, n_train).item()

    # for every model: the objective gap J(w) - J(w*), how well it minimises
    # the exact objective; the test accuracy on the incomplete test points,
    # every model with the exact expected prediction, so they only differ in w;
    # and the test accuracy of sigmoid(w^T x) on the complete test points, how
    # good w is as a classifier
    setting = (n_features, missing_rate)
    scores[setting] = {}
    with torch.no_grad():
        for name, w in weights.items():
            scores[setting][name] = {
                "gap": objective(exact.estimate_diff, w, x_missing, y_train, n_train).item() - best,
                "accuracy_missing": accuracy(exact.estimate_diff(w, x_test_missing), y_test),
                "accuracy_complete": accuracy(sigmoid_prediction(w, x_test), y_test),
            }

    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'':<26}{'objective gap':>15}{'acc. missing':>14}{'acc. complete':>15}")
    for name, score in scores[setting].items():
        print(f"    {name:<26}{score['gap']:>15.1e}{score['accuracy_missing']:>14.4f}"
              f"{score['accuracy_complete']:>15.4f}")

# ── comparison ────────────────────────────────────────────────────────────────

# all settings against the number of features, one column per missing rate:
# the objective gap with exact as the target, the test accuracies with complete
# data (sklearn) as the target, and complete data and mean imputation by
# sklearn against adam. in figures/train/<MODEL>/ and the thesis
compare(scores, FEATURES, MISSING_RATES, f"train/{MODEL}")
