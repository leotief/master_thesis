# training a logistic regression when the training points have missing
# features
#
# the story, for every setting (number of features, missing rate):
#   setup       the training and test points with some features hidden, and a
#               density model p(x) fit on the incomplete training points (its
#               likelihood leaves the missing features out exactly, so this
#               needs no pcpg)
#   training    mc, pcpg and exact are trained the same way, adam on the same
#               minibatches of the binary cross-entropy of their predictions
#               plus the l2 penalty of sklearn's LogisticRegression (C). complete
#               data and mean imputation are fit by sklearn. no intercept, as
#               in estimate.py
#   evaluation  two questions. does training with mc or pcpg give the classifier
#               of exact training: the mean difference of their test predictions
#               to those of exact. is training on the marginal likelihood better
#               than the usual ways: accuracy on the incomplete test points, with
#               three prediction rules for every model (the exact expected
#               prediction, mean imputation, its own rule), and on the complete
#               test points
# and at the end the comparison of all settings, as figures. one run is one
# configuration (NORM), the figures with both configurations are drawn once
# both have run.
#
# the models:
#   Complete data    logistic regression on the complete training points. an
#                    oracle, this data is not available when features are missing
#   Mean imputation  logistic regression on the training points with the gaps
#                    filled by the means
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
#                             small pcs, the reference of mc and pcpg
# the helpers are in experiment_utils.py

import itertools

import torch
import torch.nn.functional as F

from experiment_utils import accuracy, compare, fit_density, logistic_regression
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

# NORM: standardized features and a strong l2 penalty (C = 1e-3), which keeps
# z = w^T x on a moderate scale, the setup that worked best in estimate.py.
# without it the raw data and sklearn's default C = 1
NORM = True

# pcpg's nodes, mc gets the same budget. gauss-hermite nodes per mixture
# component of the exact expectation
N_OUTER, N_INNER = 20, 60
BUDGET           = N_OUTER * N_INNER
EXACT_NODES      = 64

# the l2 penalty
C = 1e-3 if NORM else 1.0

# the figures are comparison<SUFFIX>_<score>
SUFFIX = "_" + str(INFORMATIVE) + "_" + str(N_OUTER) + "_" + str(N_INNER) + ("_norm" if NORM else "")

# optimizer 
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

# the scores of every model in every setting, scores[setting][model][score]
scores = {}

for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):

    # ── setup ─────────────────────────────────────────────────────────────────

    # the training and test points: complete, and with missing_rate of the features of every point hidden (nan)
    x_train, x_missing, y_train, x_test, x_test_missing, y_test = load_data(n_features, missing_rate, INFORMATIVE, SEED, device, standardize=NORM)
    n_train = len(x_train)

    # the training and test points with the gaps filled by the training means, for mean imputation
    x_filled = torch.where(torch.isnan(x_missing), x_missing.nanmean(dim=0), x_missing)
    x_test_filled = torch.where(torch.isnan(x_test_missing), x_missing.nanmean(dim=0), x_test_missing)

    # the density model p(x), fit on the incomplete training points
    px = fit_density(MODEL, x_missing, f"cache/train/{MODEL}/{n_features}features_{missing_rate}missing_{INFORMATIVE}informative{'_norm' if NORM else ''}.pt", device)

    # the three estimators of the expected prediction E[sigmoid(w^T x) | x_obs].
    # their estimate_diff(w, x) returns it for the points x as a tensor, with gradients through w
    pcpg  = PCPG(px, n_outer=N_OUTER, n_inner=N_INNER)
    mc    = MCBaseline(px, n_samples=BUDGET)
    exact = MixtureGaussHermite(px, n_nodes=EXACT_NODES)

    # the models trained on the marginal likelihood, every one with its estimator
    models = {
        "MC":    mc,
        "PCPG":  pcpg,
        "Exact": exact,
    }

    # ── training ──────────────────────────────────────────────────────────────

    # every model from w = 0, by adam on the same minibatches of the incomplete
    # training points (their own generator, mc's samples don't change them)
    weights = {}
    for name, estimator in models.items():
        torch.manual_seed(SEED)
        order = torch.Generator(device=device).manual_seed(SEED)
        w = torch.zeros(n_features, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=LR)
        for epoch in range(EPOCHS):
            for batch in torch.randperm(n_train, device=device, generator=order).split(BATCH):
                optimizer.zero_grad()
                loss = objective(estimator.estimate_diff, w, x_missing[batch], y_train[batch], n_train)
                loss.backward()
                optimizer.step()
        weights[name] = w.detach()

    # the two logistic regressions on complete points the usual way, by sklearn
    weights["Complete data"] = logistic_regression(x_train, y_train, C)
    weights["Mean imputation"] = logistic_regression(x_filled, y_train, C)

    # ── evaluation ────────────────────────────────────────────────────────────

    # for every model: the mean difference of its exact expected predictions of
    # the incomplete test points to those of exact, the accuracy on these points
    # with three prediction rules (the exact expected prediction, mean
    # imputation, its own rule: mean imputation for complete data and mean
    # imputation, its estimator for mc, pcpg and exact), and the accuracy of
    # sigmoid(w^T x) on the complete test points
    setting = (n_features, missing_rate)
    scores[setting] = {}
    with torch.no_grad():
        p_exact = exact.estimate_diff(weights["Exact"], x_test_missing)
        for name, w in weights.items():
            p = exact.estimate_diff(w, x_test_missing)
            p_mean = sigmoid_prediction(w, x_test_filled)
            if name in models:
                p_own = torch.as_tensor(models[name].estimate(w, x_test_missing), device=device)
            else:
                p_own = p_mean
            scores[setting][name] = {
                "difference": (p - p_exact).abs().mean().item(),
                "accuracy_exact": accuracy(p, y_test),
                "accuracy_mean": accuracy(p_mean, y_test),
                "accuracy_own": accuracy(p_own, y_test),
                "accuracy_complete": accuracy(sigmoid_prediction(w, x_test), y_test),
            }

    print(f"\n{n_features} features, {missing_rate:.0%} missing")
    print(f"    {'':<18}{'difference':>12}{'acc. exact':>12}{'acc. mean':>11}{'acc. own':>10}{'acc. complete':>15}")
    for name, score in scores[setting].items():
        print(f"    {name:<18}{score['difference']:>12.1e}{score['accuracy_exact']:>12.4f}{score['accuracy_mean']:>11.4f}"
              f"{score['accuracy_own']:>10.4f}{score['accuracy_complete']:>15.4f}")

# ── comparison ────────────────────────────────────────────────────────────────

compare(scores, FEATURES, MISSING_RATES, f"train/{MODEL}", SUFFIX)
