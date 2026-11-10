# the training comparison: how well pclr and the usual ways to train a logistic regression
# do when the training points have missing features. on data where logistic regression is
# the right model (side/utils/data.py), for every number of features and missing rate:
# every model is trained on the incomplete training points, all by l-bfgs until convergence,
# and evaluated twice by log-loss and accuracy:
#   complete test    on complete test points, sigmoid(w^T x), so only the training differs
#   incomplete test  on incomplete test points, every model with its own prediction rule
# and its learned w is compared with the true one. saved as json and drawn. then the
# convergence: the error of w against the number of training points, at one setting.
# the baselines are in side/utils/baselines.py, the helpers in experiment_utils.py

import copy
import itertools

import torch

from experiment_utils import compare_convergence, compare_training, fit_density, print_training_scores, training_scores
from pclr import PCLR
from side.utils.baselines import CompleteData, ConditionalMean, MeanImputation
from side.utils.data import load_lr_data

# ── config ────────────────────────────────────────────────────────────────────

# the density model, so far only the gmm has everything cfi needs
MODEL = "gmm"

# the numbers of features (powers of 2, evenly spaced on a log axis), the fractions of missing features
# per point, and the seed of the data and the hidden features
FEATURES      = [2 ** k for k in range(2, 9)]
MISSING_RATES = [0.3, 0.5, 0.8]
SEED          = 0

# pclr: cfi's frequencies, the weight of the label term against the density term (1 is
# plain maximum likelihood of the joint model), and the iterations of l-bfgs
N_FREQ   = 64
LAMBDA   = 1.0
MAX_ITER = 1000

# the convergence: its number of features and missing rate, and the numbers of training
# points (powers of 2, evenly spaced on a log axis), nested subsets of one dataset with one w_true
CONVERGENCE_FEATURES = 100
CONVERGENCE_RATE     = 0.5
N_TRAINS             = [2 ** k for k in range(10, 16)]

SUFFIX = "_" + str(LAMBDA)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── comparison ────────────────────────────────────────────────────────────────

# scores[(n_features, rate)][model][score], the scores log_loss / accuracy on complete and
# on incomplete test points, and the relative error |w - w_true| / |w_true|
scores = {}

for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):

    # the data, and the density fit on the incomplete training points alone
    x_train, x_missing, y_train, x_test, x_test_missing, y_test, w_true = load_lr_data(n_features, missing_rate, SEED, device)
    cache = f"cache/train/{MODEL}/lr_data_{n_features}features_{missing_rate}missing_seed{SEED}"
    px = fit_density(MODEL, x_missing, f"{cache}.pt", device)

    # the models. pclr starts from the fitted density, on its own copy, since it trains the
    # density it gets (the fixed-density variant only trains w)
    models = {
        "Complete data":      CompleteData(px, n_freq=N_FREQ),
        "Mean imputation":    MeanImputation(),
        "Conditional mean":   ConditionalMean(px),
        "PCLR fixed density": PCLR(copy.deepcopy(px), n_features, n_freq=N_FREQ, lam=LAMBDA, train_density=False, max_iter=MAX_ITER),
        "PCLR":               PCLR(copy.deepcopy(px), n_features, n_freq=N_FREQ, lam=LAMBDA, train_density=True, max_iter=MAX_ITER),
    }

    # every model trained, only the upper bound on the complete training points
    for name, model in models.items():
        if name == "Complete data":
            model.fit(x_train, y_train)
        else:
            model.fit(x_missing, y_train)

    # every model's scores on complete and on incomplete test points (its own prediction
    # rule), and of its weights
    scores[(n_features, missing_rate)] = {}
    with torch.no_grad():
        for name, model in models.items():
            scores[(n_features, missing_rate)][name] = training_scores(model.w.detach(), model.predict(x_test_missing), x_test, y_test, w_true)
    print_training_scores(n_features, missing_rate, scores[(n_features, missing_rate)])

compare_training(scores, FEATURES, MISSING_RATES, f"train/{MODEL}", SUFFIX)

# ── convergence ───────────────────────────────────────────────────────────────

# the largest dataset once, every number of training points its first n points, so that
# w_true is the same for all. errors[model] = one relative error of w per number of points
x_train_all, x_missing_all, y_train_all, x_test, x_test_missing, y_test, w_true = load_lr_data(CONVERGENCE_FEATURES, CONVERGENCE_RATE, SEED, device, n_train=N_TRAINS[-1], n_test=2000)
errors = {}
print(f"\nconvergence, {CONVERGENCE_FEATURES} features, {CONVERGENCE_RATE:.0%} missing")
for n_train in N_TRAINS:
    x_train, x_missing, y_train = x_train_all[:n_train], x_missing_all[:n_train], y_train_all[:n_train]
    cache = f"cache/train/{MODEL}/lr_data_convergence_{CONVERGENCE_FEATURES}features_{CONVERGENCE_RATE}missing_{n_train}points_seed{SEED}"
    px = fit_density(MODEL, x_missing, f"{cache}.pt", device)

    models = {
        "Complete data":    CompleteData(px, n_freq=N_FREQ),
        "Mean imputation":  MeanImputation(),
        "Conditional mean": ConditionalMean(px),
        "PCLR":             PCLR(copy.deepcopy(px), CONVERGENCE_FEATURES, n_freq=N_FREQ, lam=LAMBDA, train_density=True, max_iter=MAX_ITER),
    }
    for name, model in models.items():
        if name == "Complete data":
            model.fit(x_train, y_train)
        else:
            model.fit(x_missing, y_train)
        w = model.w.detach()
        errors.setdefault(name, []).append(((w - w_true).norm() / w_true.norm()).item())
    print(f"    {n_train:>7} points   " + "   ".join(f"{name} {errors[name][-1]:.3f}" for name in models))

compare_convergence(N_TRAINS, errors, CONVERGENCE_FEATURES, CONVERGENCE_RATE, f"train/{MODEL}", SUFFIX)
