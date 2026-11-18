# the training comparison: how well pclr and the usual ways to train a logistic regression
# do when the training points have missing features, on sklearn's make_classification
# (utils/data.py). every model is trained on the incomplete training points,
# all by l-bfgs until convergence, and scored by log-loss and accuracy on
#   complete test    complete test points, sigmoid(w^T x), so only the training differs
#   incomplete test  incomplete test points, every model with its own prediction rule
# the lr without missing values is the reference. two experiments:
#   comparison       against the number of features, a column per missing rate
#   learning curve   against the number of training points, at one setting
# the models are in utils/baselines.py and pclr.py, the helpers in utils/experiments.py

import itertools

import torch

from pclr import PCLR
from utils.baselines import CompleteData, ConditionalMeanImputation, MeanImputation
from utils.data import INFORMATIVE, REDUNDANT, load_data
from utils.experiments import compare_learning_curve, compare_training, fit_density, print_training_scores, training_scores

# ── config ────────────────────────────────────────────────────────────────────

# the density model, so far only the gmm has everything cfi needs, and its number of components
MODEL = "gmm"
K     = 4

# the numbers of features (powers of 2, evenly spaced on a log axis; the split into
# informative, redundant and noise features, INFORMATIVE and REDUNDANT, is that of every
# experiment, in utils/data.py), the probabilities of every entry to be missing, the seed of
# the hidden features, and whether the features are standardized
FEATURES      = [2 ** k for k in range(2, 9)]
MISSING_RATES = [0.1, 0.3, 0.5]
SEED          = 0
NORM          = True

# pclr: cfi's frequencies and the iterations of l-bfgs
N_FREQ   = 64
MAX_ITER = 1000

# the learning curve: its number of features and missing rate, the numbers of training
# points (powers of 2, evenly spaced on a log axis, nested subsets of one dataset) and the
# test points
CURVE_FEATURES = 100
CURVE_RATE     = 0.3
N_TRAINS       = [2 ** k for k in range(8, 15)]
CURVE_TEST     = 5000

# the end of the names of the figures, the json and the cached densities: standardized
# features are the default and get no suffix, raw ones _raw
NORM_SUFFIX = "" if NORM else "_raw"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── models ────────────────────────────────────────────────────────────────────

# the main comparison, every model trained on every setting (comment one out to skip it):
# the reference (the lr without missing values, trained and tested on complete points), the
# imputations, and pclr (the density and w jointly, the density starting untrained)
MODELS = {
    "Complete data":    CompleteData(),
    "Mean imputation":  MeanImputation(),
    "Conditional mean imputation": ConditionalMeanImputation(),
    "PCLR":             PCLR(n_freq=N_FREQ, max_iter=MAX_ITER),
}

# the variants of pclr: the density starting untrained (the pclr above), starting from the
# fitted density, and the fitted density kept fixed (only w trained)
PCLR_VARIANTS = {
    "PCLR":               PCLR(n_freq=N_FREQ, max_iter=MAX_ITER),
    "PCLR fitted start":  PCLR(n_freq=N_FREQ, max_iter=MAX_ITER, start="fitted"),
    "PCLR fixed density": PCLR(n_freq=N_FREQ, max_iter=MAX_ITER, train_density=False),
}

# ── experiments ───────────────────────────────────────────────────────────────

# the comparison: for every number of features and missing rate, the data, the density
# fit on the incomplete training points, and every model of models trained and scored.
# saved and drawn against the number of features, the names starting with prefix
def comparison(models, prefix):
    scores = {}
    for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):
        setting = (n_features, missing_rate)

        # the data, and the density fit on the incomplete training points alone
        x_train, x_missing, y_train, x_test, x_test_missing, y_test = load_data(n_features, missing_rate, INFORMATIVE, SEED, device, standardize=NORM, redundant=REDUNDANT)
        cache = f"cache/train/{MODEL}/K{K}_{n_features}features_{INFORMATIVE}informative_{REDUNDANT}redundant_{missing_rate}missing_seed{SEED}{NORM_SUFFIX}.pt"
        px = fit_density(MODEL, x_missing, cache, device, K)

        # every model trained on the incomplete training points, scored on complete test
        # points and on incomplete ones with its own prediction rule. the reference is
        # trained and tested on the complete points
        scores[setting] = {}
        for name, model in models.items():
            if isinstance(model, CompleteData):
                model.fit(x_train, y_train)
                x_eval = x_test
            else:
                model.fit(x_missing, y_train, px)
                x_eval = x_test_missing
            with torch.no_grad():
                scores[setting][name] = training_scores(model.w.detach(), model.predict(x_eval), x_test, y_test)
        print_training_scores(n_features, missing_rate, scores[setting])

    compare_training(scores, FEATURES, MISSING_RATES, f"train/{MODEL}", prefix, NORM_SUFFIX)


# the learning curve: the largest dataset once, every number of training points its first
# n points with its own density, every model of models trained and scored on the same
# incomplete test points. saved and drawn against the number of training points, the
# names starting with prefix
def learning_curve(models, prefix):
    n = N_TRAINS[-1] + CURVE_TEST
    x_train_all, x_missing_all, y_train_all, x_test, x_test_missing, y_test = load_data(
        CURVE_FEATURES, CURVE_RATE, INFORMATIVE, SEED, device, standardize=NORM, samples=n, test_size=CURVE_TEST / n, redundant=REDUNDANT)
    curve = {name: {"log_loss": [], "accuracy": []} for name in models}
    print(f"\nlearning curve, {CURVE_FEATURES} features, {CURVE_RATE:.0%} missing")
    for n_train in N_TRAINS:

        # the first n_train training points, and the density fit on them
        x_train, x_missing, y_train = x_train_all[:n_train], x_missing_all[:n_train], y_train_all[:n_train]
        cache = f"cache/train/{MODEL}/K{K}_curve_{CURVE_FEATURES}features_{INFORMATIVE}informative_{REDUNDANT}redundant_{CURVE_RATE}missing_{n_train}points_seed{SEED}{NORM_SUFFIX}.pt"
        px = fit_density(MODEL, x_missing, cache, device, K)

        # every model, as in the comparison, scored on the incomplete test points
        scores = {}
        for name, model in models.items():
            if isinstance(model, CompleteData):
                model.fit(x_train, y_train)
                x_eval = x_test
            else:
                model.fit(x_missing, y_train, px)
                x_eval = x_test_missing
            with torch.no_grad():
                scores[name] = training_scores(model.w.detach(), model.predict(x_eval), x_test, y_test)
        for name, score in scores.items():
            curve[name]["log_loss"].append(score["log_loss_incomplete"])
            curve[name]["accuracy"].append(score["accuracy_incomplete"])
        print(f"    {n_train:>7} points   " + "   ".join(f"{name} {score['log_loss_incomplete']:.4f}" for name, score in scores.items()))

    compare_learning_curve(N_TRAINS, curve, CURVE_FEATURES, CURVE_RATE, f"train/{MODEL}", prefix, NORM_SUFFIX)

# ── run ───────────────────────────────────────────────────────────────────────

# the pclr variants, then the main comparison (comment a line out to skip it)
comparison(PCLR_VARIANTS, "pclr_variants_")
learning_curve(PCLR_VARIANTS, "pclr_variants_")

comparison(MODELS, "")
learning_curve(MODELS, "")
