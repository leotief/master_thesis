# how well pclr and the usual ways to train a logistic regression do when the training
# points have missing features. every model is trained on the incomplete training points
# and scored on complete test points (only the training differs) and on incomplete ones
# (each model predicts in its own way). the logistic regression without missing values is
# the reference. two experiments: against the number of features, and against the number
# of training points

import itertools

import torch

from pclr import PCLR
from utils.baselines import CompleteData, ConditionalMeanImputation, MeanImputation
from utils.data import INFORMATIVE, REDUNDANT, load_data
from utils.experiments import compare_learning_curve, compare_training, fit_density, print_training_scores, training_scores

# ── config ────────────────────────────────────────────────────────────────────

MODEL = "gmm"                                   # the density model, so far only the gmm has everything cfi needs
K     = 4                                       # its number of components

FEATURES      = [2 ** k for k in range(2, 9)]   # numbers of features, evenly spaced on a log axis
MISSING_RATES = [0.1, 0.3, 0.5]                 # how likely each entry is missing
SEED          = 0                               # which features are hidden
NORM          = True                            # standardize the features

N_FREQ = 64                                     # cfi's frequencies

CURVE_FEATURES = 100                            # the setting of the learning curve
CURVE_RATE     = 0.3
N_TRAINS       = [2 ** k for k in range(8, 15)] # its numbers of training points, each the first points of one dataset
CURVE_TEST     = 5000                           # its test points

NORM_SUFFIX = "" if NORM else "_raw"            # the end of the file names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── models ────────────────────────────────────────────────────────────────────

# the main comparison (comment a model out to skip it)
MODELS = {
    "Complete data":    CompleteData(),
    "Mean imputation":  MeanImputation(),
    "Conditional mean imputation": ConditionalMeanImputation(),
    "PCLR":             PCLR(n_freq=N_FREQ),
}

# pclr with the three ways to handle its density
PCLR_VARIANTS = {
    "PCLR":               PCLR(n_freq=N_FREQ),
    "PCLR fitted start":  PCLR(n_freq=N_FREQ, density_mode="fitted"),
    "PCLR fixed density": PCLR(n_freq=N_FREQ, density_mode="fixed"),
}

# ── experiments ───────────────────────────────────────────────────────────────

# trains and scores every model in every setting, then saves the scores and draws them
# against the number of features. prefix starts the file names
def comparison(models, prefix):
    scores = {}
    for n_features, missing_rate in itertools.product(FEATURES, MISSING_RATES):
        setting = (n_features, missing_rate)

        # the density is fit on the incomplete training points alone, as a model would see them
        x_train, x_missing, y_train, x_test, x_test_missing, y_test = load_data(n_features, missing_rate, INFORMATIVE, SEED, device, standardize=NORM, redundant=REDUNDANT)
        cache = f"cache/train/{MODEL}/K{K}_{n_features}features_{INFORMATIVE}informative_{REDUNDANT}redundant_{missing_rate}missing_seed{SEED}{NORM_SUFFIX}.pt"
        px = fit_density(MODEL, x_missing, cache, device, K)

        # the reference is trained and tested on the complete points, every other model on the incomplete ones
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


# trains and scores every model on more and more training points of one setting, all on
# the same incomplete test points, then saves the scores and draws them. prefix starts
# the file names
def learning_curve(models, prefix):
    n = N_TRAINS[-1] + CURVE_TEST
    x_train_all, x_missing_all, y_train_all, x_test, x_test_missing, y_test = load_data(
        CURVE_FEATURES, CURVE_RATE, INFORMATIVE, SEED, device, standardize=NORM, samples=n, test_size=CURVE_TEST / n, redundant=REDUNDANT)
    curve = {name: {"log_loss": [], "accuracy": []} for name in models}
    print(f"\nlearning curve, {CURVE_FEATURES} features, {CURVE_RATE:.0%} missing")
    for n_train in N_TRAINS:

        # the first training points, with a density fit on them
        x_train, x_missing, y_train = x_train_all[:n_train], x_missing_all[:n_train], y_train_all[:n_train]
        cache = f"cache/train/{MODEL}/K{K}_curve_{CURVE_FEATURES}features_{INFORMATIVE}informative_{REDUNDANT}redundant_{CURVE_RATE}missing_{n_train}points_seed{SEED}{NORM_SUFFIX}.pt"
        px = fit_density(MODEL, x_missing, cache, device, K)

        # as in the comparison
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

# comment a line out to skip it
comparison(PCLR_VARIANTS, "pclr_variants_")
learning_curve(PCLR_VARIANTS, "pclr_variants_")

comparison(MODELS, "")
learning_curve(MODELS, "")
