# the real-data comparison: pclr against the usual ways to train a classifier when the
# training points have missing features, on real datasets (utils/data.py, load_real).
# for every dataset, missing rate and seed (a new split and new hidden features): every
# model of MODELS is trained on the incomplete training points and predicts the incomplete
# test points with its own rule, scored by log-loss and accuracy. the lr without missing
# values is the reference. the result is a table per missing rate and score,
# mean ± std over the seeds, printed and written as latex to paper/tables/.
# the models are in utils/baselines.py and pclr.py, the helpers in utils/experiments.py

import itertools

import torch

from pclr import PCLR
from utils.baselines import CompleteData, ConditionalMeanImputation, GenerativeClassifier, MeanImputation, MICE
from utils.data import load_real
from utils.experiments import fit_density, prediction_scores, real_data_tables

# ── config ────────────────────────────────────────────────────────────────────

# the density model and its number of components (also of the generative classifier), the
# datasets (from large to small), the probability of every entry to
# be missing (in training and test; a table per rate), and the seeds
MODEL         = "gmm"
K             = 4
DATASETS      = ["magic", "phoneme", "spambase", "banknote", "qsar", "pima"]
MISSING_RATES = [0.1, 0.3, 0.5]
SEEDS         = range(5)

# pclr: cfi's frequencies and the iterations of l-bfgs
N_FREQ   = 64
MAX_ITER = 1000

# the reference (not bold in the tables), and the column of every model in the tables
REFERENCE = "No missing values"
HEADERS = {
    "No missing values":     "No missing",
    "Mean imputation":       "Mean imp.",
    "Conditional mean imputation": "Cond.\\ mean imp.",
    "MICE":                  "MICE",
    "Generative classifier": "Generative",
    "PCLR":                  "PCLR",
    "PCLR fixed density":    "PCLR fixed",
    "PCLR fitted start":     "PCLR fitted",
}

TABLES = "../paper/tables"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── models ────────────────────────────────────────────────────────────────────

# every model is trained on every dataset, missing rate and seed (comment one out to skip
# it): the reference (the lr without missing values, trained and tested on complete
# points), the imputations, the generative classifier (one gmm over the features and the
# label, with the same number of components as pclr's density), and pclr (the
# density and w jointly, the density starting untrained). its variants: starting from the
# fitted density, and the fitted density kept fixed
MODELS = {
    "No missing values":     CompleteData(),
    "Mean imputation":       MeanImputation(),
    "Conditional mean imputation": ConditionalMeanImputation(),
    "MICE":                  MICE(),
    "Generative classifier": GenerativeClassifier(K),
    "PCLR":                  PCLR(n_freq=N_FREQ, max_iter=MAX_ITER),
}

# ── comparison ────────────────────────────────────────────────────────────────

# results[(dataset, rate)][model][score] = one value per seed, the scores log_loss / accuracy
# on the incomplete test points (the reference on the complete ones)
results = {}

for dataset, missing_rate, seed in itertools.product(DATASETS, MISSING_RATES, SEEDS):
    setting = (dataset, missing_rate)
    results.setdefault(setting, {name: {"log_loss": [], "accuracy": []} for name in MODELS})

    # the data, and the density fit on the incomplete training points
    x_train, x_missing, y_train, x_test, x_test_missing, y_test = load_real(dataset, missing_rate, seed, device)
    cache = f"cache/real/{MODEL}/K{K}_{dataset.replace(' ', '_')}_{missing_rate}missing_seed{seed}.pt"
    px = fit_density(MODEL, x_missing, cache, device, K)

    # every model trained on the incomplete training points and scored on the incomplete
    # test points with its own prediction rule. the reference is trained and tested on the
    # complete points
    scores = {}
    for name, model in MODELS.items():
        if isinstance(model, CompleteData):
            model.fit(x_train, y_train)
            x_eval = x_test
        else:
            model.fit(x_missing, y_train, px)
            x_eval = x_test_missing
        with torch.no_grad():
            scores[name] = prediction_scores(model.predict(x_eval), y_test)

    for name, score in scores.items():
        results[setting][name]["log_loss"].append(score["log_loss"])
        results[setting][name]["accuracy"].append(score["accuracy"])
    print(f"{dataset}, {missing_rate:.0%} missing, seed {seed}: " + "   ".join(f"{name} {score['log_loss']:.4f}" for name, score in scores.items()))

real_data_tables(results, DATASETS, MISSING_RATES, {name: HEADERS[name] for name in MODELS}, REFERENCE, TABLES)
