# main experiment script
#
# setup: fit a gmm as p(x) and a logistic regression for w on complete training
# data. then take one test point, hide some of its features, and estimate the
# expected prediction E[sigmoid(w^T x) | x_obs] with every method in METHODS.
#
# the methods are defined once below, the experiments are in
# side/utils/analyses.py and only called at the bottom. "budget" always means the
# number of cf evaluations (samples for mc), so the methods are compared at
# roughly the same cost

import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from pcpg import PCPGMC, PCPGMCHermite, PCPGDeterministic
from side.utils.analyses import compare_estimators
from side.utils.baselines import GMMAdaptiveQuad, GMMGaussHermite, MCBaseline, MeanImputation
from side.utils.data import DataManager
from side.utils.gmm import GMM

# ── config ────────────────────────────────────────────────────────────────────

SEED = 42

# data, density model and test point. more missing features means the cf is
# damped more, which makes pcpg easier
N_FEATURES   = 15
K            = 4        # gmm components
MISSING_RATE = 0.6
TEST_IDX     = 5

# pcpg-mc splits a budget N into n_pg * n_gauss with n_gauss / n_pg = PCPG_RATIO
PCPG_RATIO = 5
# quadrature nodes: inner gauss-hermite rule of pcpg-gh and pcpg-quad, outer pg
# rule of pcpg-quad (golub-welsch is only reliable up to ~19)
N_GAUSS_QUAD = 20
N_PG_QUAD    = 19
PG_METHOD    = "golub_welsch"
# gauss-hermite nodes of the gmm baseline
N_GH_BASELINE = 10
# samples of the ground truth. mc works for any model that can be sampled, its
# noise is about 0.5 / sqrt(N_MC_GT) ~ 1e-4, errors below that can't be resolved
N_MC_GT       = 10_000_000

# compare_estimators: one budget for all methods
BUDGET = 50_000

# ── setup ─────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data and density model. the gmm is fit on complete data, the means start at
# the first K training points
dm = DataManager("sk_class", samples=10_000, features=N_FEATURES, test_size=0.2, random_state=SEED)
x_train, _, y_train, _ = dm.get_data()
train_loader, _        = dm.get_dataloaders(batch_size=64)

x_train_tensor, _ = train_loader.dataset.tensors
px = GMM(device, K=K, means_init=x_train_tensor[:K].to(device), n_features=N_FEATURES).to(device)
px.fit(train_loader)

# classifier trained on complete data, only its weights are used (the
# intercept is dropped, the estimators only use w^T x)
clf = LogisticRegression().fit(x_train, y_train)
w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)

# the test point everything is evaluated on, with nan for the missing entries
x_partial = dm.get_test_missing(missing_rate=MISSING_RATE, idx=TEST_IDX, device=device)
n_missing = int(torch.isnan(x_partial).sum())
print(f"missing features: {n_missing} / {N_FEATURES}\n")

# ── methods ───────────────────────────────────────────────────────────────────
#
# name -> (estimator, budget -> keyword arguments). the budget N is the number
# of cf evaluations (samples for mc). None means the method is deterministic and
# takes no budget. every estimator is created once here, re-creating it would
# reset its rng and every run would give the same value.
# pcpg-quad computes its pg nodes here, which takes a few seconds

# pcpg-mc: N = n_pg * n_gauss with n_gauss / n_pg = PCPG_RATIO
def pcpg_split(N):
    n_pg = max(1, int(np.sqrt(N / PCPG_RATIO)))
    return {"n_pg": n_pg, "n_gauss": max(1, N // n_pg)}

METHODS = {
    # ground truth: mc with many samples
    "GT":                (MCBaseline(px), lambda N: {"n_samples": N_MC_GT}),
    # baselines
    "MC":                (MCBaseline(px), lambda N: {"n_samples": N}),
    "Mean imputation":   (MeanImputation(torch.tensor(x_train.mean(axis=0), device=device)), None),
    "GMM Gauss-Hermite": (GMMGaussHermite(px, n_nodes=N_GH_BASELINE), None),
    "GMM adaptive quad": (GMMAdaptiveQuad(px), None),
    # pcpg. pcpg-gh uses N_GAUSS_QUAD cf evaluations per omega, so it gets
    # N / N_GAUSS_QUAD pg samples for the same cost
    "PCPG-MC":           (PCPGMC(px), pcpg_split),
    "PCPG-GH":           (PCPGMCHermite(px, n_gauss_quad=N_GAUSS_QUAD), lambda N: {"n_pg": max(1, N // N_GAUSS_QUAD)}),
    "PCPG-Quad":         (PCPGDeterministic(px, n_pg_quad=N_PG_QUAD, n_gauss_quad=N_GAUSS_QUAD, pg_method=PG_METHOD), None),
}

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compare_estimators(METHODS, w, x_partial, gt="GT", budget=BUDGET)
