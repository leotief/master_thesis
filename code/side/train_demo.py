# how a logistic regression is trained when some training points have a
# missing feature, step by step, on a 2d toy problem where everything can be
# drawn: the points, the decision boundary, and w itself (a point in a plane).
# one figure, figures/train/step_by_step.pdf, with pcpg and mc side by side
# where they differ:
#   setup   the training points, the incomplete ones as lines (the missing
#           coordinate could be anywhere on them), and the density model
#   step 1  an incomplete point is a line of possible points: its prediction is
#           the average of sigmoid(w^T x) along the line, weighted by
#           p(x_mis | x_obs). mean imputation takes one point instead
#   step 2  two ways to compute that average: pcpg replaces sigma by a sum of
#           bells and averages every bell exactly, mc averages sigma at a few
#           random points on the line
#   step 3  training changes w: sigma along the line changes, and with it the
#           prediction, smoothly for pcpg, noisily for mc
#   step 4  from predictions to the loss: the loss of the point, and summed over
#           all points, mc's loss is too high on average (jensen), more so for a
#           longer w
#   step 5  training walks downhill: the paths of w over the loss, and the loss
#           during training, mc stops short
#   step 6  where training ends: the boundaries in the data, and how sharp the
#           prediction is across them
# the estimators, the density and the loss are the ones of train.py, on two
# features. mc gets MC_FEW samples, so what it does is visible, and steps 3 and
# 4 also show it with as many samples as pcpg has nodes.
# the plotting is in utils/plots.py

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# experiment_utils.py and pcpg.py are one directory up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiment_utils import optimum
from pcpg import PCPG
from side.utils.baselines import MCBaseline, MixtureGaussHermite
from side.utils.gmm import GMM
from side.utils.plots import plot_training_step_by_step

OUT_DIR = "figures/train"

# ── config ────────────────────────────────────────────────────────────────────

SEED = 0

# the toy data: two features, two clusters per class, and the fraction of the
# points that miss one of their two features (which one at random)
N_POINTS       = 600
MISSING_POINTS = 0.7
K              = 4

# pcpg's nodes, mc with few samples and with as many as pcpg has nodes
N_OUTER, N_INNER = 10, 32
MC_FEW           = 4
MC_SAME          = N_OUTER * N_INNER
EXACT_NODES      = 64

# the l2 penalty of sklearn and plain adam on all points at once
C      = 1.0
EPOCHS = 300
LR     = 0.05
EPS    = 1e-6

# the epochs whose boundaries step 4 shows, and the grid of the loss over w
SHOW_EPOCHS = [0, 5, 15, 40, 300]
GRID        = 61

device = torch.device("cpu")

# ── setup ─────────────────────────────────────────────────────────────────────

x_np, y_np = make_classification(n_samples=N_POINTS, n_features=2, n_informative=2, n_redundant=0,
                                 n_clusters_per_class=2, class_sep=1.0, random_state=SEED)
x_np = (x_np - x_np.mean(axis=0)).astype(np.float32)
rng = np.random.default_rng(SEED)
x_missing_np = x_np.copy()
incomplete = rng.random(N_POINTS) < MISSING_POINTS
x_missing_np[np.flatnonzero(incomplete), rng.integers(0, 2, incomplete.sum())] = np.nan

x_missing = torch.tensor(x_missing_np)
y = torch.tensor(y_np, dtype=torch.float32)

# the density model, fit on the incomplete points
torch.manual_seed(SEED)
first = torch.where(torch.isnan(x_missing[:K]), x_missing.nanmean(dim=0), x_missing[:K])
px = GMM(device, K=K, means_init=first, n_features=2)
px.fit(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_missing), batch_size=64, shuffle=True))
for parameter in px.parameters():
    parameter.requires_grad_(False)

pcpg    = PCPG(px, n_outer=N_OUTER, n_inner=N_INNER)
mc_few  = MCBaseline(px, n_samples=MC_FEW)
mc_same = MCBaseline(px, n_samples=MC_SAME)
exact   = MixtureGaussHermite(px, n_nodes=EXACT_NODES)


# the loss of train.py
def objective(predict, w, x, y, n_train):
    p = predict(w, x).to(y.dtype).clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(p, y) + w.square().sum() / (2 * C * n_train)


def exact_objective(w):
    with torch.no_grad():
        return objective(exact.differentiable, w, x_missing, y, N_POINTS).item()


# the optimum of the exact loss, and the two logistic regressions on complete
# points: on all complete points (an oracle), and with the gaps filled by the means
w_star = optimum(objective, exact.differentiable, x_missing, y, N_POINTS, device)
x_filled = torch.where(torch.isnan(x_missing), x_missing.nanmean(dim=0), x_missing)
w_complete = LogisticRegression(C=C, fit_intercept=False).fit(x_np, y_np).coef_[0]
w_mean = LogisticRegression(C=C, fit_intercept=False).fit(x_filled.numpy(), y_np).coef_[0]


# plain adam on all points from w = 0, w after every epoch
def train(predict, seed):
    torch.manual_seed(seed)
    w = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=LR)
    path = [w.detach().clone()]
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = objective(predict, w, x_missing, y, N_POINTS)
        loss.backward()
        optimizer.step()
        path.append(w.detach().clone())
    return torch.stack(path).numpy()


# ── the followed point: x2 missing, its line crosses two clusters ─────────────

lines = np.flatnonzero(np.isnan(x_missing_np[:, 1]))
with torch.no_grad():
    weights = px._component_weights(x_missing[lines]).numpy()
    prediction = exact(w_star, x_missing[lines])
means = px.means.numpy()[:, 1]
by_weight = np.argsort(-weights, axis=1)
second = np.take_along_axis(weights, by_weight[:, 1:2], axis=1)[:, 0]
spread = np.abs(means[by_weight[:, 0]] - means[by_weight[:, 1]])
# the point whose prediction changes clearly from w = 0 (0.5) to w*, whose
# line crosses two clusters, and where mean imputation differs most from the
# average along the line
filled = x_missing[lines].clone()
filled[:, 1] = x_missing[:, 1].nanmean()
mean_prediction = torch.sigmoid(filled @ w_star).numpy()
usable = np.abs(prediction - 0.5) > 0.25
point = int(lines[np.argmax(np.where(usable, np.abs(prediction - mean_prediction) * second, -1))])
x_point = x_missing[point]
x1 = float(x_point[0])

# p(x2 | x1) along the line, and sigmoid(w* . x) on it
X2 = np.linspace(x_np[:, 1].min() - 1, x_np[:, 1].max() + 1, 600)
with torch.no_grad():
    weight = px._component_weights(x_point).numpy()
stds = np.sqrt(np.exp(px.log_vars.numpy()[:, 1]) + 1e-6)
line_density = (weight[:, None] * np.exp(-(X2 - means[:, None])**2 / (2 * stds[:, None]**2))
                / np.sqrt(2 * np.pi * stds[:, None]**2)).sum(axis=0)
w_np = w_star.numpy()
line_sigmoid = 1 / (1 + np.exp(-(w_np[0] * x1 + w_np[1] * X2)))
with torch.no_grad():
    answer = float(exact(w_star, x_point))
mean_x2 = float(x_filled[point, 1])

# ── steps 2 to 4: the followed point along its line ──────────────────────────

# z = w*^T x on the line, and pcpg's sigma along it: the sum of the bells with
# the largest weights. with the sign flip (z mostly positive) pcpg computes
# 1 - the sum of the bells of -z
z_line = w_np[0] * x1 + w_np[1] * X2
gammas, rhos = (a.double().numpy() for a in pcpg._outer_nodes(N_OUTER))
by_rho = np.argsort(-rhos)
with torch.no_grad():
    flip = float(px.z_mean(w_star, x_point)) > 0
    answer_pcpg = float(pcpg(w_star, x_point))
BELL_KS = [1, 2, 4, N_OUTER]


def bell_sum(k):
    zz = -z_line if flip else z_line
    total = (0.5 * rhos[by_rho[:k], None] * np.exp(zz / 2 - np.outer(gammas[by_rho[:k]], zz**2) / 2)).sum(axis=0)
    return 1 - total if flip else total


# mc's random positions on the line
torch.manual_seed(SEED)
with torch.no_grad():
    samples = mc_few.model.sample(MC_FEW, x_obs=x_point).numpy()[:, 1]

# the point's prediction and loss along w = t w*: t = 0 is w = 0, t = 1 the
# optimum w*
T = np.linspace(0, 1.6, 81)
SHOW_T = [0.25, 0.5, 1.0]
label = int(y_np[point])


def of_label(p):
    return p if label == 1 else 1 - p


def along_point(estimator):
    with torch.no_grad():
        return np.array([float(estimator(torch.tensor(t * w_np, dtype=torch.float32), x_point)) for t in T])


# mc's loss of the point on average, over 400 draws of MC_FEW samples
def mc_point_loss(t):
    torch.manual_seed(SEED)
    with torch.no_grad():
        fills = px.sample(MC_FEW * 400, x_obs=x_point)
        p = torch.sigmoid(fills @ torch.tensor(t * w_np, dtype=torch.float32)).view(400, MC_FEW).mean(dim=-1)
        return float(-torch.log(of_label(p).clamp(EPS, 1)).mean())


norm_star = float(w_star.norm())


# ── step 3: the loss over w ───────────────────────────────────────────────────

# a square around w = 0 and w*
centre = w_np / 2
reach = 0.85 * norm_star
W1 = np.linspace(centre[0] - reach, centre[0] + reach, GRID)
W2 = np.linspace(centre[1] - reach, centre[1] + reach, GRID)
j_star = exact_objective(w_star)


def landscape(value):
    return np.array([[value(torch.tensor([a, b], dtype=torch.float32)) for a in W1] for b in W2])


def loss_with(predict):
    def value(w):
        with torch.no_grad():
            return objective(predict, w, x_missing, y, N_POINTS).item() - j_star
    return value


# what mc's loss is on average: its loss with MC_FEW samples, averaged over
# REPEATS draws of the samples. the same draws at every w, so that the
# landscape is smooth
REPEATS = 100


def mc_average_loss(w):
    torch.manual_seed(SEED)
    with torch.no_grad():
        fills = px.sample(MC_FEW * REPEATS, x_obs=x_missing)                    # (points, samples, 2)
        p = torch.sigmoid(fills @ w).view(N_POINTS, REPEATS, MC_FEW).mean(dim=-1).clamp(EPS, 1 - EPS)
        losses = -(y[:, None] * torch.log(p) + (1 - y[:, None]) * torch.log(1 - p))
        return (losses.mean() + w.square().sum() / (2 * C * N_POINTS)).item() - j_star


# the density of the data on a grid, for the setup
G1 = np.linspace(x_np[:, 0].min() - 0.5, x_np[:, 0].max() + 0.5, 200)
G2 = np.linspace(x_np[:, 1].min() - 0.5, x_np[:, 1].max() + 0.5, 200)
grid = torch.tensor(np.stack(np.meshgrid(G1, G2), axis=-1).reshape(-1, 2), dtype=torch.float32)
with torch.no_grad():
    density = np.exp(px(grid).numpy()).reshape(len(G2), len(G1))

# ── the figure ────────────────────────────────────────────────────────────────

def along(value):
    return np.array([value(torch.tensor(t * w_np, dtype=torch.float32)) for t in T])


def gaps(path):
    return np.array([exact_objective(torch.tensor(w)) - j_star for w in path])


landscape_mc = landscape(mc_average_loss)
row, column = np.unravel_index(np.argmin(landscape_mc), landscape_mc.shape)

prediction_exact, prediction_pcpg = along_point(exact), along_point(pcpg)
torch.manual_seed(SEED)
prediction_mc = [along_point(mc_few) for _ in range(3)]
path_exact, path_pcpg = train(exact.differentiable, SEED), train(pcpg.differentiable, SEED)
paths_mc = [train(mc_few.differentiable, SEED + run) for run in range(3)]
path_mc_same = train(mc_same.differentiable, SEED)

d = dict(x=x_np, x_missing=x_missing_np, y=y_np, point=point, x1=x1, X2=X2, line_density=line_density,
         line_sigmoid=line_sigmoid, answer=answer, mean_x2=mean_x2, w_star=w_np, w_complete=w_complete,
         w_mean=w_mean, n_points=N_POINTS, n_incomplete=int(incomplete.sum()), G1=G1, G2=G2, density=density,
         mc_few=MC_FEW, mc_same=MC_SAME, n_outer=N_OUTER, n_inner=N_INNER, label=label,
         # step 2
         line_bells=[bell_sum(k) for k in BELL_KS], bell_ks=BELL_KS, answer_pcpg=answer_pcpg, samples=samples,
         # steps 3 and 4
         t=T, show_t=SHOW_T, sigma_t=[1 / (1 + np.exp(-t * z_line)) for t in SHOW_T],
         prediction_exact=prediction_exact, prediction_pcpg=prediction_pcpg, prediction_mc=prediction_mc,
         point_loss_exact=-np.log(of_label(prediction_exact)), point_loss_pcpg=-np.log(of_label(prediction_pcpg)),
         point_loss_mc=[-np.log(np.clip(of_label(p), EPS, 1)) for p in prediction_mc],
         point_loss_mc_mean=np.array([mc_point_loss(t) for t in T]),
         loss_exact=along(loss_with(exact.differentiable)), loss_pcpg=along(loss_with(pcpg.differentiable)),
         loss_mc=along(mc_average_loss),
         # step 5
         W1=W1, W2=W2, landscape=landscape(loss_with(exact.differentiable)), mc_minimum=(W1[column], W2[row]),
         show_epochs=SHOW_EPOCHS, path_exact=path_exact, path_pcpg=path_pcpg, paths_mc=paths_mc,
         path_mc_same=path_mc_same, gaps_exact=gaps(path_exact), gaps_pcpg=gaps(path_pcpg),
         gaps_mc=[gaps(p) for p in paths_mc], gaps_mc_same=gaps(path_mc_same))

os.makedirs(OUT_DIR, exist_ok=True)
plot_training_step_by_step(d, f"{OUT_DIR}/step_by_step.pdf")
