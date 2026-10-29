# the 2d toy problem of the step-by-step figures (estimate_demo.py and
# train_demo.py), where everything can be drawn: the points, the boundary, and
# w itself (a point in a plane). importing it builds everything:
#   the data      two features, two clusters per class, and the points that miss
#                 one of their two features (which one at random)
#   the density   a gmm fit on the incomplete points
#   the classifier w_star, the optimum of the loss of train.py, with the exact
#                 expected prediction. and the logistic regressions on complete
#                 data and with the gaps filled by the means
#   the point     the incomplete point the figures follow: x2 missing, its line
#                 crosses two clusters
# the plotting is in utils/plots.py

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from experiment_utils import optimum
from side.utils.baselines import MixtureGaussHermite
from side.utils.gmm import GMM

# ── config ────────────────────────────────────────────────────────────────────

SEED = 0

# the toy data: two features, two clusters per class, and the fraction of the
# points that miss one of their two features
N_POINTS       = 600
MISSING_POINTS = 0.7
K              = 4

# the l2 penalty of sklearn, and the nodes of the exact expected prediction
C           = 1.0
EPS         = 1e-6
EXACT_NODES = 64

# ── the data and the density ──────────────────────────────────────────────────

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
px = GMM(torch.device("cpu"), K=K, means_init=first, n_features=2)
px.fit(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_missing), batch_size=64, shuffle=True))
for parameter in px.parameters():
    parameter.requires_grad_(False)

exact = MixtureGaussHermite(px, n_nodes=EXACT_NODES)

# ── the classifier ────────────────────────────────────────────────────────────

# the loss of train.py
def objective(predict, w, x, y, n_train):
    p = predict(w, x).to(y.dtype).clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(p, y) + w.square().sum() / (2 * C * n_train)


# the optimum of the exact loss, and the two logistic regressions on complete
# points: on all complete points (an oracle), and with the gaps filled by the means
w_star = optimum(objective, exact.estimate_diff, x_missing, y, N_POINTS, torch.device("cpu"))
x_filled = torch.where(torch.isnan(x_missing), x_missing.nanmean(dim=0), x_missing)
w_complete = LogisticRegression(C=C, fit_intercept=False).fit(x_np, y_np).coef_[0]
w_mean = LogisticRegression(C=C, fit_intercept=False).fit(x_filled.numpy(), y_np).coef_[0]
w_np = w_star.numpy()

# ── the followed point: x2 missing, its line crosses two clusters ─────────────

lines = np.flatnonzero(np.isnan(x_missing_np[:, 1]))
with torch.no_grad():
    weights = px._component_weights(x_missing[lines]).numpy()
    prediction = exact.estimate(w_star, x_missing[lines])
means = px.means.numpy()[:, 1]
by_weight = np.argsort(-weights, axis=1)
second = np.take_along_axis(weights, by_weight[:, 1:2], axis=1)[:, 0]
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
line_sigmoid = 1 / (1 + np.exp(-(w_np[0] * x1 + w_np[1] * X2)))
with torch.no_grad():
    answer = exact.estimate(w_star, x_point)
mean_x2 = float(x_filled[point, 1])

# the density of the data on a grid
G1 = np.linspace(x_np[:, 0].min() - 0.5, x_np[:, 0].max() + 0.5, 200)
G2 = np.linspace(x_np[:, 1].min() - 0.5, x_np[:, 1].max() + 0.5, 200)
grid = torch.tensor(np.stack(np.meshgrid(G1, G2), axis=-1).reshape(-1, 2), dtype=torch.float32)
with torch.no_grad():
    density = np.exp(px(grid).numpy()).reshape(len(G2), len(G1))

# what the feature space panels need, as a dict
toy = dict(x=x_np, x_missing=x_missing_np, y=y_np, point=point, x1=x1, X2=X2, line_density=line_density,
           line_sigmoid=line_sigmoid, answer=answer, mean_x2=mean_x2, w_star=w_np, w_complete=w_complete,
           w_mean=w_mean, n_points=N_POINTS, n_incomplete=int(incomplete.sum()), G1=G1, G2=G2, density=density)
