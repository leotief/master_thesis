# training the classifier on incomplete data, step by step, as two figures in
# the style of the pcpg one: with pcpg (figures/train/pcpg.pdf) and with mc
# (figures/train/mc.pdf), the same steps, so they can be read side by side:
#   goal     w is fit to the incomplete training points: the prediction of a
#            point is the area under sigmoid(z) x p(z | x_obs), and the loss
#            is -log of the area of its label
#   step 1   the method estimates the prediction of every training point, and
#            the same point asked again
#   step 2   the loss of that estimate: pcpg gives the exact loss, the mc loss
#            is random and too high on average (jensen), by about 1 / S
#   step 3   the training objective along one direction of w, w = t w*, and
#            its slope, which is what the gradient follows
#   step 4   training: how far the objective is from its optimum after every
#            epoch, and the path of w, along w* and away from it
#
# the setup of train.py: the gmm is fit on the incomplete training data
# (MISSING_RATE of every point missing, at random) and frozen, then w is
# trained by adam on the marginal likelihood, -log p(y | x_obs) with
# p(y=1 | x_obs) = E[sigma(w^T x) | x_obs], and the l2 penalty of sklearn. the exact
# expectation (gauss-hermite per component of the mixture of z) is the
# target, w* its optimum.
# the plotting is in utils/plots.py

import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

# pcpg.py is one directory up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pcpg import PCPG
from utils.baselines import MCBaseline, MixtureGaussHermite
from utils.data import DataManager
from utils.gmm import GMM
from utils.plots import plot_train_step_by_step

OUT_DIR = "figures/train"
os.makedirs(OUT_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

# setup, as in train.py
SEED         = 42
N_FEATURES   = 15
K            = 4
MISSING_RATE = 0.6
C            = 1.0
EPS          = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pcpg nodes and mc samples of the figures. mc with few samples, where its
# bias is clearly visible, both figures for the same kind of training
N_OUTER, N_INNER = 10, 32
MC_SAMPLES       = 4
EXACT_NODES      = 64

# step 1: predictions of N_SCATTER training points, the followed one asked
# N_REPEATS times. step 2: the mc bias of the loss at MC_BUDGETS, pcpg at
# QUAD_NODES. step 3: the objective at T (w = t w*) on N_SLICE training
# points, mc N_SLICE_RUNS times. step 4: training as in train.py, mc N_RUNS times
N_SCATTER    = 400
N_REPEATS    = 2000
MC_BUDGETS   = [1, 2, 4, 8, 16, 32, 64]
QUAD_NODES   = [(2, 4), (2, 8), (3, 12), (4, 16), (6, 24), (8, 32), (10, 40), (14, 56)]
T            = np.linspace(0, 2, 41)
N_SLICE      = 2000
N_SLICE_RUNS = 4
EPOCHS       = 30
BATCH        = 256
LR           = 0.05
N_RUNS       = 3

Z = np.linspace(-12, 12, 1200)

# ── setup ─────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dm = DataManager("sk_class", samples=10_000, features=N_FEATURES, test_size=0.2, random_state=SEED)
x_train, _, y_train, _ = dm.get_data()
rng = np.random.default_rng(SEED)
x_mis = x_train.copy()
x_mis[rng.random(x_mis.shape) < MISSING_RATE] = np.nan
mean = np.nanmean(x_mis, axis=0)

X = torch.tensor(x_mis, device=device)
Y = torch.tensor(y_train, dtype=torch.float32, device=device)
N = len(Y)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size=64, shuffle=True)
first = torch.tensor(np.where(np.isnan(x_mis[:K]), mean, x_mis[:K]), device=device)
px = GMM(device, K=K, means_init=first, n_features=N_FEATURES).to(device)
px.fit(loader)
for p in px.parameters():
    p.requires_grad_(False)

exact = MixtureGaussHermite(px, n_nodes=EXACT_NODES)
pcpg = PCPG(px, n_outer=N_OUTER, n_inner=N_INNER)
mc = MCBaseline(px, n_samples=MC_SAMPLES)
PREDICT = dict(exact=exact.differentiable, pcpg=pcpg.differentiable, mc=mc.differentiable)

# ── training ──────────────────────────────────────────────────────────────────

# mean binary cross-entropy plus the l2 penalty of sklearn, scaled to one point
def objective(predict, w, x, y):
    p = predict(w, x).to(y.dtype).clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(p, y) + w.square().sum() / (2 * C * N)

def objective_value(w, predict=exact.differentiable, x=X, y=Y):
    with torch.no_grad():
        return float(objective(predict, w, x, y))

# w* by full batch l-bfgs on the exact objective
def optimum():
    w = torch.zeros(N_FEATURES, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w], max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = objective(exact.differentiable, w, X, Y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return w.detach()

W_STAR = optimum()
J_STAR = objective_value(W_STAR)

# adam on minibatches from w = 0, as in train.py. the excess objective and w
# at the start and after every epoch
def train(predict, seed):
    torch.manual_seed(SEED + seed)
    w = torch.zeros(N_FEATURES, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * -(-N // BATCH))
    excesses, path = [objective_value(w.detach()) - J_STAR], [w.detach().cpu().numpy()]
    for _ in range(EPOCHS):
        for batch in torch.randperm(N, device=device).split(BATCH):
            loss = objective(predict, w, X[batch], Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        excesses.append(objective_value(w.detach()) - J_STAR)
        path.append(w.detach().cpu().numpy())
    return np.array(excesses), np.array(path)

# ── the pieces ────────────────────────────────────────────────────────────────

# the followed training point: one with label 1, at least 60 % missing and an
# uncertain prediction (between 0.3 and 0.8), the one whose sigmoid(z) varies
# most over p(z | x_obs), so the spread of mc shows
def followed_point():
    with torch.no_grad():
        mix, m, v = px.z_moments(W_STAR, X)
        # E[sigmoid(z)] and E[sigmoid(z)^2] with gauss-hermite per component
        nodes, weights = (torch.tensor(a, dtype=torch.float32, device=device)
                          for a in np.polynomial.hermite.hermgauss(EXACT_NODES))
        s = torch.sigmoid(m[..., None] + torch.sqrt(2 * v)[..., None] * nodes)
        p = ((mix * (s @ weights)).sum(-1) / np.sqrt(np.pi)).cpu().numpy()
        var = ((mix * (s**2 @ weights)).sum(-1) / np.sqrt(np.pi)).cpu().numpy() - p**2
    missing = np.isnan(x_mis).mean(axis=1)
    ok = (y_train == 1) & (missing >= 0.6) & (p > 0.3) & (p < 0.8)
    return int(np.flatnonzero(ok)[np.argmax(var[ok])])

# density of z = w^T x given x_obs for point n, on Z
def pz(w, n):
    with torch.no_grad():
        mix, m, v = (t.double().cpu().numpy() for t in px.z_moments(w, X[n]))
    return (mix[:, None] * np.exp(-(Z - m[:, None])**2 / (2 * v[:, None])) / np.sqrt(2 * np.pi * v[:, None])).sum(0)

# repeated estimates of the prediction of point n, (repeats,)
def repeats(name, n):
    with torch.no_grad():
        if name == "pcpg":
            return np.full(N_REPEATS, float(pcpg(W_STAR, X[n])))
        return mc(W_STAR, X[n].expand(N_REPEATS, -1))

# step 2: the error of the expected loss -log p for point n, against the
# budget. for mc the bias E[-log p_hat] + log p (the mean over N_REPEATS
# estimates), for pcpg the error of its one value
def loss_errors(name, n, p_exact):
    with torch.no_grad():
        if name == "pcpg":
            return ([o * i for o, i in QUAD_NODES],
                    [abs(-np.log(float(pcpg(W_STAR, X[n], n_outer=o, n_inner=i))) + np.log(p_exact))
                     for o, i in QUAD_NODES])
        return (MC_BUDGETS,
                [np.mean(-np.log(np.clip(mc(W_STAR, X[n].expand(N_REPEATS, -1), n_samples=s), EPS, 1)))
                 + np.log(p_exact) for s in MC_BUDGETS])

# step 3: the objective and its slope d/dt along w = t w*, on the first
# N_SLICE training points, (len(T),)
def along_slice(predict):
    values, slopes = [], []
    for t in T:
        t = torch.tensor(float(t), device=device, requires_grad=True)
        loss = objective(predict, t * W_STAR, X[:N_SLICE], Y[:N_SLICE])
        loss.backward()
        values.append(float(loss.detach()))
        slopes.append(float(t.grad))
    return np.array(values), np.array(slopes)

# a path of w (steps, D) in two coordinates, in units of ||w*||: along the
# direction of w* (w* at 1) and the distance from that line. the objective is
# nearly flat in some directions (make_classification has redundant features),
# so single weights can stay far from w* at an objective close to the optimum
def project(path):
    w_star = W_STAR.cpu().numpy()
    norm = np.linalg.norm(w_star)
    along = path @ w_star / norm
    off = np.linalg.norm(path - np.outer(along, w_star / norm), axis=1)
    return np.stack([along, off], axis=1) / norm

def pieces():
    n = followed_point()
    with torch.no_grad():
        p_exact = float(exact(W_STAR, X[n]))
        scatter_exact = exact(W_STAR, X[:N_SCATTER])
    slice_exact = along_slice(exact.differentiable)
    excess_exact, path_exact = train(exact.differentiable, 0)
    common = dict(z=Z, pz=pz(W_STAR, n), answer=p_exact, n=n, n_missing=int(np.isnan(x_mis[n]).sum()),
                  scatter_exact=scatter_exact, t=T, slice_exact=slice_exact, excess_exact=excess_exact,
                  path_exact=project(path_exact), j_star=J_STAR)

    out = {}
    for name in ("pcpg", "mc"):
        with torch.no_grad():
            scatter = (pcpg if name == "pcpg" else mc)(W_STAR, X[:N_SCATTER])
        # pcpg is deterministic, one run is all runs
        slices = [along_slice(PREDICT[name]) for _ in range(1 if name == "pcpg" else N_SLICE_RUNS)]
        runs = [train(PREDICT[name], seed) for seed in range(1 if name == "pcpg" else N_RUNS)]
        out[name] = dict(common, method=name, scatter=scatter, repeats=repeats(name, n),
                         loss_errors=loss_errors(name, n, p_exact), slices=slices,
                         excesses=[r[0] for r in runs], paths=[project(r[1]) for r in runs],
                         budget=N_OUTER * N_INNER if name == "pcpg" else MC_SAMPLES)
    return out

# ── the figures ───────────────────────────────────────────────────────────────

# the explanations below the panels, typeset with latex, the math without
# spaces so that wrapping never splits it. <...> are filled in below. the goal
# is the same in both figures
GOAL = dict(
    heading=r"Fit $w$ on incomplete training points",
    title=r"$\displaystyle{}p(y=1\mid{}x^o)=\mathbb{E}_{p(z\mid{}x^o)}\big[\sigma(z)\big]=\int\sigma(z)\,p(z)\,dz$",
    text=r"In training, also the training points have missing features (here <rate> of them, at random). "
         r"The density model is fit first, its likelihood leaves the missing features out exactly. Then "
         r"$w$ is fit to the prediction that is made at test time: for a point with observed $x^o$ it is "
         r"the yellow area under $\sigma(z)\times{}p(z\mid{}x^o)$, here for a training point with "
         r"<missing> of 15 features missing and label $y=1$. Its loss is $-\log$ of the area of its label, "
         r"and training minimises the mean loss over all points, "
         r"$-\frac{1}{N}\sum_n\log{}p(y_n\mid{}x_n^o)$. So the area is needed for every point in every "
         r"step, and the gradient of its log.",
)

TEXTS = dict(
    pcpg=dict(
        h1="PCPG computes the prediction of every training point",
        h2="The loss of the estimate is the exact loss",
        h3="The objective and its slope are the exact ones",
        h4="Training follows the exact objective",
        step1=r"PCPG computes the area for every training point from the model's characteristic function "
              r"(<budget> nodes per point). Every dot is one of <scatter> training points: its PCPG "
              r"estimate against the exact value. They lie on the diagonal.",
        repeat=r"The same point asked <repeats> times gives the same answer every time: PCPG is "
               r"deterministic, a fixed function of $w$.",
        loss=r"The loss is $-\log$ of the prediction of the label. As the prediction is exact up to the "
             r"quadrature error, so is the loss: the dot sits on the curve, at the exact loss.",
        loss_error=r"The error of the loss falls quickly with the number of nodes "
                   r"$n_{\mathrm{outer}}\times{}n_{\mathrm{inner}}$, it is a deterministic error that does "
                   r"not average out, but it is small.",
        slice=r"The mean loss over <n_slice> training points along one direction of $w$, $w=t\,w^*$ "
              r"($w^*$ the optimum of the exact objective, at $t=1$). Around the optimum the PCPG "
              r"objective is the exact one (in the yellow band), a smooth curve with its minimum at $t=1$. "
              r"For a much larger $w$ the scores $z$ get larger, the fixed nodes too coarse, and it drifts "
              r"above.",
        slope=r"Its slope, which the gradient follows, is the exact one up to there, and the same at "
              r"every call, so even full batch methods like L-BFGS can be used. Training starts at $w=0$ "
              r"and stays in the range where the nodes are good enough.",
        curve=r"Training with Adam on minibatches as in the experiments: the objective minus its optimum "
              r"after every epoch. PCPG trains like the exact objective (yellow band) and ends at the "
              r"same objective.",
        path=r"The path of $w$ from $w=0$, along the direction of $w^*$ and away from it (in units of "
             r"$\|w^*\|$): the same as with the exact objective. Both stop a little short of $w^*$ (star), "
             r"where the objective is nearly flat.",
    ),
    mc=dict(
        h1="MC estimates every prediction from random fill-ins",
        h2="The loss of a random estimate is too high on average",
        h3="The objective is noisy and too high, its minimum moves",
        h4=r"Training ends short of the optimum $w^*$",
        step1=r"MC estimates the area for every training point by the average of $\sigma(z_i)$ over "
              r"<budget> fill-ins of the missing features, drawn from the model. Every dot is one of "
              r"<scatter> training points: its MC estimate against the exact value. They scatter around "
              r"the diagonal.",
        repeat=r"The same point asked <repeats> times gives a different answer every time: a histogram "
               r"around the exact value (yellow), its mean is right, single estimates are not.",
        loss=r"The loss is $-\log$ of the prediction of the label. The curve bends upwards, so the losses "
             r"of the random estimates (dots) are too high on average (Jensen): their mean <mean_loss> is "
             r"above the exact loss <exact_loss>.",
        loss_error=r"The bias of the loss, mean of $-\log\hat{p}$ minus $-\log{}p$, falls like $1/S$ with "
                   r"the number of samples $S$. It is largest where the prediction varies most over the "
                   r"missing features, so it punishes $w$ for relying on them.",
        slice=r"The mean loss over <n_slice> training points along one direction of $w$, $w=t\,w^*$ "
              r"($w^*$ the optimum of the exact objective, at $t=1$). Every MC run (new fill-ins at every "
              r"$t$) is noisy and above the exact objective (yellow band), more so for larger $w$, so its "
              r"minimum moves to a smaller $w$.",
        slope=r"Its slope is noisy and too high, so every step is pushed towards $w=0$: the bias acts like an "
              r"extra penalty that shrinks $w$. With more samples it shrinks like $1/S$.",
        curve=r"Training with Adam on minibatches as in the experiments, new fill-ins in every step: the "
              r"objective minus its optimum after every epoch, <runs> runs. They stop well above the "
              r"exact objective (yellow band), at the optimum of the biased MC objective.",
        path=r"The path of $w$ from $w=0$, along the direction of $w^*$ and away from it (in units of "
             r"$\|w^*\|$): every run is different, and all of them stop far short of $w^*$ (star) and "
             r"of the exact path (yellow), at a smaller $w$.",
    ),
)

def fill(text, d):
    values = {"<rate>": rf"{round(100 * MISSING_RATE)}\%", "<missing>": str(d["n_missing"]),
              "<budget>": str(d["budget"]), "<scatter>": str(N_SCATTER), "<repeats>": f"{N_REPEATS:,}".replace(",", r"\,"),
              "<n_slice>": f"{N_SLICE:,}".replace(",", r"\,"), "<runs>": str(len(d["excesses"])),
              "<exact_loss>": f"{-np.log(d['answer']):.3f}",
              "<mean_loss>": f"{np.mean(-np.log(np.clip(d['repeats'], EPS, 1))):.3f}"}
    for placeholder, value in values.items():
        text = text.replace(placeholder, value)
    return text

def plot_steps():
    for name, d in pieces().items():
        texts = {key: fill(text, d) for key, text in {**{f"goal_{k}": v for k, v in GOAL.items()},
                                                      **TEXTS[name]}.items()}
        plot_train_step_by_step(d, texts, f"{OUT_DIR}/{name}.pdf")

if __name__ == "__main__":
    plot_steps()
