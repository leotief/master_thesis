# how a logistic regression is trained when some training points have a
# missing feature, on the 2d toy problem of utils/toy.py: with the expected
# prediction by cfi, against mc with few and with as many samples as cfi has
# frequencies. the density is kept fixed, only w is trained

# the plotting is in utils/plots.py

import os
import sys
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

# experiment_utils.py and pclr.py are one directory up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pclr
from side.utils.baselines import monte_carlo
from side.utils.plots import plot_training_step_by_step
from side.utils.toy import (C, EPS, N_POINTS, SEED, exact, objective, point, px, toy, w_np, w_star,
                            x_missing, x_point, y, y_np)

OUT_DIR = "figures/train"

# ── config ────────────────────────────────────────────────────────────────────

# cfi's frequencies, mc with few samples and with as many as cfi has frequencies
N_FREQ  = 64
MC_FEW  = 4
MC_SAME = N_FREQ

# plain adam on all points at once
EPOCHS = 300
LR     = 0.05

# the current w of steps 1 and 2 (after STEP steps), the dots on the paths of
# step 3, and the grid of the loss over w
STEP      = 15
DOT_EVERY = 20
GRID      = 61

# ── setup ─────────────────────────────────────────────────────────────────────

# the data, the density, w* and the followed point are the ones of utils/toy.py
# the predictions predict(w, x_partial) of cfi and mc, with their counts
cfi     = partial(pclr.cfi, px, n_freq=N_FREQ)
mc_few  = partial(monte_carlo, px, n_samples=MC_FEW)
mc_same = partial(monte_carlo, px, n_samples=MC_SAME)
label   = int(y_np[point])


def exact_objective(w):
    with torch.no_grad():
        return objective(exact, w, x_missing, y, N_POINTS).item()


# the loss of the figure, the average of -log p(y | x_obs), without the penalty
def loss(w):
    with torch.no_grad():
        p = exact(torch.as_tensor(w, dtype=torch.float32), x_missing).to(y.dtype).clamp(EPS, 1 - EPS)
        return F.binary_cross_entropy(p, y).item()


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


path_exact, path_cfi = train(exact, SEED), train(cfi, SEED)
paths_mc = [train(mc_few, SEED + run) for run in range(3)]
path_mc_same = train(mc_same, SEED)
w_now = path_exact[STEP]

# ── step 1: the predictions at the current w ──────────────────────────────────

with torch.no_grad():
    prediction_now_cfi = float(cfi(torch.tensor(w_now), x_point))
    predictions_now = exact(torch.tensor(w_now), x_missing).numpy()

# mc's random values of x2 on the line of the point
torch.manual_seed(SEED)
with torch.no_grad():
    mc_samples = px.sample(MC_FEW, x_obs=x_point).numpy()[:, 1]
prediction_now_mc = float((1 / (1 + np.exp(-(w_now[0] * toy["x1"] + w_now[1] * mc_samples)))).mean())

# ── step 2: the losses of all points at the current w ─────────────────────────

losses_now = -np.log(np.clip(np.where(y_np == 1, predictions_now, 1 - predictions_now), EPS, 1))

# ── goal and step 3: the loss over w ──────────────────────────────────────────

# a square around w = 0 and w*
centre = w_np / 2
reach = 0.85 * float(w_star.norm())
W1 = np.linspace(centre[0] - reach, centre[0] + reach, GRID)
W2 = np.linspace(centre[1] - reach, centre[1] + reach, GRID)
j_star = exact_objective(w_star)


def landscape(value):
    return np.array([[value(torch.tensor([a, b], dtype=torch.float32)) for a in W1] for b in W2])


# what mc's loss is on average: its loss with MC_FEW samples, averaged over
# REPEATS draws of the samples. the same draws at every w, so that the
# landscape is smooth. its minimum is where mc's training stops
REPEATS = 100


def mc_average_loss(w):
    torch.manual_seed(SEED)
    with torch.no_grad():
        fills = px.sample(MC_FEW * REPEATS, x_obs=x_missing)                    # (points, samples, 2)
        p = torch.sigmoid(fills @ w).view(N_POINTS, REPEATS, MC_FEW).mean(dim=-1).clamp(EPS, 1 - EPS)
        losses = -(y[:, None] * torch.log(p) + (1 - y[:, None]) * torch.log(1 - p))
        return (losses.mean() + w.square().sum() / (2 * C * N_POINTS)).item() - j_star


def gaps(path):
    return np.array([exact_objective(torch.tensor(w)) - j_star for w in path])


landscape_mc = landscape(mc_average_loss)
row, column = np.unravel_index(np.argmin(landscape_mc), landscape_mc.shape)

# ── the figure ────────────────────────────────────────────────────────────────

d = dict(**toy, mc_few=MC_FEW, mc_same=MC_SAME, label=label, step_epoch=STEP, w_now=w_now,
         # step 1
         prediction_now_cfi=prediction_now_cfi, prediction_now_mc=prediction_now_mc, mc_samples=mc_samples,
         # step 2
         losses_now=losses_now, loss_now=loss(w_now),
         # goal and step 3
         W1=W1, W2=W2, landscape=landscape(lambda w: exact_objective(w) - j_star), mc_minimum=(W1[column], W2[row]),
         dot_every=DOT_EVERY, path_exact=path_exact, path_cfi=path_cfi, paths_mc=paths_mc,
         path_mc_same=path_mc_same, gaps_exact=gaps(path_exact), gaps_cfi=gaps(path_cfi),
         gaps_mc=[gaps(p) for p in paths_mc], gaps_mc_same=gaps(path_mc_same))

os.makedirs(OUT_DIR, exist_ok=True)
plot_training_step_by_step(d, f"{OUT_DIR}/steps.pdf")
