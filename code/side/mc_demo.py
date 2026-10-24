# the monte carlo baseline (MCBaseline in utils/baselines.py), on the setup of
# main.py (same data, gmm, classifier and test point), as one figure,
# figures/mc/step_by_step.pdf, in the style of the pcpg one:
#   goal     the area under sigmoid(z) x p(z)
#   step 1   draw the missing features from the model, which gives samples of z
#   step 2   the prediction sigmoid(z_i) at every sample, the estimate is their average
#   step 3   the average is the area under sigmoid(z) x the histogram of the
#            samples, which closes in on the goal (10, 100, 10000 samples)
#   step 4   how fast: the error is random and shrinks like 1 / sqrt(n)
#
# the samples are drawn as in GMM.sample: a component k from p(k | x_obs), then
# the missing features from its gaussian. here the component of every sample is
# kept, so it can be coloured. mc needs nothing from the model but samples.
# the plotting is in utils/plots.py

import os
import random

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from utils.data import DataManager
from utils.gmm import GMM
from utils.plots import plot_mc_step_by_step

OUT_DIR = "figures/mc"
os.makedirs(OUT_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

# setup, the same as in main.py
SEED         = 42
N_FEATURES   = 15
K            = 4
MISSING_RATE = 0.6
TEST_IDX     = 5

# samples of the steps: the scatter and histogram of step 1, the samples of
# step 2, the progression of step 3, and the runs of step 4
N_SCATTER   = 2000
N_SHOW      = 20
SAMPLE_KS   = [10, 100, 10_000]
N_MAX       = 10_000
N_RUNS_SHOW = 5
N_RUNS      = 500

Z = np.linspace(-12, 12, 1200)

# ── setup ─────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dm = DataManager("sk_class", samples=10_000, features=N_FEATURES, test_size=0.2, random_state=SEED)
x_train, _, y_train, _ = dm.get_data()
train_loader, _        = dm.get_dataloaders(batch_size=64)
x_train_tensor, _      = train_loader.dataset.tensors
px = GMM("cpu", K=K, means_init=x_train_tensor[:K], n_features=N_FEATURES)
px.fit(train_loader)
w = torch.tensor(LogisticRegression().fit(x_train, y_train).coef_[0], dtype=torch.float32)
x_partial = dm.get_test_missing(missing_rate=MISSING_RATE, idx=TEST_IDX)

# the conditional mixture p(x_mis | x_obs): the weights p(k | x_obs), and the
# means and variances of the missing features. and the distribution of z
with torch.no_grad():
    MIS = torch.isnan(x_partial).numpy()
    MIX, M, V = (t.double().numpy() for t in px.z_moments(w, x_partial))
    MEANS = px.means.double().numpy()[:, MIS]
    STDS = np.sqrt(np.exp(px.log_vars.double().numpy()[:, MIS]) + 1e-6)
    W_MIS = w.double().numpy()[MIS]
    Z_OBS = float(w[~torch.from_numpy(MIS)] @ x_partial[~torch.from_numpy(MIS)])

# ── the pieces ────────────────────────────────────────────────────────────────

# the weighted components pi_k N(z; m_k, v_k), one row per component
def pz_components(z):
    return MIX[:, None] * np.exp(-(z - M[:, None])**2 / (2 * V[:, None])) / np.sqrt(2 * np.pi * V[:, None])

# n samples of the missing features as in GMM.sample, with their components,
# and their scores z = w_obs^T x_obs + w_mis^T x_mis
def draw(rng, n):
    k = rng.choice(len(MIX), size=n, p=MIX / MIX.sum())
    x_mis = MEANS[k] + STDS[k] * rng.standard_normal((n, len(W_MIS)))
    return k, x_mis, Z_OBS + x_mis @ W_MIS

# exact E[f(z)] under the mixture, by adaptive quadrature per component
def exact_mean(f):
    vals = [quad(lambda t: f(m + np.sqrt(v) * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi), -12, 12,
                 points=[np.clip(-m / np.sqrt(v), -12, 12)], epsabs=0, epsrel=1e-13, limit=200)[0]
            for m, v in zip(M, V)]
    return MIX @ np.array(vals)

# ── the figure ────────────────────────────────────────────────────────────────

# the explanations below the panels, typeset with latex, the math without
# spaces so that wrapping the lines never splits it. <...> are filled in by
# plot_step_by_step
TEXTS = dict(
    goal=r"Some features are missing, so the score $z=w^\top{}x$ is uncertain: $p(z)$, from "
         r"the density model (grey, area 1). The expected prediction "
         r"$\mathbb{E}[\sigma(z)\mid{}x_{\mathrm{obs}}]$ is the yellow area under "
         r"$\sigma(z)\times{}p(z)$. Monte Carlo does not compute it at all, it estimates it "
         r"by trying out random fill-ins of the missing features.",
    draw=r"Draw $n$ fill-ins $x_{\mathrm{mis}}^{(i)}\sim{}p(x_{\mathrm{mis}}\mid{}x_{\mathrm{obs}})$ "
         r"from the density model: pick a component $k$ with weight $\pi_k=p(k\mid{}x_{\mathrm{obs}})$, "
         r"then the missing features from its Gaussian (dots, coloured by component, ellipses "
         r"$2$ standard deviations). Shown are 2 of the <n_mis> missing features.",
    scores=r"Every fill-in, together with the observed features, gives one score "
           r"$z_i=w^\top{}x^{(i)}$. The scores are samples from $p(z)$: their histogram "
           r"(indigo) follows the density (grey). The model only has to be able to sample, "
           r"nothing about $p(z)$ is computed.",
    predict=r"The classifier is run on every completed input, which gives the prediction "
            r"$\sigma(z_i)$ (dots on the yellow $\sigma(z)$). Where $p(z)$ has more mass, "
            r"more samples land, so the predictions there count more often.",
    average=r"The estimate is the plain average "
            r"$\frac{1}{n}\sum_i\sigma(z_i)$ of the predictions (indigo line), here of the "
            r"<n_show> samples on the left, against the exact value (yellow). Every sample "
            r"counts the same, the weighting by $p(z)$ comes from where the samples are.",
    hists=r"Every panel: $\sigma(z)$ times the histogram of the first $n$ samples (indigo bars, "
          r"each sample adds $\sigma(z_i)/n$ to the area of its bar), against "
          r"$\sigma(z)\times{}p(z)$ (yellow, dashed) and its area. The area of the bars is exactly "
          r"the Monte Carlo estimate. With few samples the bars are a rough, random picture of "
          r"the goal, with many they fill it out.",
    runs=r"Every run of Monte Carlo gives a different estimate (indigo lines: <n_runs_show> runs, "
         r"the dark one is the run of step 3). They all close in on the exact value, inside "
         r"a band of $\pm2\sigma_{\mathrm{MC}}/\sqrt{n}$ around it (yellow), where "
         r"$\sigma_{\mathrm{MC}}^2=\mathrm{Var}[\sigma(z)]=<var>$ is the variance of one "
         r"prediction.",
    rate=r"The root mean squared error over <n_runs> runs falls like "
         r"$\sigma_{\mathrm{MC}}/\sqrt{n}$ (dashed): every digit of accuracy costs 100 times "
         r"more samples. Since $\sigma(z)\in[0,1]$ the variance is at most $1/4$, whatever the "
         r"model, but the error never becomes 0. The estimate at <n_max> samples is "
         r"<estimate> (exact <answer>).",
)

def pieces():
    rng = np.random.default_rng(SEED)
    answer = exact_mean(expit)
    var = exact_mean(lambda z: expit(z)**2) - answer**2

    # step 1: the samples in two of the missing features, the two with the
    # largest |w|, and their scores
    k, x_mis, zs = draw(rng, N_SCATTER)
    feats = np.argsort(-np.abs(W_MIS))[:2]
    mis_idx = np.flatnonzero(MIS)

    # steps 2 and 3: one run, the first n samples of it. step 4: more runs,
    # the first one is this one
    _, _, z_run = draw(rng, N_MAX)
    runs = np.vstack([z_run] + [draw(rng, N_MAX)[2] for _ in range(N_RUNS - 1)])
    preds = expit(runs)
    running = np.cumsum(preds, axis=1) / np.arange(1, N_MAX + 1)

    ns = np.unique(np.logspace(0, np.log10(N_MAX), 40).astype(int))
    rmse = np.sqrt(((running[:, ns - 1] - answer)**2).mean(axis=0))

    components = pz_components(Z)
    return dict(z=Z, pz=components.sum(axis=0), components=components, mix=MIX, order=np.argsort(-MIX),
                means=MEANS[:, feats], stds=STDS[:, feats], feats=mis_idx[feats], k=k,
                x_mis=x_mis[:, feats], z_samples=zs, z_run=z_run, n_show=N_SHOW,
                sample_ks=SAMPLE_KS, running=running[:N_RUNS_SHOW], ns=ns, rmse=rmse,
                answer=answer, var=var, estimate=running[0, -1])

def plot_step_by_step():
    d = pieces()
    fill = {"<n_mis>": str(int(MIS.sum())), "<n_show>": str(N_SHOW), "<n_runs_show>": str(N_RUNS_SHOW),
            "<n_runs>": str(N_RUNS), "<n_max>": f"{N_MAX:,}".replace(",", r"\,"),
            "<var>": f"{d['var']:.4f}", "<estimate>": f"{d['estimate']:.4f}",
            "<answer>": f"{d['answer']:.4f}"}
    texts = {}
    for key, text in TEXTS.items():
        for placeholder, value in fill.items():
            text = text.replace(placeholder, value)
        texts[key] = text
    plot_mc_step_by_step(d, texts, f"{OUT_DIR}/step_by_step.pdf")

if __name__ == "__main__":
    plot_step_by_step()
