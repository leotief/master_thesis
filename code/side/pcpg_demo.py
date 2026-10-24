# the complete pcpg procedure, on the setup of main.py (same data, gmm,
# classifier and test point), as one figure, figures/pcpg/step_by_step.pdf:
#   goal     the expected prediction as an equation, then as the area under
#            sigmoid(z) x p(z)
#   step 1   pg: where the bells come from, and how they add up to sigmoid(z)
#   step 2   hs: where the waves come from, and how they add up to a bell
#   step 3   the area under wave x p(z) comes from the model's cf
#   step 4   the wave areas add up to the area under a bell (1, 2, 4, all waves)
#   step 5   the bell areas add up to the goal (1, 2, 4, all bells)
# and the four figures of the methods chapter, figures/pcpg/{bells,waves,cf,summary}.pdf,
# also written to thesis/figures/pcpg/:
#   bells    the pg nodes, and the sums of bells closing in on sigmoid(z)
#   waves    the gauss-hermite nodes, and the sums of waves closing in on a bell
#   cf       the area under wave x p(z), and that it is the cf
#   summary  the area under sigmoid(z) x p(z) rebuilt from 1, 2, 4, all bells
#
# it uses the best rules found in the demos: golub-welsch for the outer pg
# points (with the sign flip) and gauss-hermite for the inner hs points.
# everything only depends on x through z, so it is shown in z. in pcpg the
# model gives the cf of the missing features at s = -u w_m, which is the same as
# the cf of the missing part of z at -u. for the gmm z is a mixture of
# gaussians, z | k ~ N(m_k, v_k) given the observed features, which also gives
# exact values to compare against.
# the plotting is in utils/plots.py

import os
import random
from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from utils.data import DataManager
from utils.gmm import GMM
from utils.quadrature import pg_quadrature
from utils.pg_density import pg_pdf
from utils.plots import (plot_pcpg_step_by_step, plot_thesis_bells, plot_thesis_cf,
                         plot_thesis_summary, plot_thesis_waves)

OUT_DIR = "figures/pcpg"
os.makedirs(OUT_DIR, exist_ok=True)
# the thesis figures go directly into the latex project
THESIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../thesis/figures/pcpg")
os.makedirs(THESIS_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

# setup, the same as in main.py
SEED         = 42
N_FEATURES   = 15
K            = 4
MISSING_RATE = 0.6
TEST_IDX     = 5

# outer points (golub-welsch, at most ~19) and inner points (gauss-hermite)
N_OUTER = 19
N_INNER = 60

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

# distribution of z given the observed features: p(k | x_obs), m_k and v_k.
# the sign flip: sigmoid(z) = 1 - sigmoid(-z). if z is mostly positive, compute
# with -z (-w) and mirror at the end, so the bells only have to cover the left
# side, where the factor e^{z/2} shrinks errors instead of amplifying them
with torch.no_grad():
    FLIP = bool(px.z_mean(w, x_partial) > 0)
    W_USED = -w if FLIP else w
    MIX, M, V = (t.double().numpy() for t in px.z_moments(W_USED, x_partial))

# ── the pieces ────────────────────────────────────────────────────────────────

# density of z and its components
def pz_components(z):
    return MIX[:, None] * np.exp(-(z - M[:, None])**2 / (2 * V[:, None])) / np.sqrt(2 * np.pi * V[:, None])

# characteristic function of z, E[e^{isz}]. this is what the model provides
def phi(s):
    s = np.asarray(s, dtype=float)
    return (MIX * np.exp(1j * np.multiply.outer(s, M) - np.multiply.outer(s**2, V) / 2)).sum(axis=-1)

# exact E[sigmoid(z)] for the flipped z, by adaptive quadrature per component
def exact_answer():
    vals = [quad(lambda t: expit(m + np.sqrt(v) * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi),
                 -12, 12, points=[np.clip(-m / np.sqrt(v), -12, 12)], epsabs=0, epsrel=1e-13, limit=200)[0]
            for m, v in zip(M, V)]
    return MIX @ np.array(vals)

# exact inner value E[e^{z/2 - gamma z^2/2}] for every gamma, closed form for
# the gaussian components (the whole pg + hs integrand averaged over z)
def exact_inner(gammas):
    d = 1 + np.multiply.outer(gammas, V)
    return (MIX * d**-0.5 * np.exp((M + V / 4 - np.multiply.outer(gammas, M**2)) / (2 * d))).sum(axis=-1)

@lru_cache(maxsize=None)
def outer_points(n):
    return pg_quadrature(n, "golub_welsch")

def inner_points(n):
    t, omega = np.polynomial.hermite.hermgauss(n)
    return t, omega

# the inner sum for every outer point: hs with gauss-hermite, the waves averaged
# over z through the cf,
#   e^{1/(8 gamma)} 1/sqrt(pi) sum_i omega_i e^{i u_i / (2 gamma)} phi(-u_i),  u_i = sqrt(2 gamma) t_i
# (the formula of the thesis, with the observed part of z inside phi)
def inner_estimate(gammas, n_inner):
    t, omega = inner_points(n_inner)
    u = np.sqrt(2 * gammas)[:, None] * t                          # (n_outer, n_inner)
    waves = np.exp(1j * u / (2 * gammas[:, None])) * phi(-u)
    return (np.exp(1 / (8 * gammas)) * (waves @ omega) / np.sqrt(np.pi)).real

# the whole estimator: outer sum over the inner values, then the flip undone
def pcpg(n_outer, n_inner):
    gammas, rhos = outer_points(n_outer)
    est = 0.5 * rhos @ inner_estimate(gammas, n_inner)
    return 1 - est if FLIP else est

def answer():
    return 1 - exact_answer() if FLIP else exact_answer()

# ── the figure ────────────────────────────────────────────────────────────────

# the explanations below the panels. they are typeset with latex, the math
# without spaces so that wrapping the lines never splits it. <...> are filled
# in by plot_step_by_step
TEXTS = dict(
    goal_heading=r"Compute $\mathbb{E}_{p(x^m\mid{}x^o)}\big[\sigma(w^\top{}x)\big]$",
    goal_title=r"$\displaystyle=\mathbb{E}_{p(z\mid{}x^o)}\big[\sigma(z)\big]=\int\sigma(z)\,p(z)\,dz$",
    goal=r"The prediction only sees the missing features $x^m$ through the score "
         r"$z=w_o^\top{}x^o+w_m^\top{}x^m$, a single number. As $x^m$ is uncertain, so is $z$, "
         r"with density $p(z\mid{}x^o)$, from now on $p(z)$ (grey, area 1). Averaging over $x^m$ "
         r"is therefore averaging over $z$: the yellow area under $\sigma(z)\times{}p(z)$. The "
         r"model can not compute it directly, so PCPG builds it from pieces it can.<flip>",
    pg=r"P\'olya-Gamma: $\sigma(z)$ can be written as a weighted sum of bells "
       r"$e^{z/2-\gamma{}z^2/2}$ of different widths $\gamma$. Indigo: the sums of the bells "
       r"with the largest weights, closing in on $\sigma(z)$. But where do the bells come from?",
    pg_origin=r"From the P\'olya-Gamma identity: $\sigma(z)=\tfrac{1}{2}e^{z/2}$ times the "
              r"average of $e^{-\gamma{}z^2/2}$ over widths $\gamma$ from the P\'olya-Gamma "
              r"distribution PG(1,0) (grey). PCPG replaces this average by a weighted sum over "
              r"a few well-chosen widths (lines, height = weight, Gauss quadrature): every line "
              r"is one bell. The dark indigo one is followed in the next steps.",
    hs=r"Hubbard-Stratonovich: every bell, here the one of the dark indigo width, can be written "
       r"as a weighted sum of cosine waves $\cos(u(z-c))$ around its centre $c$, with "
       r"different frequencies $u$. Summed, slowest first, they pile up where the bell is "
       r"and cancel everywhere else. But where do the waves come from?",
    hs_origin=r"From the Hubbard-Stratonovich identity: the bell, up to a constant "
              r"$e^{-\gamma(z-c)^2/2}$, is the average of the waves $\cos(u(z-c))$ over "
              r"frequencies $u$ from a Gaussian $\mathcal{N}(0,\gamma)$ (grey). Again PCPG "
              r"replaces this average by a weighted sum over a few well-chosen frequencies "
              r"(lines, Gauss-Hermite): every line is one wave.",
    wave_area=r"The area under wave $\times\,p(z)$ is the wave's average over the uncertain "
              r"$z$. A slow wave changes little over $p(z)$ and keeps most of the area, a fast "
              r"one goes up and down within $p(z)$ and its area cancels to about 0.",
    cf=r"This area does not need $p(z)$ itself: for the wave of frequency $u$ it is "
       r"$\mathrm{Re}[e^{-iuc}\phi(u)]$, with $\phi$ the characteristic function of the "
       r"density model, which the model can compute. One call to $\phi$ gives the areas of "
       r"all waves of the followed bell (dots, circled: the slow and the fast wave on the left). This is "
       r"the only place the model is used.",
    waves=r"Every panel: the sum of the slowest $k$ waves from step 2 (coral line, scaled) and "
          r"its area times $p(z)$ (coral, parts below 0 count negative), against the bell "
          r"(indigo, dashed) and its area. As the sum of waves turns into the bell, its area "
          r"turns into the bell's area. The fast waves add almost nothing.",
    bells=r"Every panel: the sum of the bells with the largest weights from step 1 (indigo "
          r"line, scaled) and its area times $p(z)$ (indigo), against $\sigma(z)$ (yellow, "
          r"dashed) and the goal area. The area of every bell comes from step 4. With few "
          r"bells the sum is far from $\sigma(z)$ and the area too small, with all of them it "
          r"is the PCPG estimate <estimate> (exact <answer>).<flip_end>",
)

# the partial sums of waves and bells, and the slow and fast wave of step 2
WAVE_KS    = [1, 2, 4, N_INNER // 2]
BELL_KS    = [1, 2, 4, N_OUTER]
SHOW_WAVES = [0, 9]

# everything the figures need, as a dict
def pieces():
    gammas, rhos = outer_points(N_OUTER)
    j = int(np.argmax(rhos))
    g, c = gammas[j], 1 / (2 * gammas[j])

    # the followed outer point. its bell e^{z/2 - gamma z^2/2} is
    # e^{1/(8 gamma)} e^{-gamma (z - c)^2/2}, and the inner points +-t_i give the
    # wave e^{1/(8 gamma)} 2 omega_i / sqrt(pi) cos(u_i (z - c)), u_i = sqrt(2 gamma) t_i,
    # slowest first. the average of cos(u (z - c)) over z is Re(e^{-iuc} phi(u)),
    # i.e. it comes from the model's cf
    t, omega = inner_points(N_INNER)
    pos = t > 0
    order = np.argsort(t[pos])
    u_pairs = np.sqrt(2 * g) * t[pos][order]
    w_waves = np.exp(1 / (8 * g)) * 2 * omega[pos][order] / np.sqrt(np.pi)
    wave_avg_fn = lambda uu: (np.exp(-1j * uu * c) * phi(uu)).real
    wave_avgs = wave_avg_fn(u_pairs)

    # the weighted bells (one per outer point) and their areas, from the waves
    bells = 0.5 * rhos[:, None] * np.exp(Z / 2 - np.outer(gammas, Z**2) / 2)
    inner = inner_estimate(gammas, N_INNER)

    gamma_axis = np.linspace(0, 2.5, 600)
    return dict(z=Z, pz=pz_components(Z).sum(axis=0), answer=exact_answer(), gamma=g, j=j,
                gammas=gammas, rhos=rhos, gamma_axis=gamma_axis, pg_density=pg_pdf(gamma_axis),
                u_all=np.sqrt(2 * g) * t, u_weights=omega / np.sqrt(np.pi),
                by_weight=np.argsort(-rhos), bells=bells, bell_parts=0.5 * rhos * inner,
                bell=np.exp(Z / 2 - g * Z**2 / 2), bell_avg=w_waves @ wave_avgs,
                u_pairs=u_pairs, cosines=np.cos(np.outer(u_pairs, Z - c)),
                waves=w_waves[:, None] * np.cos(np.outer(u_pairs, Z - c)),
                wave_parts=w_waves * wave_avgs, wave_avg_fn=wave_avg_fn, wave_avgs=wave_avgs,
                estimate=0.5 * rhos @ inner,
                wave_ks=WAVE_KS, bell_ks=BELL_KS, show_waves=SHOW_WAVES)

def plot_step_by_step():
    d = pieces()
    fill = {"<flip>": (r" Here $z$ is mostly positive, so $-w$ is used (the sign flip) and the "
                       r"result is mirrored at the end.") if FLIP else "",
            "<flip_end>": f" Mirrored back (flip): {1 - d['estimate']:.4f}." if FLIP else "",
            "<estimate>": f"{d['estimate']:.4f}", "<answer>": f"{d['answer']:.4f}"}
    texts = {}
    for key, text in TEXTS.items():
        for placeholder, value in fill.items():
            text = text.replace(placeholder, value)
        texts[key] = text
    plot_pcpg_step_by_step(d, texts, f"{OUT_DIR}/step_by_step.pdf")

# the four figures of the methods chapter, into figures/pcpg and straight into
# the thesis
def thesis_figures():
    d = pieces()
    for out_dir in (OUT_DIR, THESIS_DIR):
        plot_thesis_bells(d, f"{out_dir}/bells.pdf")
        plot_thesis_waves(d, f"{out_dir}/waves.pdf")
        plot_thesis_cf(d, f"{out_dir}/cf.pdf")
        plot_thesis_summary(d, f"{out_dir}/summary.pdf")

if __name__ == "__main__":
    plot_step_by_step()
    thesis_figures()
