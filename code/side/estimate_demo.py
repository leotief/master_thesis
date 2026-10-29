# how the expected prediction E[sigmoid(w^T x) | x_obs] of an incomplete point
# is computed, step by step, on the 2d toy problem of utils/toy.py (the
# classifier w*, the density model and the point of the training step by step).
# the pcpg figure sets the goal and computes it, the mc and the gmm
# gauss-hermite figures only show how they reach the same goal differently.
# all three in figures/estimate_overview/:
#   pcpg.pdf
#     setup       the classifier sigmoid(w^T x), the density model p(x), and a
#                 point with a missing feature: a line of possible points
#     goal        its prediction, the average of sigmoid(w^T x) along the line,
#                 weighted by p(x_mis | x_obs): in z, the area under sigmoid(z) x p(z)
#     step 1      pg: where the bells come from, and how they add up to sigmoid(z)
#     step 2      hs: where the waves come from, and how they add up to a bell
#     step 3      the area under wave x p(z) comes from the model's cf
#     step 4      the wave areas add up to the area under a bell (1, 2, all waves)
#     step 5      the bell areas add up to the goal (1, 2, all bells)
#   mc.pdf (MCBaseline in utils/baselines.py)
#     step 1      draw the missing feature from the model, which gives samples of z
#     step 2      the prediction sigmoid(z_i) at every sample, the estimate is their average
#     step 3      the average is the area under sigmoid(z) x the histogram of the
#                 samples, which closes in on the goal (10, 100, 10000 samples)
#     step 4      how fast: the error is random and shrinks like 1 / sqrt(n)
#   gmm_gh.pdf (MixtureGaussHermite in utils/baselines.py)
#     step 1      p(z) is a mixture of gaussians, so the area is a sum of component areas
#     step 2      the area of a component is a weighted sum of sigmoid at the
#                 gauss-hermite points, and where the points come from
#     step 3      why it works: n points give the exact area under the polynomial
#                 of degree n - 1 through sigmoid at the points (1, 2, all points)
#     step 4      the component areas add up to the goal (1, 2, all components)
# and the four figures of the methods chapter, {bells,waves,cf,summary}.pdf, also
# written to thesis/figures/pcpg/. these stay on the setup of main.py (same
# data, gmm, classifier and test point), which the thesis text describes:
#   bells    the pg nodes, and the sums of bells closing in on sigmoid(z)
#   waves    the gauss-hermite nodes, and the sums of waves closing in on a bell
#   cf       the area under wave x p(z), and that it is the cf
#   summary  the area under sigmoid(z) x p(z) rebuilt from 1, 2, 4, all bells
#
# pcpg uses the best rules found in the demos: golub-welsch for the outer pg
# points (with the sign flip) and gauss-hermite for the inner hs points.
# everything only depends on x through z, so it is shown in z. in pcpg the
# model gives the cf of the missing features at s = -u w_m, which is the same as
# the cf of the missing part of z at -u. for the gmm z is a mixture of
# gaussians, z | k ~ N(m_k, v_k) given the observed features, which also gives
# exact values to compare against. mc needs nothing from the model but samples,
# drawn as in GMM.sample: a component k from p(k | x_obs), then the missing
# features from its gaussian, here keeping the component of every sample so it
# can be coloured.
# the plotting is in utils/plots.py

import os
import random
import sys
from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# experiment_utils.py (for utils/toy.py) is one directory up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from side.utils.data import classification, hide, loader
from side.utils.gmm import GMM
from side.utils.pg_density import pg_pdf
from side.utils.plots import (plot_gmm_gh_step_by_step, plot_mc_step_by_step, plot_pcpg_step_by_step,
                              plot_thesis_bells, plot_thesis_cf, plot_thesis_summary, plot_thesis_waves)
from side.utils.quadrature import pg_quadrature
from side.utils.toy import px as toy_px, toy, w_star, x_point

OUT_DIR = "figures/estimate"
os.makedirs(OUT_DIR, exist_ok=True)
# the thesis figures go directly into the latex project
THESIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../thesis/figures/pcpg")
os.makedirs(THESIS_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

SEED = 42

# the setup of main.py, for the thesis figures
N_FEATURES   = 15
K            = 4
MISSING_RATE = 0.6
TEST_IDX     = 5

# pcpg: outer points (golub-welsch, at most ~19) and inner points (gauss-hermite)
N_OUTER = 19
N_INNER = 60

# mc: the samples of the steps: the scatter and histogram of step 1, the
# samples of step 2, the progression of step 3, and the runs of step 4
N_SCATTER   = 2000
N_SHOW      = 20
SAMPLE_KS   = [10, 100, 10_000]
N_MAX       = 10_000
N_RUNS_SHOW = 5
N_RUNS      = 500

# gmm gauss-hermite: points per component, as N_GH_BASELINE in main.py
N_NODES = 10

Z = np.linspace(-12, 12, 1200)

# ── the distribution of z ─────────────────────────────────────────────────────

# z = w^T x given the observed features: for the gmm a mixture of gaussians,
# z | k ~ N(m_k, v_k) with weights p(k | x_obs). all three methods only need
# this (pcpg its cf, mc samples of it, gauss-hermite the components)
class Score:
    def __init__(self, px, w, x_obs):
        with torch.no_grad():
            self.mix, self.m, self.v = (t.double().numpy() for t in px.z_mixture(w, x_obs))

    # the weighted components pi_k N(z; m_k, v_k), one row per component
    def components(self, z):
        m, v = self.m[:, None], self.v[:, None]
        return self.mix[:, None] * np.exp(-(z - m)**2 / (2 * v)) / np.sqrt(2 * np.pi * v)

    # characteristic function of z, E[e^{isz}]. this is what the model provides
    def phi(self, s):
        s = np.asarray(s, dtype=float)
        return (self.mix * np.exp(1j * np.multiply.outer(s, self.m) - np.multiply.outer(s**2, self.v) / 2)).sum(axis=-1)

    # exact E[f(z) | k] of every component, by adaptive quadrature
    def component_means(self, f):
        return np.array([quad(lambda t: f(m + np.sqrt(v) * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi), -12, 12,
                              points=[np.clip(-m / np.sqrt(v), -12, 12)], epsabs=0, epsrel=1e-13, limit=200)[0]
                         for m, v in zip(self.m, self.v)])

    # exact E[f(z)]
    def mean(self, f):
        return self.mix @ self.component_means(f)

    # the z where p(z) has its mass (components with weight > 1%), for the
    # panels with areas
    def window(self):
        keep, s = self.mix > 0.01, np.sqrt(self.v)
        return np.floor((self.m - 4 * s)[keep].min()), np.ceil((self.m + 4 * s)[keep].max())

# the sign flip: sigmoid(z) = 1 - sigmoid(-z). if z is mostly positive, compute
# with -z (-w) and mirror at the end, so the bells only have to cover the left
# side, where the factor e^{z/2} shrinks errors instead of amplifying them
def flipped_score(px, w, x_obs):
    with torch.no_grad():
        flip = bool(px.z_mean(w, x_obs) > 0)
    return flip, Score(px, -w if flip else w, x_obs)

# ── pcpg ──────────────────────────────────────────────────────────────────────

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
def inner_estimate(score, gammas, n_inner):
    t, omega = inner_points(n_inner)
    u = np.sqrt(2 * gammas)[:, None] * t                          # (n_outer, n_inner)
    waves = np.exp(1j * u / (2 * gammas[:, None])) * score.phi(-u)
    return (np.exp(1 / (8 * gammas)) * (waves @ omega) / np.sqrt(np.pi)).real

# the partial sums of waves and bells, the followed bell (0 = the one with the
# largest weight, 1 the next, ...), and the slow and fast wave of step 3. for
# the test point of the thesis, and for the toy point. there p(z) lies left of
# 0 and is narrow, so it follows a narrower bell whose centre 1/(2 gamma) is
# close to p(z), so that bell x p(z) is not just the bell's tail, and its waves
# only cancel at higher frequencies
WAVE_KS        = [1, 2, 4, N_INNER // 2]
BELL_KS        = [1, 2, 4, N_OUTER]
SHOW_WAVES     = [0, 9]
TOY_FOLLOWED   = 2
TOY_SHOW_WAVES = [0, 8]

# everything the pcpg figures need, as a dict, for the (flipped) score
def pcpg_pieces(score, show_waves=SHOW_WAVES, followed=0):
    gammas, rhos = outer_points(N_OUTER)
    j = int(np.argsort(-rhos)[followed])
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
    wave_avg_fn = lambda uu: (np.exp(-1j * uu * c) * score.phi(uu)).real
    wave_avgs = wave_avg_fn(u_pairs)

    # the weighted bells (one per outer point) and their areas, from the waves
    bells = 0.5 * rhos[:, None] * np.exp(Z / 2 - np.outer(gammas, Z**2) / 2)
    inner = inner_estimate(score, gammas, N_INNER)

    gamma_axis = np.linspace(0, 2.5, 600)
    return dict(z=Z, pz=score.components(Z).sum(axis=0), answer=score.mean(expit), window=score.window(),
                gamma=g, j=j, gammas=gammas, rhos=rhos, gamma_axis=gamma_axis, pg_density=pg_pdf(gamma_axis),
                u_all=np.sqrt(2 * g) * t, u_weights=omega / np.sqrt(np.pi),
                by_weight=np.argsort(-rhos), bells=bells, bell_parts=0.5 * rhos * inner,
                bell=np.exp(Z / 2 - g * Z**2 / 2), bell_avg=w_waves @ wave_avgs,
                u_pairs=u_pairs, cosines=np.cos(np.outer(u_pairs, Z - c)),
                waves=w_waves[:, None] * np.cos(np.outer(u_pairs, Z - c)),
                wave_parts=w_waves * wave_avgs, wave_avg_fn=wave_avg_fn, wave_avgs=wave_avgs,
                estimate=0.5 * rhos @ inner,
                wave_ks=WAVE_KS, bell_ks=BELL_KS, show_waves=show_waves)

# ── monte carlo ───────────────────────────────────────────────────────────────

def mc_pieces(px, w, x_obs):
    score = Score(px, w, x_obs)
    with torch.no_grad():
        mis = torch.isnan(x_obs)
        z_obs = float(w[~mis] @ x_obs[~mis])
        mis = mis.numpy()
        means = px.means.double().numpy()
        stds = np.sqrt(np.exp(px.log_vars.double().numpy()) + 1e-6)
        w_mis = w.double().numpy()[mis]

    # n samples of the missing features as in GMM.sample, with their components,
    # and their scores z = w_obs^T x_obs + w_mis^T x_mis
    def draw(rng, n):
        k = rng.choice(len(score.mix), size=n, p=score.mix / score.mix.sum())
        x_mis = means[k][:, mis] + stds[k][:, mis] * rng.standard_normal((n, mis.sum()))
        return k, x_mis, z_obs + x_mis @ w_mis

    rng = np.random.default_rng(SEED)
    answer = score.mean(expit)
    var = score.mean(lambda z: expit(z)**2) - answer**2

    # step 1: the samples on the line of the point, and their scores
    k, x_mis, zs = draw(rng, N_SCATTER)

    # steps 2 and 3: one run, the first n samples of it. step 4: more runs,
    # the first one is this one
    _, _, z_run = draw(rng, N_MAX)
    runs = np.vstack([z_run] + [draw(rng, N_MAX)[2] for _ in range(N_RUNS - 1)])
    running = np.cumsum(expit(runs), axis=1) / np.arange(1, N_MAX + 1)

    ns = np.unique(np.logspace(0, np.log10(N_MAX), 40).astype(int))
    rmse = np.sqrt(((running[:, ns - 1] - answer)**2).mean(axis=0))

    components = score.components(Z)
    return dict(z=Z, pz=components.sum(axis=0), window=score.window(), components=components, mix=score.mix,
                order=np.argsort(-score.mix), comp_means=means, comp_stds=stds, k=k, x_mis=x_mis[:, 0],
                z_samples=zs, z_run=z_run, n_show=N_SHOW, sample_ks=SAMPLE_KS, running=running[:N_RUNS_SHOW],
                ns=ns, rmse=rmse, answer=answer, var=var, estimate=running[0, -1])

# ── gmm gauss-hermite ─────────────────────────────────────────────────────────

# gauss-hermite for the weight e^{-t^2}, with the 1/sqrt(pi) folded into the
# weights so they sum to 1
def gh_rule(n):
    t, h = np.polynomial.hermite.hermgauss(n)
    return t, h / np.sqrt(np.pi)

# the numbers of points of step 3
NODE_KS = [1, 2, 4, N_NODES]

def gmm_gh_pieces(score):
    mix, M, V = score.mix, score.m, score.v
    order = np.argsort(-mix)
    j = int(order[0])
    m_j, s_j = M[j], np.sqrt(V[j])
    t, h = gh_rule(N_NODES)
    nodes = m_j + np.sqrt(2) * s_j * t

    # step 3: the interpolating polynomial through sigma at n points. gauss
    # quadrature integrates it exactly, so its area is the n-point sum
    polys, poly_nodes, poly_areas = [], [], []
    for n in NODE_KS:
        tn, hn = gh_rule(n)
        pts = m_j + np.sqrt(2) * s_j * tn
        poly = np.polynomial.Polynomial.fit(pts, expit(pts), n - 1) if n > 1 else \
            np.polynomial.Polynomial([expit(pts[0])])
        polys.append(poly(Z))
        poly_nodes.append(pts)
        poly_areas.append(hn @ expit(pts))

    # E[sigmoid(z) | k] for every component, exact and with N_NODES points,
    # z_i = m_k + sqrt(2 v_k) t_i
    comp_exact = score.component_means(expit)
    comp_gh = expit(M[:, None] + np.sqrt(2 * V)[:, None] * t) @ h
    components = score.components(Z)
    return dict(z=Z, pz=components.sum(axis=0), window=score.window(), components=components, mix=mix,
                order=order, j=j, m_j=m_j, s_j=s_j, t=t, weights=h, nodes=nodes, sigma_nodes=expit(nodes),
                comp_exact=comp_exact, comp_gh=comp_gh,
                answer=mix @ comp_exact, estimate=mix @ comp_gh, node_ks=NODE_KS, polys=polys,
                poly_nodes=poly_nodes, poly_areas=poly_areas)

# ── the explanations ──────────────────────────────────────────────────────────

# the explanations below the panels. they are typeset with latex, the math
# without spaces so that wrapping the lines never splits it. <...> are filled
# in by fill
PCPG_TEXTS = dict(
    goal_heading=r"Compute $\mathbb{E}_{p(x^m\mid{}x^o)}\big[\sigma(w^\top{}x)\big]$",
    goal_title=r"$\displaystyle=\mathbb{E}_{p(z\mid{}x^o)}\big[\sigma(z)\big]=\int\sigma(z)\,p(z)\,dz$",
    setup=r"A logistic regression $\sigma(w^\top{}x)$ predicts the probability of label 1 (yellow, from "
          r"white for 0 to yellow for 1), with its boundary through the origin (dashed). It is the one trained in "
          r"the training step by step and is fixed here. A density model $p(x)$, a mixture of four Gaussians "
          r"(grey contours), is fit beforehand: it describes where points lie and how the two features go "
          r"together. Now a point comes in with $x_1$ observed and $x_2$ missing. It is not a point but a line "
          r"of possible points, and the density model says where on the line it probably is: "
          r"$p(x_2\mid{}x_1)$ (grey, here two bumps).",
    goal=r"The prediction for the point is the average of $\sigma(w^\top{}x)$ along its line, weighted by "
         r"$p(x_2\mid{}x_1)$. It only sees the missing features through the score $z=w_o^\top{}x^o+w_m^\top{}x^m$, a "
         r"single number, however many features are missing. Here it changes linearly along the line, so its "
         r"density $p(z\mid{}x^o)$, from now on $p(z)$ (grey, area 1), is $p(x_2\mid{}x_1)$ "
         r"stretched<mirrored>, and the prediction is the yellow area under $\sigma(z)\times{}p(z)$: <answer>. "
         r"Mean imputation would take $\sigma$ at one point of the line and get <mean_prediction>. The model can "
         r"not compute the area directly, so PCPG builds it from pieces it can.<flip>",
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
       r"the <n_waves> waves of the followed bell (dots, circled: the slow and the fast wave on the left). This is "
       r"the only place the model is used.",
    waves=r"Every panel: the sum of the slowest $k$ waves from step 2 (coral line, scaled) and "
          r"its area times $p(z)$ (coral, parts below 0 count negative), against the bell "
          r"(indigo, dashed) and its area. As the sum of waves turns into the bell, its area "
          r"turns into the bell's area. The fast waves add almost nothing.",
    bells=r"Every panel: the sum of the bells with the largest weights from step 1 (indigo "
          r"line, scaled) and its area times $p(z)$ (indigo), against $\sigma(z)$ (yellow, "
          r"dashed) and the goal area. The area of every bell comes from step 4. With few "
          r"bells the sum is far from $\sigma(z)$ and the area too small, with <n_bells> bells it "
          r"is the PCPG estimate <estimate> (exact <answer>).<flip_end>",
)

MC_TEXTS = dict(
    draw=r"Monte Carlo reaches the goal of the PCPG step by step, the area under $\sigma(z)\times{}p(z)$, "
         r"without computing anything about $p(z)$: it tries out random fill-ins of the missing feature. Draw $n$ "
         r"fill-ins $x_{\mathrm{mis}}^{(i)}\sim{}p(x_{\mathrm{mis}}\mid{}x_{\mathrm{obs}})$ from the density "
         r"model: pick a component $k$ with weight $\pi_k=p(k\mid{}x_{\mathrm{obs}})$, then the missing feature "
         r"from its Gaussian (dots on the line of the point, coloured by component and spread sideways so they "
         r"can be seen; ellipses: the components, $2$ standard deviations).",
    scores=r"Every fill-in, together with the observed feature, gives one score "
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

GMM_GH_TEXTS = dict(
    mix=r"For a Gaussian mixture model the goal of the PCPG step by step, the area under "
        r"$\sigma(z)\times{}p(z)$, can be computed directly, with one small quadrature per component, "
        r"because $p(z)$ itself is known. Given the observed features, $z$ is again a mixture: in component $k$ "
        r"it is Gaussian, $z\mid{}k\sim\mathcal{N}(m_k,v_k)$, with weight "
        r"$\pi_k=p(k\mid{}x_{\mathrm{obs}})$, mean $m_k=w^\top\mu_{k}$ (observed features "
        r"filled in) and variance $v_k=\sum_{j\,\mathrm{missing}}w_j^2\sigma_{kj}^2$.",
    split=r"So the area splits into one area per component: "
          r"$\mathbb{E}[\sigma(z)]=\sum_k\pi_k\,\mathbb{E}[\sigma(z)\mid{}k]$, and each "
          r"$\mathbb{E}[\sigma(z)\mid{}k]$ is a 1D integral of $\sigma$ against a Gaussian. "
          r"The dark indigo component (largest weight) is followed in the next steps.",
    nodes=r"Gauss-Hermite replaces the integral of $\sigma(z)$ against the Gaussian of the "
          r"component by a weighted sum $\sum_i\frac{h_i}{\sqrt{\pi}}\sigma(z_i)$ over "
          r"<n> points $z_i$ (lines, height = weight). Only $\sigma$ is evaluated, the "
          r"Gaussian enters through where the points are and how much they weigh.",
    origin=r"The points $t_i$ and weights $h_i$ are fixed once, for the weight $e^{-t^2}$ "
           r"(grey), and moved to every component by $z_i=m_k+\sqrt{2v_k}\,t_i$. They are "
           r"chosen so that the sum is exact for every polynomial up to degree $2n-1$: "
           r"the points crowd where the Gaussian has its mass.",
    polys=r"Every panel: with $n$ points the sum is exactly the area under the polynomial "
          r"of degree $n-1$ through $\sigma$ at the points (coral line and dots, scaled) times "
          r"the Gaussian of the component (coral area), against $\sigma(z)$ (yellow, dashed) and its area. One "
          r"point is a constant, two a line. As the polynomial closes in on $\sigma$ where "
          r"the Gaussian has its mass, the area closes in on the component's area. Far out "
          r"the polynomial runs off, but there the Gaussian is about 0.",
    sums=r"Every panel: $\sigma(z)$ times the sum of the components with the largest weights "
         r"(dark indigo, then lighter, stacked) and its area $\sum_k\pi_k\,\mathbb{E}[\sigma(z)\mid{}k]$, "
         r"every component area from <n> Gauss-Hermite points, against the goal (yellow). "
         r"With <n_comp> components it is the GMM Gauss-Hermite estimate <estimate> "
         r"(exact <answer>).",
)

# the texts with every <placeholder> replaced by its value
def fill(texts, values):
    filled = {}
    for key, text in texts.items():
        for placeholder, value in values.items():
            text = text.replace(placeholder, value)
        filled[key] = text
    return filled

# ── the figures ───────────────────────────────────────────────────────────────

# the three step-by-step figures, on the point of utils/toy.py
def step_by_step():
    flip, score = flipped_score(toy_px, w_star, x_point)
    unflipped = Score(toy_px, w_star, x_point)
    pcpg = pcpg_pieces(score, TOY_SHOW_WAVES, TOY_FOLLOWED)
    mc, gh = mc_pieces(toy_px, w_star, x_point), gmm_gh_pieces(unflipped)

    mean_prediction = expit(toy["w_star"] @ np.array([toy["x1"], toy["mean_x2"]]))
    pcpg_values = {"<mean_prediction>": f"{mean_prediction:.4f}",
                   "<mirrored>": r" by $|w_2|$ and mirrored, as $w_2<0$" if toy["w_star"][1] < 0 else r" by $w_2$",
                   "<flip>": (r" Here $z$ is mostly positive, so $-w$ is used (the sign flip) and the "
                              r"result is mirrored at the end.") if flip else "",
                   "<flip_end>": f" Mirrored back (flip): {1 - pcpg['estimate']:.4f}." if flip else "",
                   "<estimate>": f"{pcpg['estimate']:.4f}", "<answer>": f"{pcpg['answer']:.4f}",
                   "<n_waves>": str(len(pcpg["waves"])), "<n_bells>": str(len(pcpg["bells"]))}
    plot_pcpg_step_by_step({**toy, **pcpg}, fill(PCPG_TEXTS, pcpg_values),
                           f"{OUT_DIR}/pcpg.pdf")

    mc_values = {"<n_show>": str(N_SHOW), "<n_runs_show>": str(N_RUNS_SHOW), "<n_runs>": str(N_RUNS),
                 "<n_max>": f"{N_MAX:,}".replace(",", r"\,"), "<var>": f"{mc['var']:.4f}",
                 "<estimate>": f"{mc['estimate']:.4f}", "<answer>": f"{mc['answer']:.4f}"}
    plot_mc_step_by_step({**toy, **mc}, fill(MC_TEXTS, mc_values), f"{OUT_DIR}/mc.pdf")

    gh_values = {"<n>": str(N_NODES), "<n_comp>": str(len(gh["mix"])), "<estimate>": f"{gh['estimate']:.4f}", "<answer>": f"{gh['answer']:.4f}"}
    plot_gmm_gh_step_by_step(gh, fill(GMM_GH_TEXTS, gh_values), f"{OUT_DIR}/gmm_gh.pdf")

# the four figures of the methods chapter, on the setup of main.py, into
# figures/estimate_overview and straight into the thesis
def thesis_figures():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    x_train, x_test, y_train, _ = classification(N_FEATURES, seed=SEED)
    px = GMM("cpu", K=K, means_init=torch.tensor(x_train[:K]), n_features=N_FEATURES)
    px.fit(loader(torch.tensor(x_train)))
    w = torch.tensor(LogisticRegression().fit(x_train, y_train).coef_[0], dtype=torch.float32)
    x_partial = torch.tensor(hide(x_test[TEST_IDX], MISSING_RATE, np.random.RandomState(SEED)))

    d = pcpg_pieces(flipped_score(px, w, x_partial)[1])
    for out_dir in (OUT_DIR, THESIS_DIR):
        plot_thesis_bells(d, f"{out_dir}/bells.pdf")
        plot_thesis_waves(d, f"{out_dir}/waves.pdf")
        plot_thesis_cf(d, f"{out_dir}/cf.pdf")
        plot_thesis_summary(d, f"{out_dir}/summary.pdf")

if __name__ == "__main__":
    step_by_step()
    thesis_figures()
