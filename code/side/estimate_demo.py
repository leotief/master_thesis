# how the expected prediction E[sigmoid(w^T x) | x_obs] of an incomplete point
# is computed, step by step, on the 2d toy problem of utils/toy.py .
# the cfi figure sets the goal and computes it, the mc and the gmm
# gauss-hermite figures only show how they reach the same goal differently.
# the two figures of the methods chapter, {waves,cf}.pdf, also written to
# thesis/figures/cfi/. these are on a 15-feature gmm setup (data, gmm,
# classifier and one test point with missing features)

# the plotting is in utils/plots.py

import os
import random
import sys

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# experiment_utils.py (for utils/toy.py) and pclr.py are one directory up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pclr import S_MAX, cfi
from side.utils.data import classification, hide, loader
from side.utils.gmm import GMM
from side.utils.plots import (plot_cfi_step_by_step, plot_gmm_gh_step_by_step, plot_mc_step_by_step,
                              plot_thesis_cf, plot_thesis_waves)
from side.utils.toy import px as toy_px, toy, w_star, x_point

OUT_DIR = "figures/estimate"
os.makedirs(OUT_DIR, exist_ok=True)
# the thesis figures go directly into the latex project
THESIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../thesis/figures/cfi")
os.makedirs(THESIS_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

SEED = 42

# the setup of the thesis figures, and its frequencies: z is wider there, so the
# step is smaller and more frequencies are needed
N_FEATURES      = 15
K               = 4
MISSING_RATE    = 0.6
TEST_IDX        = 5
THESIS_FREQ     = 64
THESIS_WAVE_KS  = [1, 4, 16, THESIS_FREQ]

# cfi: the number of frequencies, the partial sums of waves of steps 2 and 4,
# the slow and the fast wave of step 3 (indices of the frequencies), and the z
# of the threshold in step 1
N_FREQ     = 16
WAVE_KS    = [1, 2, 4, N_FREQ]
SHOW_WAVES = [0, 5]
Z0         = 1.5

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

# the score axis, a wider one for the sums of waves of step 2 (where they
# repeat), and the axis of the threshold
Z      = np.linspace(-12, 12, 1200)
Z_WIDE = np.linspace(-24, 24, 2400)
T      = np.linspace(-8, 8, 800)

# ── the distribution of z ─────────────────────────────────────────────────────

# z = w^T x given the observed features: for the gmm a mixture of gaussians,
# z | k ~ N(m_k, v_k) with weights p(k | x_obs). all three methods only need
# this (cfi its cf, mc samples of it, gauss-hermite the components)
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

    # mean and standard deviation of z, for cfi's step
    def mean_std(self):
        mean = self.mix @ self.m
        return mean, np.sqrt(self.mix @ (self.v + self.m**2) - mean**2)

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

# ── cfi ───────────────────────────────────────────────────────────────────────

# the logistic density, the density of the threshold L with P(L <= z) = sigmoid(z)
def logistic_density(t):
    return expit(t) * (1 - expit(t))

# everything the cfi figures need, as a dict. the same formula as pclr.cfi:
#   E[sigmoid(z)] = 1/2 + h (E[z] / (2 pi) + sum_k Im phi(s_k) / sinh(pi s_k)),  s_k = k h
# with the step h chosen from the mean and spread of z
def cfi_pieces(score, n_freq=N_FREQ, wave_ks=WAVE_KS, show_waves=SHOW_WAVES):
    mean, std = score.mean_std()
    h = min(np.pi / (abs(mean) + 6 * std + 1), S_MAX / n_freq)
    s = h * np.arange(1, n_freq + 1)
    weights = h / np.sinh(np.pi * s)

    # step 1: the density of z - L, p(z) smeared out by the logistic, by a sum
    # over the grid, and its area right of 0
    pz, dz = score.components(Z).sum(axis=0), Z[1] - Z[0]
    p_diff = (pz[None, :] * logistic_density(Z[None, :] - Z[:, None])).sum(axis=1) * dz
    spread = np.sqrt(std**2 + np.pi**2 / 3)

    # steps 2 and 4: the weighted waves h sin(s_k z) / sinh(pi s_k), and the
    # rest of the sum, the constant 1/2 and the term at frequency 0
    def waves_on(z):
        return weights[:, None] * np.sin(np.outer(s, z)), 0.5 + h * z / (2 * np.pi)
    waves, base = waves_on(Z)
    waves_wide, base_wide = waves_on(Z_WIDE)

    # step 3: the area under every wave x p(z), from the cf
    im_phi_fn = lambda ss: score.phi(ss).imag
    im_phi = im_phi_fn(s)
    base_part = 0.5 + h * mean / (2 * np.pi)

    return dict(z=Z, pz=pz, answer=score.mean(expit), window=score.window(), h=h, s=s,
                t=T, logistic=logistic_density(T), z0=Z0, p_diff=p_diff, diff_area=p_diff[Z > 0].sum() * dz,
                diff_window=(np.floor(mean - 4 * spread), np.ceil(mean + 4 * spread)),
                z_wide=Z_WIDE, pz_wide=score.components(Z_WIDE).sum(axis=0), waves_wide=waves_wide,
                base_wide=base_wide, waves=waves, base=base, wave_ks=wave_ks,
                sines=np.sin(np.outer(s, Z)), im_phi=im_phi, im_phi_fn=im_phi_fn, show_waves=show_waves,
                base_part=base_part, wave_parts=weights * im_phi, estimate=base_part + weights @ im_phi)

# ── monte carlo ───────────────────────────────────────────────────────────────

def mc_pieces(px, w, x_obs):
    score = Score(px, w, x_obs)
    with torch.no_grad():
        mis = torch.isnan(x_obs)
        z_obs = float(w[~mis] @ x_obs[~mis])
        mis = mis.numpy()
        means = px.means.double().numpy()
        stds = np.sqrt(px.variances().double().numpy())
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
CFI_TEXTS = dict(
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
         r"not compute the area directly, so CFI builds it from pieces it can.",
    threshold=r"The sigmoid is a probability: for a random threshold $L$ from the logistic distribution (grey), "
              r"$\sigma(z)=P(L\le{}z)$, the yellow area left of $z$, here for $z=<z0>$. So the classifier says "
              r"label 1 when the score beats a random threshold.",
    difference=r"With a missing feature the score is random too, and the prediction is the probability that "
               r"it beats the threshold, $P(z-L>0)$. The difference $z-L$ (indigo) is $p(z)$ (grey) smeared "
               r"out by $L$, and its area right of 0 (yellow) is again <answer>. What is left is the "
               r"probability that a random number is positive.",
    sines=r"That probability comes from sine waves (Gil-Pelaez): a step from 0 to 1 is a sum of waves "
          r"$\sin(sz)$, and smoothing the step by $L$ gives "
          r"$\sigma(z)=\tfrac{1}{2}+\int_0^\infty\frac{\sin(sz)}{\sinh(\pi{}s)}ds$. CFI takes <n_freq> "
          r"frequencies $s_k=kh$, $h=<h>$, and adds the waves with weight $h/\sinh(\pi{}s_k)$ to the "
          r"line $\tfrac{1}{2}+hz/(2\pi)$ of frequency 0. Coral: the sums of the slowest waves, closing in on "
          r"$\sigma(z)$. Only within $|z|<\pi/h=<half>$: beyond it the sum climbs on to the next step of a "
          r"staircase instead of staying at 0 or 1, so $h$ is chosen per point such that $p(z)$ (grey) lies "
          r"within.",
    weights=r"Where do the weights come from? The sharp step needs every frequency equally. The threshold "
            r"multiplies the weight of frequency $s$ by $\frac{\pi{}s}{\sinh(\pi{}s)}$, the characteristic "
            r"function of $L$, which falls off exponentially: fast waves hardly count, and beyond $s=12$ they "
            r"are below double precision (lines: the chosen frequencies).",
    wave_area=r"The area under wave $\times\,p(z)$ is the wave's average over the uncertain "
              r"$z$. A slow wave changes little over $p(z)$ and keeps most of its area, a fast "
              r"one goes up and down within $p(z)$ and its area cancels to about 0.",
    cf=r"This area does not need $p(z)$ itself: for the wave of frequency $s$ it is $\mathrm{Im}\,\phi(s)$, "
       r"with $\phi$ the characteristic function of $z$, which the density model can compute; the observed "
       r"features only add a phase. One batched call gives the areas of all <n_freq> waves (dots, circled: "
       r"the slow and the fast wave on the left). This is the only place the model is used.",
    sums=r"Every panel: the line of frequency 0 plus the slowest $k$ waves from step 2 (coral line, scaled) "
         r"and its area times $p(z)$ (coral, parts below 0 count negative), against $\sigma(z)$ (yellow, "
         r"dashed) and the goal area. The area of every wave comes from step 3. The fast waves add almost "
         r"nothing, and with <n_freq> waves it is the CFI estimate <estimate> (exact <answer>).",
)

MC_TEXTS = dict(
    draw=r"Monte Carlo reaches the goal of the CFI step by step, the area under $\sigma(z)\times{}p(z)$, "
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
    mix=r"For a Gaussian mixture model the goal of the CFI step by step, the area under "
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
    score = Score(toy_px, w_star, x_point)
    d, mc, gh = cfi_pieces(score), mc_pieces(toy_px, w_star, x_point), gmm_gh_pieces(score)
    # the same value as the method itself
    with torch.no_grad():
        assert abs(d["estimate"] - float(cfi(toy_px, w_star, x_point, N_FREQ))) < 1e-5

    mean_prediction = expit(toy["w_star"] @ np.array([toy["x1"], toy["mean_x2"]]))
    cfi_values = {"<mean_prediction>": f"{mean_prediction:.4f}",
                  "<mirrored>": r" by $|w_2|$ and mirrored, as $w_2<0$" if toy["w_star"][1] < 0 else r" by $w_2$",
                  "<z0>": f"{Z0:g}", "<h>": f"{d['h']:.3f}", "<half>": f"{np.pi / d['h']:.1f}",
                  "<n_freq>": str(N_FREQ), "<estimate>": f"{d['estimate']:.4f}", "<answer>": f"{d['answer']:.4f}"}
    plot_cfi_step_by_step({**toy, **d}, fill(CFI_TEXTS, cfi_values), f"{OUT_DIR}/cfi.pdf")

    mc_values = {"<n_show>": str(N_SHOW), "<n_runs_show>": str(N_RUNS_SHOW), "<n_runs>": str(N_RUNS),
                 "<n_max>": f"{N_MAX:,}".replace(",", r"\,"), "<var>": f"{mc['var']:.4f}",
                 "<estimate>": f"{mc['estimate']:.4f}", "<answer>": f"{mc['answer']:.4f}"}
    plot_mc_step_by_step({**toy, **mc}, fill(MC_TEXTS, mc_values), f"{OUT_DIR}/mc.pdf")

    gh_values = {"<n>": str(N_NODES), "<n_comp>": str(len(gh["mix"])), "<estimate>": f"{gh['estimate']:.4f}", "<answer>": f"{gh['answer']:.4f}"}
    plot_gmm_gh_step_by_step(gh, fill(GMM_GH_TEXTS, gh_values), f"{OUT_DIR}/gmm_gh.pdf")

# the two figures of the methods chapter, into figures/estimate and straight
# into the thesis
def thesis_figures():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    x_train, x_test, y_train, _ = classification(N_FEATURES, seed=SEED)
    px = GMM("cpu", K=K, means_init=torch.tensor(x_train[:K]), n_features=N_FEATURES)
    px.fit(loader(torch.tensor(x_train)))
    w = torch.tensor(LogisticRegression().fit(x_train, y_train).coef_[0], dtype=torch.float32)
    x_partial = torch.tensor(hide(x_test[TEST_IDX], MISSING_RATE, np.random.RandomState(SEED)))

    d = cfi_pieces(Score(px, w, x_partial), THESIS_FREQ, THESIS_WAVE_KS)
    for out_dir in (OUT_DIR, THESIS_DIR):
        plot_thesis_waves(d, f"{out_dir}/waves.pdf")
        plot_thesis_cf(d, f"{out_dir}/cf.pdf")

if __name__ == "__main__":
    step_by_step()
    thesis_figures()
