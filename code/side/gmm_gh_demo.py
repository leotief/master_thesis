# the gmm gauss-hermite baseline (GMMGaussHermite in utils/baselines.py), on the
# setup of main.py (same data, gmm, classifier and test point), as one figure,
# figures/gmm_gh/step_by_step.pdf, in the style of the pcpg one:
#   goal     the area under sigmoid(z) x p(z)
#   step 1   p(z) is a mixture of gaussians, so the area is a sum of component areas
#   step 2   the area of a component is a weighted sum of sigmoid at the
#            gauss-hermite points, and where the points come from
#   step 3   why it works: n points give the exact area under the polynomial
#            of degree n - 1 through sigmoid at the points (1, 2, 4, all points)
#   step 4   the component areas add up to the goal (1, 2, 3, all components)
#
# for the gmm, z = w^T x given the observed features is a mixture of gaussians,
# z | k ~ N(m_k, v_k) with weights p(k | x_obs), so every component area is a 1d
# gaussian integral. unlike pcpg this needs the model to give the distribution
# of z directly, not only its cf.
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
from utils.plots import plot_gmm_gh_step_by_step

OUT_DIR = "figures/gmm_gh"
os.makedirs(OUT_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

# setup, the same as in main.py
SEED         = 42
N_FEATURES   = 15
K            = 4
MISSING_RATE = 0.6
TEST_IDX     = 5

# gauss-hermite points per component, as N_GH_BASELINE in main.py
N_NODES = 10

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

# distribution of z given the observed features: p(k | x_obs), m_k and v_k
with torch.no_grad():
    MIX, M, V = (t.double().numpy() for t in px.z_moments(w, x_partial))

# ── the pieces ────────────────────────────────────────────────────────────────

# the weighted components pi_k N(z; m_k, v_k), one row per component
def pz_components(z):
    return MIX[:, None] * np.exp(-(z - M[:, None])**2 / (2 * V[:, None])) / np.sqrt(2 * np.pi * V[:, None])

# gauss-hermite for the weight e^{-t^2}, with the 1/sqrt(pi) folded into the
# weights so they sum to 1
def gh_rule(n):
    t, h = np.polynomial.hermite.hermgauss(n)
    return t, h / np.sqrt(np.pi)

# E[sigmoid(z) | k] for every component with n points, z_i = m_k + sqrt(2 v_k) t_i
def component_areas(n):
    t, h = gh_rule(n)
    return expit(M[:, None] + np.sqrt(2 * V)[:, None] * t) @ h

# exact E[sigmoid(z) | k] of one component, by adaptive quadrature
def exact_component_area(m, v):
    s = np.sqrt(v)
    return quad(lambda t: expit(m + s * t) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi), -12, 12,
                points=[np.clip(-m / s, -12, 12)], epsabs=0, epsrel=1e-13, limit=200)[0]

# ── the figure ────────────────────────────────────────────────────────────────

# the explanations below the panels, typeset with latex, the math without
# spaces so that wrapping the lines never splits it. <...> are filled in by
# plot_step_by_step
TEXTS = dict(
    goal=r"Some features are missing, so the score $z=w^\top{}x$ is uncertain: $p(z)$, from "
         r"the density model (grey, area 1). The expected prediction "
         r"$\mathbb{E}[\sigma(z)\mid{}x_{\mathrm{obs}}]$ is the yellow area under "
         r"$\sigma(z)\times{}p(z)$. For a Gaussian mixture model this area can be computed "
         r"directly, with one small quadrature per component.",
    mix=r"For the GMM, $z$ given the observed features is again a mixture: in component $k$ "
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
         r"With all components it is the GMM Gauss-Hermite estimate <estimate> "
         r"(exact <answer>).",
)

# the numbers of points of step 3
NODE_KS = [1, 2, 4, N_NODES]

def pieces():
    order = np.argsort(-MIX)
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

    comp_exact = np.array([exact_component_area(m, v) for m, v in zip(M, V)])
    comp_gh = component_areas(N_NODES)
    components = pz_components(Z)
    return dict(z=Z, pz=components.sum(axis=0), components=components, mix=MIX, order=order, j=j,
                m_j=m_j, s_j=s_j, t=t, weights=h, nodes=nodes, sigma_nodes=expit(nodes),
                comp_exact=comp_exact, comp_gh=comp_gh, answer=MIX @ comp_exact,
                estimate=MIX @ comp_gh, node_ks=NODE_KS, polys=polys, poly_nodes=poly_nodes,
                poly_areas=poly_areas)

def plot_step_by_step():
    d = pieces()
    fill = {"<n>": str(N_NODES), "<estimate>": f"{d['estimate']:.4f}", "<answer>": f"{d['answer']:.4f}"}
    texts = {}
    for key, text in TEXTS.items():
        for placeholder, value in fill.items():
            text = text.replace(placeholder, value)
        texts[key] = text
    plot_gmm_gh_step_by_step(d, texts, f"{OUT_DIR}/step_by_step.pdf")

if __name__ == "__main__":
    plot_step_by_step()
