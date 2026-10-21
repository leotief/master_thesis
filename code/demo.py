import random
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from baselines import MCBaseline
from data import DataManager
from gmm import GMM
from pcpg import PCPGMC, PCPGMCHermite, PCPGQuadrature
from utils import *

pcpg_ratio   = 5   
n_pg         = 100
n_gauss      = n_pg * pcpg_ratio

n_gauss_quad = 20
n_pg_quad    = 20

K = 4
n_features = 15
missing_rate = 0.6

# ── setup ─────────────────────────────────────────────────────────────────────

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dm = DataManager('sk_class', samples=10_000, features=n_features,
                 test_size=0.2, random_state=seed)
x_train, _, y_train, _ = dm.get_data()
train_dataloader, _    = dm.get_dataloaders(batch_size=64)

x_train_tensor, _ = train_dataloader.dataset.tensors
px = GMM(device, K=K, means_init=x_train_tensor[:K].to(device), n_features=n_features).to(device)
px.fit(train_dataloader)

clf = LogisticRegression()
clf.fit(x_train, y_train)
w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)

x_partial = dm.get_test_missing(missing_rate=(1-missing_rate), idx=5, device=device)
print(f"Missing features: {torch.isnan(x_partial).sum().item()} / {n_features}\n")

mc        = MCBaseline(px)
pcpg      = PCPGMC(px)
pcpg_gh   = PCPGMCHermite(px, n_gauss_quad=n_gauss_quad)
pcpg_quad = PCPGQuadrature(px, n_pg_quad=n_pg_quad, n_gauss_quad=n_gauss_quad)

# ── evaluation ──────────────────────────────────────────────────────────────────

def eval_estimators():
    x_complete = dm.get_test(idx=5, device=device)
    true_val   = float(torch.sigmoid(w @ x_complete))
    print(f"{'True σ(wᵀx)':<28}  {true_val:.4f}")

    def evaluate(name, fn, n_runs=10):
        t0   = time.perf_counter()
        vals = [fn() for _ in range(n_runs)]
        ms   = (time.perf_counter() - t0) / n_runs * 1000
        print(f"{name:<28}  {np.mean(vals):.4f} ± {np.std(vals):.5f}   ({ms:.1f} ms/run, n={n_runs})")

    evaluate("GT (MC 10M)",     lambda: mc(w, x_partial, n_samples=10_000_000))
    evaluate("MC",              lambda: mc(w, x_partial, n_samples=n_pg * n_gauss))
    evaluate("PCPG-MC",         lambda: pcpg(w, x_partial, n_pg=n_pg, n_gauss=n_gauss))
    evaluate("PCPG-GH",         lambda: pcpg_gh(w, x_partial, n_pg=n_pg))
    evaluate("PCPG-Quadrature", lambda: pcpg_quad(w, x_partial), n_runs=1)

def allocation_analysis():
    def alloc(N, ratio):
        n_pg    = max(10, int(np.sqrt(N / ratio)))
        n_gauss = max(10, N // n_pg)
        return n_pg, n_gauss

    alloc_budgets = [10**i for i in range(4, 7)]
    results_alloc = {name: {"means": [], "stds": []} for name in
                     ["MC", "PCPG inner-heavy", "PCPG balanced", "PCPG outer-heavy"]}

    for N in alloc_budgets:
        pg_ih, g_ih = alloc(N, 10)
        pg_ba, g_ba = alloc(N,  1)
        pg_oh, g_oh = alloc(N, .1)
        for name, fn in [
            ("MC",                lambda: mc(w, x_partial, n_samples=N)),
            ("PCPG inner-heavy",  lambda: pcpg(w, x_partial, n_pg=pg_ih, n_gauss=g_ih)),
            ("PCPG balanced",     lambda: pcpg(w, x_partial, n_pg=pg_ba, n_gauss=g_ba)),
            ("PCPG outer-heavy",  lambda: pcpg(w, x_partial, n_pg=pg_oh, n_gauss=g_oh)),
        ]:
            vals = [fn() for _ in range(30)]
            results_alloc[name]["means"].append(np.mean(vals))
            results_alloc[name]["stds"].append(np.std(vals))

    plot_convergence(
        results_alloc, alloc_budgets,
        filename=f'pcpg_allocation_convergence_{n_features}feat.png',
        title=rf"PCPG-MC budget allocation  ({n_features} features, mean $\pm$ 1 std)",
    )

def variance_analysis():
    convergence_budgets = [10**i for i in range(3, 8)]
    results             = {name: {"means": [], "stds": []} for name in ["MC", "PCPG", "PCPG-GH"]}

    for N in convergence_budgets:
        n_pg_pcpg    = max(1, int(np.sqrt(N / pcpg_ratio)))
        n_gauss_pcpg = max(1, N // n_pg_pcpg)
        n_pg_gh      = max(1, N // n_gauss_quad)
        for name, fn in [
            ("MC",      lambda: mc(w, x_partial, n_samples=N)),
            ("PCPG",    lambda: pcpg(w, x_partial, n_pg=n_pg_pcpg, n_gauss=n_gauss_pcpg)),
            ("PCPG-GH", lambda: pcpg_gh(w, x_partial, n_pg=n_pg_gh)),
        ]:
            vals = [fn() for _ in range(10)]
            results[name]["means"].append(np.mean(vals))
            results[name]["stds"].append(np.std(vals))

    plot_convergence(results, convergence_budgets,
                     filename=f'convergence_{n_features}.png',
                     title=rf'Estimator convergence ({int(missing_rate * n_features)}/{n_features} features, mean $\pm$ 1 std)')
    
    n_reps = 1000
    
    mc1_vals    = [mc(w, x_partial, n_samples=1) for _ in range(n_reps)]
    pcpg11_vals = [pcpg(w, x_partial, n_pg=1, n_gauss=1) for _ in range(n_reps)]
    gh1_vals    = [pcpg_gh(w, x_partial, n_pg=1) for _ in range(n_reps)]

    sigma2_mc = np.var(mc1_vals)
    sigma2_B  = 4 * np.var(gh1_vals)
    sigma2_A  = 4 * np.var(pcpg11_vals) - sigma2_B

    print_variance_components(sigma2_mc, sigma2_B, sigma2_A, pcpg11_vals)
    plot_variance_analysis(sigma2_mc, sigma2_B, sigma2_A, convergence_budgets, results,
                           n_gauss_quad=n_gauss_quad, pcpg_ratio=pcpg_ratio,
                           filename=f'variance_{n_features}.png',
                           title=f'Formula vs measured variance  ({int(missing_rate * n_features)}/{n_features} features)')

# ── run ───────────────────────────────────────────────────────────────────────

eval_estimators()
# allocation_analysis()
variance_analysis()
