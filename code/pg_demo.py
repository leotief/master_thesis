import numpy as np
from polyagamma import random_polyagamma
import matplotlib.pyplot as plt
import os
from pcpg import pg_gw_quadrature, pg_nnls_quadrature

os.makedirs("figures", exist_ok=True)

ns = [10, 50, 100, 500]
mc_ns = [10, 100, 500]
quad_ns = [3, 5, 10, 15]

# ── core functions ──────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mc_estimate(x_vals, pg_samples):
    integrand = np.exp(-np.outer(pg_samples, x_vals**2) / 2)  
    return 0.5 * np.exp(x_vals / 2) * integrand.mean(axis=0)

def quad_estimate(x_vals, nodes, weights):
    integrand = np.exp(-np.outer(nodes, x_vals**2) / 2)        
    return 0.5 * np.exp(x_vals / 2) * (weights[:, None] * integrand).sum(axis=0)

def _pg_pdf(omega, n_terms=50):
    omega = np.asarray(omega, dtype=float)
    result = np.zeros_like(omega)
    for k in range(n_terms):
        result += (-1)**k * (2*k + 1) * np.exp(-(2*k + 1)**2 * np.pi**2 * omega / 2)
    return 2 * np.pi * result

# ── plots ─────────────────────────────────────────────────────────────────────

def plot_pg_samples(ns, out_path="figures/pg_samples.pdf"):
    rng   = np.random.default_rng(0)
    omega = np.linspace(0.01, 2.5, 500)
    pdf   = _pg_pdf(omega)

    fig, axes = plt.subplots(1, len(ns), figsize=(3.5 * len(ns), 4), sharey=True)
    for ax, n in zip(axes, ns):
        samples = random_polyagamma(1.0, 0.0, size=n, random_state=rng)
        ax.hist(samples, bins=max(10, n // 5), density=True,
                color="#90CAF9", alpha=0.8, label=f"{n} draws")
        ax.plot(omega, pdf, color="#1565C0", lw=2, label="exact PDF")
        ax.set_xlim(0, 2.5)
        ax.set_xlabel(r"$\omega$", fontsize=12)
        ax.set_title(f"$N = {n}$", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.20)

    axes[0].set_ylabel("density", fontsize=12)
    fig.suptitle(r"Draws from $\mathrm{PG}(1,\,0)$", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_sigmoid_comparison(mc_ns, quad_ns,
                            out_path="figures/pg_sigmoid_comparison.pdf"):
    x     = np.linspace(-30, 30, 1000)
    truth = sigmoid(x)
    rng   = np.random.default_rng(0)
    n_runs = 1
    
    mc_curves = {}
    for n in mc_ns:
        samples = random_polyagamma(1.0, np.zeros(n * n_runs), random_state=rng)
        mc_curves[n] = [
            mc_estimate(x, samples[i*n:(i+1)*n]) for i in range(n_runs)
        ]

    gw_curves = {n: quad_estimate(x, *pg_gw_quadrature(n)) for n in quad_ns}

    nodes_nnls, weights_nnls = pg_nnls_quadrature()
    est_nnls = quad_estimate(x, nodes_nnls, weights_nnls)

    colors_mc = plt.cm.Blues(np.linspace(0.35, 0.90, len(mc_ns)))
    colors_gw = plt.cm.Reds( np.linspace(0.35, 0.90, len(quad_ns)))

    fig, (ax_fn, ax_err) = plt.subplots(1, 2, figsize=(11, 4))

    ax_fn.plot(x, truth, color="0.15", lw=2, zorder=10, label="exact $\\sigma(x)$")

    for color, n in zip(colors_mc, mc_ns):
        for i, curve in enumerate(mc_curves[n]):
            ax_fn.plot(x, curve, color=color, alpha=0.4, lw=0.9,
                       label=f"MC $N={n}$" if i == 0 else None)
            ax_err.plot(x, np.abs(curve - truth), color=color, alpha=0.4, lw=0.9,
                        label=f"MC $N={n}$" if i == 0 else None)

    for color, n in zip(colors_gw, quad_ns):
        ax_fn.plot(x, gw_curves[n], color=color, lw=1.8, label=f"GW $n={n}$")
        ax_err.plot(x, np.abs(gw_curves[n] - truth), color=color, lw=1.8,
                    label=f"GW $n={n}$")

    ax_fn.plot(x, est_nnls, color="#2E7D32", lw=2, ls="--",
               label=f"NNLS ({len(nodes_nnls)} nodes)")
    ax_err.plot(x, np.abs(est_nnls - truth), color="#2E7D32", lw=2, ls="--",
                label=f"NNLS ({len(nodes_nnls)} nodes)")

    ax_fn.set_xlabel("$x$", fontsize=11)
    ax_fn.set_ylabel("$\\sigma(x)$", fontsize=11)
    ax_fn.set_ylim(-0.05, 1.15)
    ax_fn.set_title("PG-sigmoid identity: MC vs quadrature", fontsize=11)
    ax_fn.legend(fontsize=8)

    ax_err.set_xlabel("$x$", fontsize=11)
    ax_err.set_ylabel("|error|", fontsize=11)
    ax_err.set_yscale("log")
    ax_err.set_title("Absolute error vs exact $\\sigma(x)$", fontsize=11)
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot_pg_samples(ns)
    plot_sigmoid_comparison(mc_ns, quad_ns)
