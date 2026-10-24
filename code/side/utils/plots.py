# plotting helpers, all in one style, see code/PLOTTING.md
#
# the plot functions only save to figures/ and close the figure, they never
# call plt.show() or print, so the calling script decides what to output.
# plot_data, plot_gmm_results and plot_classifier_results only make sense for
# 2d data (e.g. moons), the plots after them are for the side demos

import functools
import os

import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import torch

os.makedirs('figures', exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────

# latex fonts (latin modern, like the thesis) at the sizes of the thesis, so a
# figure made THESIS_WIDTH wide is included without scaling and its text has
# the size of a caption
THESIS_WIDTH = 5.51  # \textwidth of the thesis, 398pt
STYLE = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "figure.titlesize": 10,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "0.3", "axes.labelcolor": "0.1",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.color": "0.3", "ytick.color": "0.3",
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
    "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.prop_cycle": matplotlib.cycler(color=["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
                                                "#56B4E9"]),
}

# colours (okabe-ito, colour blind safe). the roles are the same in every
# figure: ACCENT the one highlighted piece or the method we use, GOAL the target
# area, SLOW and FAST a contrasting pair, TARGET the exact curve (a thick light
# band under the estimates), DENSITY the fill of a density in the background.
# families of curves use sequential maps (Purples for bells, Blues for waves
# and sample sizes), darker = more pieces
ACCENT = "#D55E00"
GOAL = "#009E73"
SLOW, FAST = "#0072B2", "#E69F00"
TARGET = "0.82"
DENSITY = "0.88"

# n shades of blue, light to dark, as an (n, 4) rgba array
def blues(n):
    return plt.cm.Blues(np.linspace(0.4, 0.95, n))

# the experiment figures go to figures/<folder>/ and directly into the thesis
THESIS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../thesis/figures")

# plot(*args, path) for figures/<folder>/<name>.pdf and the same in the thesis
def save_figure(plot, folder, name, *args):
    for root in ("figures", THESIS_FIGURES):
        os.makedirs(f"{root}/{folder}", exist_ok=True)
        plot(*args, f"{root}/{folder}/{name}.pdf")

# runs the plot function inside the style
def styled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return fn(*args, **kwargs)
    return wrapper

# the exact curve: a thick light band, the estimates are drawn thin on top
def plot_target(ax, x, y, label=None):
    ax.plot(x, y, color=TARGET, lw=4, solid_capstyle="butt", zorder=0, label=label)

# a thin zero line
def zero_line(ax):
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)

# ── 2d checks ─────────────────────────────────────────────────────────────────

# scatter of the first two features, coloured by label if given
@styled
def plot_data(X, y=None, title=None, xlabel='Feature 1', ylabel='Feature 2'):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    if y is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=6, lw=0)
        fig.colorbar(scatter, ax=ax, label='label')
    else:
        ax.scatter(X[:, 0], X[:, 1], color=SLOW, alpha=0.7, s=6, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', 'datalim')
    filename = title.replace(' ', '_') + '.pdf' if title else 'plot_data.pdf'
    fig.savefig(f'figures/{filename}')
    plt.close(fig)

# visual check of the gmm fit (2d): left the data coloured by its most likely
# component, right samples drawn from the gmm. if the fit is good both look alike
@styled
def plot_gmm_results(gmm_model, X, title=None, num_samples=1000):
    with torch.no_grad():
        # log p(x | k) + log pi_k for every point and component
        X_tensor = torch.tensor(X, dtype=torch.float32).to(gmm_model.device)
        variances = torch.exp(gmm_model.log_vars) + 1e-6
        log_probs = torch.distributions.Normal(gmm_model.means, variances.sqrt()).log_prob(X_tensor.unsqueeze(1)).sum(dim=-1)
        weighted_log_probs = log_probs + torch.log_softmax(gmm_model.pi, dim=-1)
        # most likely component for every point
        clusters = torch.argmax(weighted_log_probs, dim=1).cpu().numpy()

    samples = gmm_model.sample(num_samples).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(THESIS_WIDTH, 2.7), sharex=True, sharey=True)
    axes[0].scatter(X[:, 0], X[:, 1], c=clusters, cmap='tab10', alpha=0.6, s=5, lw=0)
    axes[0].set_title("data, coloured by most likely component")
    axes[1].scatter(samples[:, 0], samples[:, 1], color=ACCENT, alpha=0.5, s=5, lw=0)
    axes[1].set_title("samples from the GMM")
    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_aspect('equal', 'datalim')
    axes[0].set_ylabel('Feature 2')
    if title:
        fig.suptitle(title)

    filename = title.replace(' ', '_') + '_combined.pdf' if title else 'gmm_combined.pdf'
    fig.savefig(f'figures/{filename}')
    plt.close(fig)

# visual check of a 2d classifier: left the predicted probability over the plane
# with the data on top, right the decision boundary and which points are wrong
@styled
def plot_classifier_results(clf, X, y, title="Logistic Regression"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # evaluate the classifier on a grid for the contour plot
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid_points)[:, 1].reshape(xx.shape)

    y_pred = clf.predict(X)
    correct = (y_pred == y)

    fig, axes = plt.subplots(1, 2, figsize=(THESIS_WIDTH, 2.7))

    contour = axes[0].contourf(xx, yy, probs, cmap='RdBu_r', levels=20, alpha=0.8)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu_r', edgecolor='0.2', lw=0.3, s=6)
    axes[0].set_title("predicted probability")
    fig.colorbar(contour, ax=axes[0], label=r'$p(y = 1 \mid x)$')

    axes[1].contour(xx, yy, probs, levels=[0.5], colors='0.1', linewidths=1)
    axes[1].scatter(X[correct, 0], X[correct, 1], color=GOAL, label='correct', alpha=0.6, s=5, lw=0)
    axes[1].scatter(X[~correct, 0], X[~correct, 1], color=ACCENT, label='wrong', marker='x', s=12, lw=0.8)
    axes[1].set_title("decision boundary")
    axes[1].legend(loc="upper right")

    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
    axes[0].set_ylabel('Feature 2')
    fig.suptitle(title)

    filename = f"{title.lower().replace(' ', '_')}_results.pdf"
    fig.savefig(f'figures/{filename}')
    plt.close(fig)

# ── quadrature comparison plots (pg_identity_demo.py, hs_identity_demo.py) ────

# left: approximations against the exact function, right: absolute error on a
# log scale. curves is a list of (estimate, label, colour, linestyle, alpha),
# every estimate evaluated on x. truth is the exact function on x
@styled
def plot_function_and_error(x, truth, curves, out_path, title=None,
                            truth_label=r"exact $\sigma(x)$", ylim=(-0.15, 1.15), xlabel="$x$"):
    fig, (ax_fn, ax_err) = plt.subplots(1, 2, figsize=(THESIS_WIDTH, 2.4))
    plot_target(ax_fn, x, truth, truth_label)
    for est, label, color, ls, alpha in curves:
        ax_fn.plot(x, est, color=color, ls=ls, alpha=alpha, label=label)
        # floor at 1e-17 so errors that are exactly 0 don't break the log axis
        ax_err.plot(x, np.maximum(np.abs(est - truth), 1e-17), color=color, ls=ls, alpha=alpha, lw=0.9)

    ax_fn.set_xlabel(xlabel)
    ax_fn.set_ylim(*ylim)
    ax_fn.set_title("approximation")
    # below both panels, so it covers no curve
    fig.legend(*ax_fn.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=3)

    ax_err.set_xlabel(xlabel)
    ax_err.set_ylabel(r"$|\mathrm{error}|$")
    ax_err.set_yscale("log")
    ax_err.set_title("absolute error")
    if title:
        fig.suptitle(title, y=1.04)
    fig.savefig(out_path)
    plt.close(fig)

# error against number of nodes, log-log, one panel per entry of panels, all
# with the same y axis. panels is a list of (title, lines), lines a list of
# (ns, errors, label, colour, linestyle). an entry with a single n (a method
# that picks its own number of nodes) is drawn as a star instead of a line.
# xticks labels these node counts instead of only the powers of 10
@styled
def plot_error_vs_nodes(panels, out_path, suptitle=None, xticks=None,
                        xlabel="number of nodes", ylabel="error"):
    fig, axes = plt.subplots(1, len(panels), figsize=(max(0.75 * THESIS_WIDTH, 2.9 * len(panels)), 2.5),
                             sharey=True, squeeze=False)
    for ax, (title, lines) in zip(axes[0], panels):
        for ns, errs, label, color, ls in lines:
            if len(ns) == 1:
                ax.plot(ns, errs, "*", color=color, ms=7, label=label)
            else:
                ax.plot(ns, errs, color=color, ls=ls, marker="o", ms=2.5, label=label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        if xticks is not None:
            ax.set_xticks(xticks, labels=[str(n) for n in xticks])
            ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(fontsize=6.5)
    axes[0, 0].set_ylabel(ylabel)
    if suptitle:
        fig.suptitle(suptitle, y=1.05)
    fig.savefig(out_path)
    plt.close(fig)

# ── pg identity plots (pg_identity_demo.py) ───────────────────────────────────

# the pg identity step by step, one panel per step. step 1 is drawn twice:
# above on a linear gamma axis with the density p(gamma), below on a log axis
# with the density of log(gamma), p(gamma) * gamma, so small points are visible
# too. nodes and weights are the picked points, z is the z axis, bells the
# weighted bells w_j e^{-gamma_j z^2/2} (one row per node), exact_avg the exact
# average sech(z/2), estimate the resulting sigmoid estimate
@styled
def plot_pg_identity_steps(gamma_lin, density_lin, gamma_log, density_log, nodes, weights,
                           z, bells, exact_avg, estimate, out_path, title=None):
    colors = plt.cm.Purples(np.linspace(0.4, 0.95, len(nodes)))
    fig = plt.figure(figsize=(2 * THESIS_WIDTH, 2.8))
    grid = fig.add_gridspec(2, 4, hspace=0.9, wspace=0.3)
    axes = [None] + [fig.add_subplot(grid[:, col]) for col in range(1, 4)]

    for row, (gamma, density, log) in enumerate(((gamma_lin, density_lin, False),
                                                 (gamma_log, density_log, True))):
        ax = fig.add_subplot(grid[row, 0])
        shown = (nodes >= gamma[0]) & (nodes <= gamma[-1])
        w_max = weights[shown].max()
        ax.fill_between(gamma, density / density.max() * w_max, color=DENSITY, lw=0, label=r"$\mathrm{PG}(1,0)$")
        ax.vlines(nodes[shown], 0, weights[shown], colors=colors[shown], lw=1.1)
        ax.plot(nodes[shown], weights[shown], "o", color=plt.cm.Purples(0.8), ms=2)
        if log:
            ax.set_xscale("log")
        ax.set_xlim(gamma[0], gamma[-1])
        ax.set_ylim(0, 1.15 * w_max)
        ax.set_xlabel(r"$\gamma$ (log scale)" if log else r"$\gamma$")
        ax.set_ylabel("weight $w_j$")
        if not log:
            ax.set_title(r"1. pick points $\gamma_j$ with weights $w_j$")
            ax.legend(loc="upper right")

    ax = axes[1]
    for color, bell in zip(colors, bells):
        ax.plot(z, bell, color=color, lw=0.9)
    ax.set_xlabel("$z$")
    ax.set_title(r"2. every point gives a bell $w_j\, e^{-\gamma_j z^2/2}$")

    ax = axes[2]
    plot_target(ax, z, exact_avg, r"exact $\mathbb{E}[e^{-\gamma z^2/2}]$")
    ax.plot(z, bells.sum(axis=0), color=ACCENT, ls="--", label="sum of the bells")
    ax.set_ylim(-0.05, 1.4)
    ax.set_xlabel("$z$")
    ax.set_title("3. add the bells up")
    ax.legend(loc="upper right")

    ax = axes[3]
    plot_target(ax, z, 1 / (1 + np.exp(-z)), r"exact $\sigma(z)$")
    ax.plot(z, estimate, color=ACCENT, ls="--", label=r"$\frac{1}{2} e^{z/2}\,\cdot$ sum")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("$z$")
    ax.set_title(r"4. multiply by $\frac{1}{2} e^{z/2}$")
    ax.legend(loc="upper left")

    if title:
        fig.suptitle(title, y=1.04)
    fig.savefig(out_path)
    plt.close(fig)

# the points every method picks, one panel per method: every point as a
# vertical line whose height is its weight, on top of the density (grey,
# scaled to the height of the largest weight, only its shape matters).
# rows is a list of (panels, row_label), one figure row each, panels a list of
# (label, nodes, weights) or None for an empty cell. gamma is the x axis, on a
# log scale unless log is False
@styled
def plot_nodes_on_density(gamma, density, rows, out_path, title=None, log=True,
                          xlabel=r"$\gamma$ (log scale)"):
    n_cols = max(len(panels) for panels, _ in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.1 * n_cols, 1.45 * len(rows)),
                             squeeze=False, gridspec_kw=dict(hspace=0.7, wspace=0.15))
    for row_axes, (panels, row_label) in zip(axes, rows):
        for ax, panel in zip(row_axes, panels + [None] * (n_cols - len(panels))):
            if panel is None:
                ax.axis("off")
                continue
            label, nodes, weights = panel
            ax.fill_between(gamma, density / density.max() * weights.max(), color=DENSITY, lw=0)
            ax.vlines(nodes, 0, weights, color=ACCENT, lw=1)
            ax.plot(nodes, weights, "o", color=ACCENT, ms=2)
            if log:
                ax.set_xscale("log")
            ax.set_xlim(gamma[0], gamma[-1])
            ax.set_ylim(0, 1.1 * weights.max())
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.set_title(f"{label} ({len(nodes)} points)")
        # row label left of the first column, also when that cell is empty
        row_axes[0].text(-0.06, 0.5, row_label, transform=row_axes[0].transAxes,
                         rotation=90, ha="right", va="center")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    if title:
        fig.suptitle(title, y=1.1)
    fig.savefig(out_path)
    plt.close(fig)

# ── hs identity plots (hs_identity_demo.py) ───────────────────────────────────

# the hs identity step by step, one panel per step. u and density are the u
# axis and the N(0, 1) density on it, nodes and weights the picked points. t is
# the t axis, waves the weighted waves w_j cos(t u_j) (one row per node), exact
# the exact average e^{-t^2/2}
@styled
def plot_hs_identity_steps(u, density, nodes, weights, t, waves, exact, out_path, title=None):
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(nodes)))
    estimate = waves.sum(axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(2 * THESIS_WIDTH, 2.3), gridspec_kw=dict(wspace=0.3))

    ax = axes[0]
    ax.fill_between(u, density / density.max() * weights.max(), color=DENSITY, lw=0,
                    label=r"$\mathcal{N}(0,1)$ density")
    ax.vlines(nodes, 0, weights, colors=colors, lw=1.1)
    ax.plot(nodes, weights, "o", color=plt.cm.Blues(0.8), ms=2)
    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(0, 1.3 * weights.max())
    ax.set_xlabel("$u$")
    ax.set_ylabel("weight $w_j$")
    ax.set_title("1. pick points $u_j$ with weights $w_j$")
    ax.legend(loc="upper right")

    ax = axes[1]
    for color, wave in zip(colors, waves):
        ax.plot(t, wave, color=color, lw=0.8)
    zero_line(ax)
    ax.set_xlabel("$t$")
    ax.set_title(r"2. every point gives a wave $w_j \cos(t\, u_j)$")

    ax = axes[2]
    plot_target(ax, t, exact, r"exact $e^{-t^2/2}$")
    ax.plot(t, estimate, color=ACCENT, ls="--", label="sum of the waves")
    zero_line(ax)
    ax.set_ylim(-0.3, 1.25)
    ax.set_xlabel("$t$")
    ax.set_title("3. add the waves up")
    ax.legend(loc="upper right")

    ax = axes[3]
    ax.plot(t, np.maximum(np.abs(estimate - exact), 1e-17), color=ACCENT)
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_title("4. error: no cancelling at large $t$")

    if title:
        fig.suptitle(title, y=1.06)
    fig.savefig(out_path)
    plt.close(fig)


# ── pcpg step by step (pcpg_demo.py) ──────────────────────────────────────────

# what pcpg does, one row per step, with explanations below (texts, a dict):
#   goal     the area under sigmoid(z) x p(z)
#   step 1   pg: sums of bells adding up to sigmoid(z), and where the bells
#            come from (the chosen widths)
#   step 2   hs: sums of waves adding up to a bell, and where the waves come
#            from (the chosen frequencies)
#   step 3   the area under a wave, and for every wave from the model's cf
#   step 4   partial sums of waves and their areas x p(z) building up a bell
#   step 5   partial sums of bells and their areas x p(z) building up the goal
# every area is at true scale. d is a dict, see plot_step_by_step in
# pcpg_demo.py
#
# both step-by-step figures (this one and the gmm one) have a pastel look
# (catppuccin) with three colour families, each with one role in every panel:
# gold for sigmoid(z) and the goal area, indigo for the pieces that add up to
# the goal (the bells, the components; the followed one dark), coral for
# everything gauss-hermite (the waves, the frequencies, the points; only the
# fast wave of step 3 is a deep wine red, to tell it from the slow one). p(z)
# and the densities the points come from are grey. a target curve is a dashed
# line in its colour over a pale area (a light band where there are only
# lines), the sum closing in on it a solid line. every step has a badge, its
# heading, its panels and the explanation

# catppuccin: the neutrals of latte, and every colour as a pastel (mocha) and a
# deep (latte) version, lines and fills are mixed from them
_CAT = dict(text="#4c4f69", subtext="#6c6f85", overlay="#9ca0b0", surface="#ccd0da",
            grey="#dce0e8")
_PASTEL = dict(gold="#ffe794", indigo="#b4befe", coral="#f5a097", wine="#c97b95")
_DEEP = dict(gold="#d49a00", indigo="#3b4cca", coral="#e64553", wine="#7a1f3d")

# a tone of a colour family, from 0 (white) over 1 (pastel) to 2 (deep)
def _tone(family, t):
    white, pastel, deep = (np.array(matplotlib.colors.to_rgb(c))
                           for c in ("white", _PASTEL[family], _DEEP[family]))
    return tuple(white + t * (pastel - white) if t <= 1 else pastel + (t - 1) * (deep - pastel))

# n tones of a family, light to dark, for the sums of 1, 2, ... pieces
def _tones(family, n, lo=0.45, hi=2.0):
    return [_tone(family, t) for t in np.linspace(lo, hi, n)]

# the line and fill colour of every family
_LINE = {f: _tone(f, 1.55) for f in _PASTEL}
_FILL = {f: _tone(f, 1.0) for f in _PASTEL}

# the colour of a target: sigmoid(z) butter, the bell mist blue. the same
# colour for its band (only lines) and for its area under a target x p(z),
# in every step, so a target always looks the same
_BAND = dict(gold="#ffe794", indigo="#d6dcff")

# the thesis style, scaled up for the large step-by-step figures, with the
# catppuccin neutrals for text and axes
_BIG_STYLE = {**STYLE, "font.size": 14, "axes.labelsize": 13, "legend.fontsize": 11.5,
              "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.1,
              "xtick.major.width": 1.1, "ytick.major.width": 1.1, "xtick.major.size": 5,
              "ytick.major.size": 5, "lines.linewidth": 1.6, "legend.frameon": False,
              "axes.facecolor": "none", "axes.edgecolor": _CAT["text"],
              "axes.labelcolor": _CAT["text"], "xtick.color": _CAT["text"],
              "ytick.color": _CAT["text"], "text.color": _CAT["text"]}

def plot_pcpg_step_by_step(d, texts, out_path, title=None):
    with plt.rc_context(_BIG_STYLE):
        _plot_pcpg_step_by_step(d, texts, out_path, title)

# bold with usetex, where fontweight has no effect
def _bold(text):
    return r"{\bfseries\boldmath " + text + "}"

# a target curve over its pale area: a dashed line in its colour, on top of
# the solid estimates, so it stays visible where they match it
def _dashed(ax, x, y, family, label=None):
    ax.plot(x, y, color=_tone(family, 1.25), lw=2.4, ls=(0, (5, 2.5)), zorder=4, label=label)

# the partial sums of bells and waves among only lines (steps 1 and 2 of pcpg,
# and the bells and waves figures of the thesis): the target a pale wide band
# (butter for sigmoid(z), mist blue for the bell), the sums lines from light to
# near-black that also get heavier, so the build-up reads as one progression
_SIGMA_BAND, _BELL_BAND = _BAND["gold"], _BAND["indigo"]
_BELL_LINES = ["#aab4fb", "#6f7ef0", "#3343c0", "#161d5e"]
_WAVE_LINES = ["#f7b3a8", "#ef7d70", "#d93a4a", "#5c0f1a"]
_SUM_WIDTHS = (1.4, 1.9, 2.5, 3.2)

# a target band, the sums drawn on top
def _band(ax, x, y, color, label=None, lw=13):
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="butt", zorder=1, label=label)

# zero line, x label, room above the data for the legend, and a bold title
def _finish(ax, title, xlabel="score $z$", room=0.35, legend=True, fontsize=15):
    ax.axhline(0, color=_CAT["surface"], lw=0.9, zorder=0)
    ax.set_xlabel(xlabel, fontsize=12)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + room * (hi - lo))
    if legend:
        ax.legend(loc="upper left")
    if title:
        ax.set_title(_bold(title), fontsize=fontsize, pad=10, color=_CAT["text"])

# an explanation from x0 to x1 (figure fractions), wrapped to that width,
# centred below the axes ax
def _caption(fig, ax, text, x0=None, x1=None, fontsize=13):
    pos = ax.get_position()
    x0 = pos.x0 if x0 is None else x0
    x1 = pos.x1 if x1 is None else x1
    width, height = fig.get_size_inches()
    chars = int((x1 - x0) * width * 11.5 * 13 / fontsize)
    return fig.text((x0 + x1) / 2, pos.y0 - 0.62 / height, _wrap(text, chars), va="top", ha="center",
                    multialignment="center", fontsize=fontsize, color=_CAT["text"], linespacing=1.5)

# one explanation centred below a whole row of panels, over nearly its full
# width. several texts are joined into one
def _row_caption(fig, axes, *texts):
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    inset = 0.03 * (x1 - x0)
    return _caption(fig, axes[0], " ".join(texts), x0=x0 + inset, x1=x1 - inset)

# wraps latex text to lines of about width visible characters. the length of
# a word is counted without the latex commands, braces and $, so lines with a
# lot of math are not cut short
def _wrap(text, width):
    import re
    visible = lambda word: len(re.sub(r"\\[a-zA-Z]+|[{}$^_\\]", "", word)) or 1
    lines, line, length = [], [], 0
    for word in text.split():
        if line and length + 1 + visible(word) > width:
            lines.append(" ".join(line))
            line, length = [], 0
        length += visible(word) + (1 if line else 0)
        line.append(word)
    return "\n".join(lines + [" ".join(line)])

# the headings of the steps. rows are (badge, heading, axes): above the axes,
# centred, a small badge ("Step 1") on a short thin rule and the
# heading below it, in an optional size and lifted by an optional amount
# (inches). the title goes above the first badge
def _headings(fig, rows, title=None):
    width, height = fig.get_size_inches()
    x_left = min(ax.get_position().x0 for _, _, axes, *_ in rows for ax in axes)
    x_right = max(ax.get_position().x1 for _, _, axes, *_ in rows for ax in axes)
    centre = (x_left + x_right) / 2
    for badge, heading, axes, *opts in rows:
        size, lift = (opts + [23, 0][len(opts):])[:2]
        top = max(ax.get_position().y1 for ax in axes) + lift / height
        y_badge = top + 1.42 / height
        fig.add_artist(matplotlib.lines.Line2D([centre - 3 / width, centre + 3 / width], [y_badge] * 2,
                                               color=_CAT["surface"], lw=1, zorder=1))
        fig.text(centre, y_badge, _bold(r"\ " + badge.upper() + r"\ "), fontsize=14, color=_CAT["subtext"],
                 va="center", ha="center", zorder=3,
                 bbox=dict(boxstyle="square,pad=0.45", fc="white", ec="none"))
        fig.text(centre, top + 0.62 / height, _bold(heading), fontsize=size, color=_CAT["text"],
                 va="bottom", ha="center")
    if title:
        top = max(ax.get_position().y1 for ax in rows[0][2])
        fig.text(centre, top + 2.3 / height, _bold(title), fontsize=28, ha="center", va="bottom",
                 color=_CAT["text"])

# the panels of a progression in the step-by-step figures: 1, 2 and all pieces
def _first_two_and_all(seq):
    return [seq[0], seq[1], seq[-1]]

# the goal panel, the same in both figures
def _goal_panel(ax, z, pz, sigma, answer, title=r"The area under $\sigma(z) \times p(z)$", fontsize=15):
    ax.fill_between(z, pz, color=_CAT["grey"], lw=0, label="$p(z)$: area 1")
    ax.plot(z, sigma * pz.max(), color=_LINE["gold"], lw=1.6, ls=(0, (4, 2)), label=r"$\sigma(z)$ (scaled)")
    ax.fill_between(z, sigma * pz, color=_FILL["gold"], lw=0,
                    label=rf"$\sigma(z) \times p(z)$: area {answer:.4f}")
    ax.plot(z, sigma * pz, color=_LINE["gold"], lw=1.2)
    ax.set_xlim(z[0], z[-1])
    _finish(ax, title, room=0.6, fontsize=fontsize)

def _plot_pcpg_step_by_step(d, texts, out_path, title=None):
    z, pz = d["z"], d["pz"]
    sigma = 1 / (1 + np.exp(-z))
    by_weight = d["by_weight"]
    n_waves, n_bells = len(d["waves"]), len(d["bells"])
    plural = lambda k, unit, n: f"all {unit}s" if k == n else f"{k} {unit}{'s' if k > 1 else ''}"

    # 6 columns: the goal over the middle 4, the other rows two panels of 3 or
    # three panels of 2
    fig = plt.figure(figsize=(16, 50))
    grid = fig.add_gridspec(6, 6, height_ratios=[1, 1, 1, 1, 0.9, 0.9], hspace=1.95, wspace=0.6)
    ax_goal = fig.add_subplot(grid[0, 1:5])
    ax_pg, ax_pg_origin = fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6])
    ax_hs, ax_hs_origin = fig.add_subplot(grid[2, 0:3]), fig.add_subplot(grid[2, 3:6])
    ax_wave, ax_cf = fig.add_subplot(grid[3, 0:3]), fig.add_subplot(grid[3, 3:6])
    ax_waves = [fig.add_subplot(grid[4, c:c + 2]) for c in (0, 2, 4)]
    ax_bells = [fig.add_subplot(grid[5, c:c + 2]) for c in (0, 2, 4)]
    for ax in ax_waves[1:]:
        ax.sharey(ax_waves[0])
    for ax in ax_bells[1:]:
        ax.sharey(ax_bells[0])

    # a partial sum against its target, as curves and as areas under curve x
    # p(z). the curves are scaled by one common factor, so the target curve is
    # as high as its area and the two curves stay comparable
    def build_up(ax, part, target, family, target_family, unit, target_name, title, first):
        scale = (target * pz).max() / target.max()
        ax.fill_between(z, target * pz, color=_BAND[target_family], lw=0,
                        label=rf"{target_name} $\times\, p(z)$")
        ax.fill_between(z, part * pz, color=_FILL[family], alpha=0.75, lw=0,
                        label=rf"sum of {unit} $\times\, p(z)$")
        ax.plot(z, part * pz, color=_LINE[family], lw=0.8)
        _dashed(ax, z, target * scale, target_family, label=f"{target_name} (scaled)")
        ax.plot(z, part * scale, color=_LINE[family], lw=2, label=f"sum of {unit} (scaled)")
        ax.set_xlim(-9, 10)
        # shared y axis, so the legend room only once
        _finish(ax, title, room=0.6 if first else 0, legend=first)

    # ── goal ──
    _goal_panel(ax_goal, z, pz, sigma, d["answer"], texts["goal_title"])

    # ── steps 1 and 2: the sums (only curves), then where the pieces come from ──
    ax = ax_pg
    _band(ax, z, sigma, _SIGMA_BAND, label=r"$\sigma(z)$")
    for color, lw, k in zip(_BELL_LINES, _SUM_WIDTHS, d["bell_ks"]):
        ax.plot(z, d["bells"][by_weight[:k]].sum(axis=0), color=color, lw=lw, label=plural(k, "bell", n_bells))
    ax.set_xlim(z[0], z[-1])
    _finish(ax, r"$\sigma(z)$ as a sum of bells", room=0.4)

    ax = ax_pg_origin
    ga, gd = d["gamma_axis"], d["pg_density"]
    ax.fill_between(ga, gd / gd.max() * d["rhos"].max(), color=_CAT["grey"], lw=0,
                    label="PG(1,0) density (scaled)")
    others = np.arange(n_bells) != d["j"]
    ax.vlines(d["gammas"][others], 0, d["rhos"][others], color=_FILL["indigo"], lw=3.5,
              label=r"chosen widths $\gamma$")
    ax.vlines(d["gamma"], 0, d["rhos"][d["j"]], color=_LINE["indigo"], lw=3.5,
              label=rf"the followed width $\gamma = {d['gamma']:.3f}$")
    ax.set_xlim(ga[0], ga[-1])
    _finish(ax, r"Where do the bells come from? Widths $\gamma$ from PG(1,0)", xlabel=r"bell width $\gamma$")

    ax = ax_hs
    _band(ax, z, d["bell"], _BELL_BAND, label="the followed bell")
    for color, lw, k in zip(_WAVE_LINES, _SUM_WIDTHS, d["wave_ks"]):
        ax.plot(z, d["waves"][:k].sum(axis=0), color=color, lw=lw, label=plural(k, "wave", n_waves))
    ax.set_xlim(z[0], z[-1])
    _finish(ax, "A bell as a sum of waves")

    ax = ax_hs_origin
    u_max = 1.1 * np.abs(d["u_all"]).max()
    u_axis = np.linspace(-u_max, u_max, 600)
    dens = np.exp(-u_axis**2 / (2 * d["gamma"]))
    ax.fill_between(u_axis, dens * d["u_weights"].max(), color=_CAT["grey"], lw=0,
                    label=r"$N(0, \gamma)$ density (scaled)")
    ax.vlines(d["u_all"], 0, d["u_weights"], color=_LINE["coral"], lw=2.2, label=r"chosen frequencies $u$")
    ax.set_xlim(-u_max, u_max)
    _finish(ax, r"Where do the waves come from? Frequencies $u$ from $N(0, \gamma)$",
            xlabel="wave frequency $u$")

    # ── step 3: the area under a wave, the slow one coral, the fast one wine,
    # both in one panel as in the thesis figure ──
    ax = ax_wave
    ax.fill_between(z, pz, color=_CAT["grey"], lw=0, label="$p(z)$")
    for family, i, name in zip(("coral", "wine"), d["show_waves"], ("slow", "fast")):
        ax.plot(z, d["cosines"][i] * pz.max(), color=_LINE[family], lw=1, ls=(0, (4, 2)))
        ax.fill_between(z, d["cosines"][i] * pz, color=_FILL[family], alpha=0.85, lw=0,
                        label=rf"{name} wave, $u = {d['u_pairs'][i]:.2f}$: area ${d['wave_avgs'][i]:+.3f}$")
        ax.plot(z, d["cosines"][i] * pz, color=_LINE[family], lw=1.2)
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(-1.15 * pz.max(), 1.25 * pz.max())
    _finish(ax, r"The area under wave $\times\, p(z)$", room=0.45)

    ax = ax_cf
    u_max = 1.05 * d["u_pairs"].max()
    u_axis = np.linspace(0, u_max, 800)
    ax.plot(u_axis, d["wave_avg_fn"](u_axis), color=_tone("coral", 0.7), lw=2.5,
            label=r"area for any frequency $u$: $\mathrm{Re}[e^{-iuc}\,\phi(u)]$")
    ax.plot(d["u_pairs"], d["wave_avgs"], "o", color=_LINE["coral"], ms=5.5,
            label="the waves of the followed bell")
    for family, i, name in zip(("coral", "wine"), d["show_waves"], ("slow", "fast")):
        u, a = d["u_pairs"][i], d["wave_avgs"][i]
        ax.plot(u, a, "o", ms=15, mfc="none", mec=_LINE[family], mew=2)
        ax.annotate(name, (u, a), xytext=(14, 10), textcoords="offset points", fontsize=12,
                    color=_LINE[family])
    ax.set_xlim(0, u_max)
    ax.set_ylabel(r"area under wave $\times\, p(z)$", fontsize=12)
    _finish(ax, r"The area of every wave, from the characteristic function $\phi$",
            xlabel="wave frequency $u$", room=0.5)

    # ── steps 4 and 5: the sums and their areas build up, for 1, 2 and all pieces ──
    for ax, k in zip(ax_waves, _first_two_and_all(d["wave_ks"])):
        extra = rf"  (bell {d['bell_avg']:.4f})" if k == n_waves else ""
        build_up(ax, d["waves"][:k].sum(axis=0), d["bell"], "coral", "indigo", "waves", "bell",
                 f"{plural(k, 'wave', n_waves).capitalize()}: area {d['wave_parts'][:k].sum():.4f}{extra}",
                 ax is ax_waves[0])

    for ax, k in zip(ax_bells, _first_two_and_all(d["bell_ks"])):
        extra = rf"  (exact {d['answer']:.4f})" if k == n_bells else ""
        build_up(ax, d["bells"][by_weight[:k]].sum(axis=0), sigma, "indigo", "gold", "bells", r"$\sigma(z)$",
                 f"{plural(k, 'bell', n_bells).capitalize()}: area "
                 f"{d['bell_parts'][by_weight[:k]].sum():.4f}{extra}",
                 ax is ax_bells[0])

    _caption(fig, ax_goal, texts["goal"])
    _row_caption(fig, [ax_pg, ax_pg_origin], texts["pg"], texts["pg_origin"])
    _row_caption(fig, [ax_hs, ax_hs_origin], texts["hs"], texts["hs_origin"])
    _row_caption(fig, [ax_wave, ax_cf], texts["wave_area"], texts["cf"])
    _row_caption(fig, ax_waves, texts["waves"])
    _row_caption(fig, ax_bells, texts["bells"])
    rows = [("Goal", texts["goal_heading"], [ax_goal]),
            ("Step 1", r"$\sigma(z)$ is a sum of bells", [ax_pg, ax_pg_origin]),
            ("Step 2", "Every bell is a sum of waves", [ax_hs, ax_hs_origin]),
            ("Step 3", "The area under a wave comes from the model", [ax_wave, ax_cf]),
            ("Step 4", "The wave areas add up to the area under a bell", ax_waves),
            ("Step 5", r"The bell areas add up to the area under $\sigma(z) \times p(z)$", ax_bells)]
    _headings(fig, rows, title)
    # a white margin, so the page border of a pdf viewer does not touch the panels
    fig.savefig(out_path, bbox_inches="tight", pad_inches=1)
    plt.close(fig)


# ── pcpg figures for the thesis (pcpg_demo.py) ───────────────────────────────

# the thesis figures in the colours of the step-by-step figure, at the thesis
# sizes: the sums of bells and waves as in its steps 1 and 2, the slow and fast
# wave as in its step 3 (coral and wine), the summary as in its step 5 (indigo
# bells against the butter-yellow sigmoid(z) x p(z))

# the line widths of the step-by-step figure, scaled to the thesis
_THESIS_SCALE = 0.45

def _count(k, n, unit):
    return f"all {unit}s" if k == n else f"{k} {unit}{'s' if k > 1 else ''}"

# a single panel for the thesis, at 3/4 of the text width
def _thesis_panel():
    return plt.subplots(figsize=(0.75 * THESIS_WIDTH, 2.2))

# partial sums against their target, as in steps 1 and 2 of the step-by-step figure
def _thesis_sums(ax, z, target, band, label, sums, ks, n, unit, colors):
    _band(ax, z, target, band, label=label, lw=13 * _THESIS_SCALE)
    for color, lw, k, part in zip(colors, _SUM_WIDTHS, ks, sums):
        ax.plot(z, part, color=color, lw=lw * _THESIS_SCALE, label=_count(k, n, unit))

# sums of the bells with the largest weights closing in on sigmoid(z)
@styled
def plot_thesis_bells(d, out_path):
    fig, ax = _thesis_panel()
    z = d["z"]
    ks = d["bell_ks"]
    _thesis_sums(ax, z, 1 / (1 + np.exp(-z)), _SIGMA_BAND, r"$\sigma(z)$",
                 [d["bells"][d["by_weight"][:k]].sum(axis=0) for k in ks], ks, len(d["bells"]), "bell",
                 _BELL_LINES)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"$z$")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)

# sums of the slowest waves closing in on the followed bell
@styled
def plot_thesis_waves(d, out_path):
    fig, ax = _thesis_panel()
    z = d["z"]
    ks = d["wave_ks"]
    _thesis_sums(ax, z, d["bell"], _BELL_BAND, r"bell $e^{z/2 - \gamma z^2/2}$",
                 [d["waves"][:k].sum(axis=0) for k in ks], ks, len(d["waves"]), "wave", _WAVE_LINES)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(-0.25 * d["bell"].max(), 1.55 * d["bell"].max())
    ax.set_xlabel(r"$z$")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out_path)
    plt.close(fig)

# the area under wave x p(z) for a slow (coral) and a fast (wine) wave: the
# expectation of the wave, which the characteristic function gives
@styled
def plot_thesis_cf(d, out_path):
    fig, ax = _thesis_panel()
    z, pz = d["z"], d["pz"]
    ax.fill_between(z, pz, color=_CAT["grey"], lw=0, label=r"$p(z)$")
    for family, i in zip(("coral", "wine"), d["show_waves"]):
        ax.plot(z, d["cosines"][i] * pz.max(), color=_LINE[family], lw=0.6, ls=(0, (3, 2)))
        ax.fill_between(z, d["cosines"][i] * pz, color=_FILL[family], alpha=0.85, lw=0,
                        label=rf"$u = {d['u_pairs'][i]:.2f}$: area ${d['wave_avgs'][i]:+.3f}$")
        ax.plot(z, d["cosines"][i] * pz, color=_LINE[family], lw=0.8)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.1 * pz.max(), 2.1 * pz.max())
    ax.set_xlabel(r"$z$")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)

# the area under sigmoid(z) x p(z) rebuilt from the bells with the largest
# weights, nested fills in the bell colours (the partial sums only grow, every
# bell is positive), the exact area as a dashed outline. the numbers are the
# pcpg estimates, every bell's area from its waves
@styled
def plot_thesis_summary(d, out_path):
    fig, ax = plt.subplots(figsize=(THESIS_WIDTH, 2.3))
    z, pz = d["z"], d["pz"]
    sigma = 1 / (1 + np.exp(-z))
    n = len(d["bells"])
    ax.fill_between(z, pz, color=_CAT["grey"], lw=0, label=r"$p(z)$")
    ks = d["bell_ks"]
    for color, k in reversed(list(zip(_BELL_LINES, ks))):
        part = d["bells"][d["by_weight"][:k]].sum(axis=0) * pz
        estimate = d["bell_parts"][d["by_weight"][:k]].sum()
        ax.fill_between(z, part, color=color, lw=0)
        ax.fill_between([], [], color=color, label=rf"{_count(k, n, 'bell')}: ${estimate:.4f}$")
    ax.plot(z, sigma * pz, color=_LINE["gold"], lw=1.3, ls=(0, (3, 2)),
            label=rf"$\sigma(z)\, p(z)$: ${d['answer']:.4f}$ (exact)")
    ax.set_xlim(-9, 9)
    ax.set_ylim(0, 1.08 * pz.max())
    ax.set_xlabel(r"$z$")
    lines = ax.get_legend_handles_labels()
    order = [0] + list(range(len(lines[0]) - 2, 0, -1)) + [len(lines[0]) - 1]
    ax.legend([lines[0][i] for i in order], [lines[1][i] for i in order], loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


# ── gmm gauss-hermite step by step (gmm_gh_demo.py) ──────────────────────────

def plot_gmm_gh_step_by_step(d, texts, out_path, title=None):
    with plt.rc_context(_BIG_STYLE):
        _plot_gmm_gh_step_by_step(d, texts, out_path, title)

def _plot_gmm_gh_step_by_step(d, texts, out_path, title=None):
    z, pz, mix, j = d["z"], d["pz"], d["mix"], d["j"]
    sigma = 1 / (1 + np.exp(-z))
    order, n_comp, n_nodes = d["order"], len(d["mix"]), len(d["nodes"])
    # the components in blue, the followed one (largest weight) dark, the others
    # lighter, the same colour for a component in every panel
    comp_line = dict(zip(order, [_LINE["indigo"]] + _tones("indigo", n_comp - 1, lo=0.6, hi=1.2)))
    comp_fill = dict(zip(order, [_tone("indigo", 1.25)] + _tones("indigo", n_comp - 1, lo=0.3, hi=0.75)))

    # 6 columns, as in the pcpg figure
    fig = plt.figure(figsize=(16, 42))
    grid = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 0.9, 0.9], hspace=1.95, wspace=0.6)
    ax_goal = fig.add_subplot(grid[0, 1:5])
    ax_mix, ax_split = fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6])
    ax_nodes, ax_origin = fig.add_subplot(grid[2, 0:3]), fig.add_subplot(grid[2, 3:6])
    ax_polys = [fig.add_subplot(grid[3, c:c + 2]) for c in (0, 2, 4)]
    ax_sums = [fig.add_subplot(grid[4, c:c + 2]) for c in (0, 2, 4)]
    for ax in ax_polys[1:]:
        ax.sharey(ax_polys[0])
    for ax in ax_sums[1:]:
        ax.sharey(ax_sums[0])

    # the areas sigma(z) x pi_k N(m_k, v_k) of the components in part, stacked
    # in their colours. all of them stacked are the goal area
    def stack(ax, part, labels=False):
        base = np.zeros_like(z)
        for k in part:
            top = base + sigma * d["components"][k]
            label = (rf"$\sigma(z) \times \pi_{k + 1} \mathcal{{N}}(m_{k + 1}, v_{k + 1})$: "
                     rf"area {mix[k] * d['comp_exact'][k]:.4f}") if labels else None
            ax.fill_between(z, base, top, color=comp_fill[k], lw=0, label=label)
            ax.plot(z, top, color="white", lw=0.8)
            base = top

    # ── goal ──
    _goal_panel(ax_goal, z, pz, sigma, d["answer"])

    # ── step 1: the mixture, and the area split by component ──
    ax = ax_mix
    ax.fill_between(z, pz, color=_CAT["grey"], lw=0, label="$p(z)$")
    for k in order:
        ax.plot(z, d["components"][k], color=comp_line[k], lw=2.2,
                label=rf"$\pi_{k + 1} \, \mathcal{{N}}(m_{k + 1}, v_{k + 1})$, $\pi_{k + 1} = {mix[k]:.3f}$")
    ax.set_xlim(z[0], z[-1])
    _finish(ax, r"$p(z)$ as a sum of Gaussians", room=0.6)

    ax = ax_split
    stack(ax, order, labels=True)
    ax.plot(z, sigma * pz, color=_LINE["gold"], lw=1.8,
            label=rf"$\sigma(z) \times p(z)$: area {d['answer']:.4f}")
    ax.set_xlim(z[0], z[-1])
    _finish(ax, "The area as a sum of component areas", room=0.6)

    # ── step 2: the points of the followed component, and where they come from ──
    ax = ax_nodes
    dens = d["components"][j] / mix[j]
    # the gaussian and the weights scaled to 0.6, so they read next to sigma
    ax.fill_between(z, dens * 0.6 / dens.max(), color=_BAND["indigo"], lw=0,
                    label=rf"$\mathcal{{N}}(m_{j + 1}, v_{j + 1})$, the followed component (scaled)")
    _dashed(ax, z, sigma, "gold", label=r"$\sigma(z)$")
    scale = 0.6 / d["weights"].max()
    ax.vlines(d["nodes"], 0, d["weights"] * scale, color=_LINE["coral"], lw=3,
              label=r"points $z_i$ (height = weight $h_i / \sqrt{\pi}$, scaled)")
    ax.plot(d["nodes"], d["sigma_nodes"], "o", color=_LINE["coral"], ms=8, zorder=3,
            label=rf"$\sigma(z_i)$: weighted sum {d['comp_gh'][j]:.4f} (exact {d['comp_exact'][j]:.4f})")
    ax.set_xlim(z[0], z[-1])
    _finish(ax, r"The component's area from $\sigma$ at a few points", room=0.6)

    ax = ax_origin
    t_max = 1.15 * np.abs(d["t"]).max()
    t_axis = np.linspace(-t_max, t_max, 600)
    ax.fill_between(t_axis, np.exp(-t_axis**2) * d["weights"].max(), color=_CAT["grey"], lw=0,
                    label=r"$e^{-t^2}$ (scaled)")
    ax.vlines(d["t"], 0, d["weights"], color=_LINE["coral"], lw=3,
              label=r"Gauss-Hermite points $t_i$, weights $h_i / \sqrt{\pi}$")
    ax.set_xlim(-t_max, t_max)
    _finish(ax, r"Where do the points come from? Gauss-Hermite for $e^{-t^2}$",
            xlabel=r"standardised point $t$, \ $z = m_k + \sqrt{2 v_k}\, t$", room=0.4)

    # ── step 3: n points integrate the polynomial through sigma at them ──
    sel = (z > d["m_j"] - 4.5 * d["s_j"]) & (z < d["m_j"] + 4.5 * d["s_j"])
    area_scale = (sigma * dens).max()
    for ax, n, poly, pts, area in zip(ax_polys, *(_first_two_and_all(d[key]) for key in
                                                  ("node_ks", "polys", "poly_nodes", "poly_areas"))):
        first = ax is ax_polys[0]
        ax.fill_between(z[sel], (sigma * dens)[sel], color=_BAND["gold"], lw=0,
                        label=r"$\sigma(z) \times \mathcal{N}$")
        ax.fill_between(z[sel], (poly * dens)[sel], color=_FILL["coral"], alpha=0.75, lw=0,
                        label=r"polynomial $\times\, \mathcal{N}$")
        _dashed(ax, z[sel], (sigma * area_scale)[sel], "gold", label=r"$\sigma(z)$ (scaled)")
        ax.plot(z[sel], (poly * area_scale)[sel], color=_LINE["coral"], lw=2, label="polynomial (scaled)")
        ax.plot(pts, 1 / (1 + np.exp(-pts)) * area_scale, "o", color=_LINE["coral"], ms=7, zorder=3)
        ax.set_xlim(z[sel][0], z[sel][-1])
        # shared y axis, so the same room for the legend in every panel
        ax.set_ylim(-0.25 * area_scale, 1.9 * area_scale)
        extra = rf"  (exact {d['comp_exact'][j]:.4f})" if n == n_nodes else ""
        _finish(ax, f"{_count(n, n_nodes, 'point').capitalize()}: area {area:.4f}{extra}", room=0,
                legend=first)

    # ── step 4: the component areas add up, for 1, 2 and all components ──
    for i, ax in zip(_first_two_and_all(range(n_comp)), ax_sums):
        first, part = ax is ax_sums[0], order[:i + 1]
        ax.fill_between(z, sigma * pz, color=_BAND["gold"], lw=0, label=r"$\sigma(z) \times p(z)$")
        stack(ax, part)
        ax.fill_between([], [], color=comp_fill[order[0]], lw=0, label=r"$\sigma(z) \times$ the followed component")
        ax.fill_between([], [], color=comp_fill[order[1]], lw=0, label=r"$\sigma(z) \times$ the other components")
        ax.plot(z, sigma * pz, color=_LINE["gold"], lw=1.4)
        ax.set_xlim(-9, 10)
        total = mix[part] @ d["comp_gh"][part]
        extra = rf"  (exact {d['answer']:.4f})" if i == n_comp - 1 else ""
        _finish(ax, f"{_count(i + 1, n_comp, 'component').capitalize()}: area {total:.4f}{extra}",
                room=0.6 if first else 0, legend=first)

    _caption(fig, ax_goal, texts["goal"])
    _row_caption(fig, [ax_mix, ax_split], texts["mix"], texts["split"])
    _row_caption(fig, [ax_nodes, ax_origin], texts["nodes"], texts["origin"])
    _row_caption(fig, ax_polys, texts["polys"])
    _row_caption(fig, ax_sums, texts["sums"])
    rows = [("Goal", "What we want to compute", [ax_goal]),
            ("Step 1", r"$p(z)$ is a mixture of Gaussians, so the area splits by component", [ax_mix, ax_split]),
            ("Step 2", r"The area of a component is a weighted sum of $\sigma$ at a few points",
             [ax_nodes, ax_origin]),
            ("Step 3", r"Why it works: the sum is the area under a polynomial through $\sigma$", ax_polys),
            ("Step 4", r"The component areas add up to the area under $\sigma(z) \times p(z)$", ax_sums)]
    _headings(fig, rows, title)
    # a white margin, so the page border of a pdf viewer does not touch the panels
    fig.savefig(out_path, bbox_inches="tight", pad_inches=1)
    plt.close(fig)


# ── monte carlo step by step (mc_demo.py) ────────────────────────────────────

def plot_mc_step_by_step(d, texts, out_path, title=None):
    with plt.rc_context(_BIG_STYLE):
        _plot_mc_step_by_step(d, texts, out_path, title)

def _plot_mc_step_by_step(d, texts, out_path, title=None):
    z, pz, mix = d["z"], d["pz"], d["mix"]
    sigma = 1 / (1 + np.exp(-z))
    order, n_comp, n_show = d["order"], len(d["mix"]), d["n_show"]
    # the components in blue as in the gmm figure, the one with the largest
    # weight dark, the others lighter
    comp_line = dict(zip(order, [_LINE["indigo"]] + _tones("indigo", n_comp - 1, lo=0.6, hi=1.2)))

    # 6 columns, as in the pcpg figure
    fig = plt.figure(figsize=(16, 42))
    grid = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 0.9, 1], hspace=1.95, wspace=0.6)
    ax_goal = fig.add_subplot(grid[0, 1:5])
    ax_draw, ax_scores = fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6])
    ax_predict, ax_average = fig.add_subplot(grid[2, 0:3]), fig.add_subplot(grid[2, 3:6])
    ax_hists = [fig.add_subplot(grid[3, c:c + 2]) for c in (0, 2, 4)]
    ax_runs, ax_rate = fig.add_subplot(grid[4, 0:3]), fig.add_subplot(grid[4, 3:6])
    for ax in ax_hists[1:]:
        ax.sharey(ax_hists[0])

    # ── goal ──
    _goal_panel(ax_goal, z, pz, sigma, d["answer"])

    # ── step 1: fill-ins of the missing features, and their scores ──
    ax = ax_draw
    for k in order[::-1]:
        sel = d["k"] == k
        ax.plot(*d["x_mis"][sel].T, "o", color=comp_line[k], ms=3, alpha=0.55, mec="none",
                label=rf"component {k + 1}, $\pi_{k + 1} = {mix[k]:.3f}$")
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(d["means"][k, 0] + 2 * d["stds"][k, 0] * np.cos(t),
                d["means"][k, 1] + 2 * d["stds"][k, 1] * np.sin(t), color=comp_line[k], lw=2)
    # the legend from the top component down
    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylabel(rf"missing feature $x_{{{d['feats'][1] + 1}}}$", fontsize=12)
    _finish(ax, r"Fill-ins $x_{\mathrm{mis}}^{(i)}$ drawn from the model",
            xlabel=rf"missing feature $x_{{{d['feats'][0] + 1}}}$", room=0.85, legend=False)
    ax.legend(handles[::-1], labels[::-1], loc="upper left", markerscale=2.5)

    ax = ax_scores
    ax.fill_between(z, pz, color=_CAT["grey"], lw=0, label="$p(z)$")
    ax.hist(d["z_samples"], bins=60, range=(z[0], z[-1]), density=True, histtype="step",
            color=_LINE["indigo"], lw=2, label=rf"histogram of {len(d['z_samples'])} scores $z_i$")
    ax.plot(d["z_samples"][:200], np.full(200, -0.03 * pz.max()), "|", color=_LINE["indigo"],
            ms=10, alpha=0.5)
    ax.set_xlim(z[0], z[-1])
    _finish(ax, r"Their scores $z_i = w^\top x^{(i)}$ are samples of $p(z)$", room=0.45)

    # ── step 2: the prediction at every sample, and their average ──
    zs = d["z_run"][:n_show]
    preds = 1 / (1 + np.exp(-zs))
    ax = ax_predict
    ax.fill_between(z, pz / pz.max() * 0.6, color=_CAT["grey"], lw=0, label="$p(z)$ (scaled)")
    _dashed(ax, z, sigma, "gold", label=r"$\sigma(z)$")
    ax.vlines(zs, 0, preds, color=_FILL["indigo"], lw=1.5)
    ax.plot(zs, preds, "o", color=_LINE["indigo"], ms=7, zorder=5,
            label=rf"predictions $\sigma(z_i)$ of {n_show} samples")
    ax.set_xlim(z[0], z[-1])
    _finish(ax, r"The prediction $\sigma(z_i)$ for every sample", room=0.4)

    ax = ax_average
    i = np.arange(1, n_show + 1)
    _band(ax, [0.3, n_show + 0.7], [d["answer"]] * 2, _SIGMA_BAND, label=f"exact {d['answer']:.4f}")
    ax.vlines(i, 0, preds, color=_FILL["indigo"], lw=1.5)
    ax.plot(i, preds, "o", color=_LINE["indigo"], ms=7, zorder=5, label=r"$\sigma(z_i)$")
    ax.plot([0.3, n_show + 0.7], [preds.mean()] * 2, color=_LINE["indigo"], lw=2.5,
            label=rf"average {preds.mean():.4f}")
    ax.set_xlim(0.3, n_show + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 1)
    _finish(ax, "The estimate is their average", xlabel="sample $i$", room=0.4)

    # ── step 3: the average is the area under sigma x the histogram ──
    edges = np.linspace(-9, 10, 39)
    width = edges[1] - edges[0]
    for ax, n in zip(ax_hists, d["sample_ks"]):
        first = ax is ax_hists[0]
        zs = d["z_run"][:n]
        bars = np.histogram(zs, bins=edges, weights=1 / (1 + np.exp(-zs)) / (n * width))[0]
        ax.fill_between(z, sigma * pz, color=_BAND["gold"], lw=0, label=r"$\sigma(z) \times p(z)$")
        ax.stairs(bars, edges, fill=True, color=_FILL["indigo"], alpha=0.75, lw=0,
                  label=r"$\sigma(z) \times$ histogram")
        ax.stairs(bars, edges, color=_LINE["indigo"], lw=0.8)
        _dashed(ax, z, sigma * pz, "gold")
        ax.set_xlim(edges[0], edges[-1])
        # shared y axis, so the same room for the legend in every panel
        ax.set_ylim(0, 2.2 * (sigma * pz).max())
        extra = rf"  (exact {d['answer']:.4f})" if n == d["sample_ks"][-1] else ""
        _finish(ax, f"{n:,} samples: area {bars.sum() * width:.4f}{extra}".replace(",", r"\,"),
                room=0, legend=first)

    # ── step 4: many runs, and how fast the error shrinks ──
    ax = ax_runs
    n = np.arange(1, d["running"].shape[1] + 1)
    sd = np.sqrt(d["var"])
    ax.fill_between(n, d["answer"] - 2 * sd / np.sqrt(n), d["answer"] + 2 * sd / np.sqrt(n),
                    color=_SIGMA_BAND, lw=0, label=r"exact $\pm 2\sigma_{\mathrm{MC}}/\sqrt{n}$")
    ax.axhline(d["answer"], color=_LINE["gold"], lw=1.2, ls=(0, (4, 2)))
    runs = d["running"]
    for run, color in zip(runs[1:], _tones("indigo", len(runs) - 1, lo=0.5, hi=1.1)):
        ax.plot(n, run, color=color, lw=1.2)
    ax.plot(n, runs[0], color=_DEEP["indigo"], lw=2, label=f"{len(runs)} runs, the run of step 3 dark")
    ax.set_xscale("log")
    ax.set_xlim(1, n[-1])
    ax.set_ylim(max(0, d["answer"] - 0.45), min(1, d["answer"] + 0.45))
    ax.set_ylabel("estimate after $n$ samples", fontsize=12)
    _finish(ax, "Every run is different, but all close in", xlabel="samples $n$", room=0.35)

    ax = ax_rate
    ax.plot(d["ns"], sd / np.sqrt(d["ns"]), color=_CAT["overlay"], lw=2, ls=(0, (5, 2.5)),
            label=r"$\sigma_{\mathrm{MC}}/\sqrt{n}$")
    ax.plot(d["ns"], d["rmse"], "o", color=_LINE["indigo"], ms=6, label="root mean squared error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 1.25 * d["ns"][-1])
    ax.set_ylabel("error", fontsize=12)
    ax.set_title(_bold(r"The error shrinks like $1/\sqrt{n}$"), fontsize=15, pad=10, color=_CAT["text"])
    ax.set_xlabel("samples $n$", fontsize=12)
    ax.legend(loc="lower left")

    _caption(fig, ax_goal, texts["goal"])
    _row_caption(fig, [ax_draw, ax_scores], texts["draw"], texts["scores"])
    _row_caption(fig, [ax_predict, ax_average], texts["predict"], texts["average"])
    _row_caption(fig, ax_hists, texts["hists"])
    _row_caption(fig, [ax_runs, ax_rate], texts["runs"], texts["rate"])
    rows = [("Goal", "What we want to compute", [ax_goal]),
            ("Step 1", "Fill in the missing features at random, from the model", [ax_draw, ax_scores]),
            ("Step 2", "Predict for every fill-in and average", [ax_predict, ax_average]),
            ("Step 3", r"The average is an area under $\sigma(z) \times$ a histogram", ax_hists),
            ("Step 4", "The estimate is random, its error shrinks slowly", [ax_runs, ax_rate])]
    _headings(fig, rows, title)
    # a white margin, so the page border of a pdf viewer does not touch the panels
    fig.savefig(out_path, bbox_inches="tight", pad_inches=1)
    plt.close(fig)


# ── estimator comparison (estimate.py) ───────────────────────────────────────

# thesis figures of the experiments: the error (rmse over the test points) of
# the methods, in the line colours of the method figures (_LINE, as in the
# step-by-step figure). every method has one colour in every figure: pcpg
# indigo like the bells, mc coral, mean imputation the grey of p(z). the pcpg
# variants are coloured by the part that is mc: the outer integral (bells)
# pastel indigo, the inner one (waves) pastel coral, both wine
_METHOD_STYLE = {
    "Mean imputation": dict(color=_CAT["overlay"], ls=(0, (4, 2)), marker=None),
    "MC":              dict(color=_LINE["coral"], marker="o"),
    "PCPG":            dict(color=_LINE["indigo"], marker="s"),
    "PCPG outer MC":   dict(color=_FILL["indigo"], marker="^"),
    "PCPG inner MC":   dict(color=_FILL["coral"], marker="v"),
    "PCPG full MC":    dict(color=_LINE["wine"], marker="D"),
}
def _percent(rate):
    return rf"{round(100 * rate)}\%"

# one row of three panels at the text width, the legend above them
def _three_panels(**kwargs):
    return plt.subplots(1, 3, figsize=(THESIS_WIDTH, 2.1), layout="constrained", **kwargs)

def _top_legend(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol)

def _method_line(ax, x, y, name, **kwargs):
    style = _METHOD_STYLE[name]
    ax.plot(x, y, color=style["color"], ls=style.get("ls", "-"), marker=style["marker"], ms=3, lw=1.1,
            mfc=style.get("mfc"), label=name, **kwargs)

# the main methods against the number of features (x), one column per missing
# rate: the error on top, the time for all n_points test points below. errors
# and times: missing rate -> name -> one value per number of features
@styled
def plot_estimate_comparison(features, rates, errors, times, n_points, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(THESIS_WIDTH, 3.8), layout="constrained", sharex=True, sharey="row")
    for column, rate in zip(axes.T, rates):
        for ax, values in zip(column, (errors, times)):
            for name, ys in values[rate].items():
                _method_line(ax, features, ys, name)
            ax.set_xscale("log")
            ax.set_yscale("log")
        column[0].set_title(f"{_percent(rate)} missing")
        column[1].set_xlabel("features")
    axes[0, 0].set_ylabel("RMSE")
    axes[1, 0].set_ylabel(f"time for {n_points:,} points (s)".replace(",", r"\,"))
    _top_legend(fig, axes[0, 0], ncol=len(errors[rates[0]]))
    fig.savefig(out_path)
    plt.close(fig)

# the four pcpg variants, error against budget, one panel per missing rate.
# results: one dict per missing rate, name -> (budgets, errors)
@styled
def plot_pcpg_variants(rates, results, out_path):
    fig, axes = _three_panels(sharey=True)
    for ax, rate, result in zip(axes, rates, results):
        for name, (budgets, values) in result.items():
            _method_line(ax, budgets, values, name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("budget (cf evaluations)")
        ax.set_title(f"{_percent(rate)} missing")
    axes[0].set_ylabel("RMSE")
    _top_legend(fig, axes[0], ncol=len(results[0]))
    fig.savefig(out_path)
    plt.close(fig)

# ── training on incomplete data (train.py) ───────────────────────────────────

# the methods of estimate.py keep their colours, the oracle on complete data
# is a dotted slate line. the target of a figure (exact for the objective
# gap, complete data for the accuracies) is a pale wide butter band, as every
# target, the methods that reach it sit inside it
_METHOD_STYLE["Complete data"] = dict(color=_CAT["text"], ls=(0, (1, 1.5)), marker=None)

# complete data and mean imputation fit by sklearn (the comparison of sklearn
# against adam) in their colours with open markers, the ones by adam with
# filled markers and solid lines
_METHOD_STYLE["Complete data, sklearn"] = dict(color=_CAT["text"], ls=(0, (1, 1.5)), marker="o", mfc="white")
_METHOD_STYLE["Mean imputation, sklearn"] = dict(color=_CAT["overlay"], ls=(0, (4, 2)), marker="o", mfc="white")
_SOLID = {"Complete data": dict(color=_CAT["text"], marker="s"), "Mean imputation": dict(color=_CAT["overlay"], marker="s")}

# a method's line, or the band if it is the target
def _train_line(ax, x, y, name, target):
    if name == target:
        _band(ax, x, y, _SIGMA_BAND, label=name, lw=13 * _THESIS_SCALE)
    else:
        _method_line(ax, x, y, name)

# the objective gap is positive, but can come out at 0 or slightly below
# for the optimum itself (rounding), the log axis needs a floor
def _positive(values):
    return np.maximum(np.asarray(values, dtype=float), 1e-8)

# all training methods against the number of features, one panel per missing
# rate, as the comparison of estimate.py. results: missing rate -> method -> one
# value per number of features, ylabel the score, log for a log y axis (the
# objective gap), target the method drawn as the target band
@styled
def plot_train_features(features, rates, results, ylabel, log, target, out_path):
    fig, axes = _three_panels(sharey=True)
    for ax, rate in zip(axes, rates):
        # the target first, so the legend starts with it
        for name, values in sorted(results[rate].items(), key=lambda item: item[0] != target):
            _train_line(ax, features, _positive(values) if log else values, name, target)
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("features")
        ax.set_title(f"{_percent(rate)} missing")
    axes[0].set_ylabel(ylabel)
    _top_legend(fig, axes[0], ncol=3)
    fig.savefig(out_path)
    plt.close(fig)


# ── training step by step (train_demo.py) ────────────────────────────────────

# how w is trained on incomplete data, with pcpg or with mc, one row per step,
# the same rows in both figures so they can be compared:
#   goal     the prediction of a training point, the area under sigma(z) x p(z)
#   step 1   the estimates of many training points against the exact values,
#            and the followed point asked many times
#   step 2   the loss -log p of these estimates, and its error against the budget
#   step 3   the objective along w = t w*, and its slope
#   step 4   the excess objective during training, and the path of w
# the method has its colour of the experiment figures (pcpg indigo, mc coral),
# the exact values are the target in butter yellow (bands and dashed lines),
# p(z) grey. d is a dict, see pieces in train_demo.py, texts the explanations
# and the headings

def plot_train_step_by_step(d, texts, out_path, title=None):
    with plt.rc_context(_BIG_STYLE):
        _plot_train_step_by_step(d, texts, out_path, title)

# a bold title and the legend, for panels where _finish does not fit (log axes)
def _title(ax, title, xlabel, legend="upper left"):
    ax.set_xlabel(xlabel, fontsize=12)
    if legend:
        ax.legend(loc=legend)
    ax.set_title(_bold(title), fontsize=15, pad=10, color=_CAT["text"])

def _plot_train_step_by_step(d, texts, out_path, title=None):
    z, pz, answer = d["z"], d["pz"], d["answer"]
    sigma = 1 / (1 + np.exp(-z))
    pcpg = d["method"] == "pcpg"
    f = "indigo" if pcpg else "coral"
    name = "PCPG" if pcpg else "MC"
    unit = "nodes" if pcpg else "samples"
    # several runs from dark (the first) to light
    run_colors = [_LINE[f]] + _tones(f, max(1, len(d["excesses"]) - 1), lo=0.75, hi=1.2)

    fig = plt.figure(figsize=(16, 42))
    grid = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 1, 1], hspace=1.95, wspace=0.6)
    ax_goal = fig.add_subplot(grid[0, 1:5])
    ax_scatter, ax_repeat = fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6])
    ax_loss, ax_error = fig.add_subplot(grid[2, 0:3]), fig.add_subplot(grid[2, 3:6])
    ax_slice, ax_slope = fig.add_subplot(grid[3, 0:3]), fig.add_subplot(grid[3, 3:6])
    ax_curve, ax_path = fig.add_subplot(grid[4, 0:3]), fig.add_subplot(grid[4, 3:6])

    # ── goal ──
    _goal_panel(ax_goal, z, pz, sigma, answer, texts["goal_title"])

    # ── step 1: estimates of many points, and of one point many times ──
    ax = ax_scatter
    _band(ax, [0, 1], [0, 1], _SIGMA_BAND, label="exact")
    ax.plot(d["scatter_exact"], d["scatter"], "o", color=_LINE[f], ms=4.5, alpha=0.6, mec="none",
            label=f"{name}, {d['budget']} {unit} per point")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel(f"{name} estimate", fontsize=12)
    _finish(ax, f"{len(d['scatter'])} training points", xlabel="exact prediction $p(y=1\\mid x^o)$",
            room=0.3)

    ax = ax_repeat
    reps = d["repeats"]
    ax.axvline(answer, color=_SIGMA_BAND, lw=13, zorder=1, label=f"exact {answer:.4f}")
    if pcpg:
        ax.axvline(reps[0], color=_LINE[f], lw=2.5, zorder=3,
                   label=f"all {len(reps):,} estimates: {reps[0]:.4f}".replace(",", r"\,"))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
    else:
        ax.hist(reps, bins=40, range=(0, 1), density=True, color=_FILL[f], alpha=0.75, lw=0,
                label=f"{len(reps):,} estimates".replace(",", r"\,"), zorder=2)
        ax.hist(reps, bins=40, range=(0, 1), density=True, histtype="step", color=_LINE[f], lw=1.2, zorder=2)
        ax.axvline(reps.mean(), color=_LINE[f], lw=2.5, zorder=3, label=f"their mean {reps.mean():.4f}")
        ax.set_ylabel("density", fontsize=12)
    ax.set_xlim(0, 1)
    _finish(ax, "The followed point, asked again and again", xlabel="estimate of $p(y=1\\mid x^o)$",
            room=0.5)

    # ── step 2: the loss of the estimates, and its error against the budget ──
    ax = ax_loss
    p_axis = np.linspace(0.02, 1, 400)
    ax.plot(p_axis, -np.log(p_axis), color=_CAT["overlay"], lw=2, label=r"loss $-\log\hat{p}$")
    exact_loss = -np.log(answer)
    ax.axhline(exact_loss, color=_LINE["gold"], lw=1.6, ls=(0, (4, 2)), label=f"exact loss {exact_loss:.3f}")
    ax.plot(answer, exact_loss, "o", ms=16, color=_SIGMA_BAND, zorder=2)
    shown = np.clip(reps[:150], 0.02, 1)
    ax.plot(shown, -np.log(shown), "o", color=_LINE[f], ms=6, alpha=0.5 if not pcpg else 1, mec="none",
            zorder=3, label=f"the losses of the estimates" if not pcpg else "the loss of the estimate")
    if not pcpg:
        mean_loss = np.mean(-np.log(np.clip(reps, 1e-6, 1)))
        ax.axhline(mean_loss, color=_LINE[f], lw=2.5, label=f"their mean {mean_loss:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("loss", fontsize=12)
    _finish(ax, r"The loss $-\log$ of the estimate", xlabel=r"estimate $\hat{p}$ of $p(y=1\mid x^o)$",
            room=0.35, legend=False)
    ax.legend(loc="upper right")

    ax = ax_error
    budgets, errors = d["loss_errors"]
    errors = np.maximum(np.abs(errors), 1e-12)
    if not pcpg:
        # the bias of -log of an average of S samples, var / (2 S p^2), with the
        # variance of one sample from the spread of the estimates
        var = d["budget"] * np.var(d["repeats"])
        b = np.array(budgets, dtype=float)
        ax.plot(b, var / (2 * b * answer**2), color=_CAT["overlay"], lw=2, ls=(0, (5, 2.5)),
                label=r"$\mathrm{Var}[\sigma(z)]/(2Sp^2)$")
    ax.plot(budgets, errors, "o-", color=_LINE[f], ms=7, lw=2,
            label="error of the loss" if pcpg else r"bias: mean of $-\log\hat{p}$ minus $-\log p$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("error of the loss", fontsize=12)
    _title(ax, f"The error of the loss, against the {unit}",
           xlabel=r"nodes $n_{\mathrm{outer}} \times n_{\mathrm{inner}}$" if pcpg else "samples $S$ per point",
           legend="upper right")

    # ── step 3: the objective and its slope along w = t w* ──
    t = d["t"]
    for ax, i, ylabel, label in ((ax_slice, 0, "mean loss", "objective"), (ax_slope, 1, "slope $d/dt$", "slope")):
        _band(ax, t, d["slice_exact"][i], _SIGMA_BAND, label=f"exact {label}")
        for k, (color, run) in enumerate(zip(run_colors + [run_colors[-1]] * 9, d["slices"])):
            ax.plot(t, run[i], color=color, lw=2 if k == 0 else 1.4,
                    label=f"{name} {label}" + ("" if pcpg else f", {len(d['slices'])} runs") if k == 0 else None)
        ax.axvline(1, color=_CAT["surface"], lw=1.2, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel(ylabel, fontsize=12)
    ax_slice.set_ylim(min(d["slice_exact"][0]) - 0.05, min(d["slice_exact"][0]) + 0.3)
    lo, hi = np.percentile(np.concatenate([d["slice_exact"][1]] + [r[1] for r in d["slices"]]), [2, 98])
    ax_slope.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    _finish(ax_slice, r"The objective along $w = t\,w^*$", xlabel=r"$t$ \ ($w = t\,w^*$, the optimum at $t=1$)",
            room=0.3)
    _finish(ax_slope, "Its slope, which the gradient follows",
            xlabel=r"$t$ \ ($w = t\,w^*$, the optimum at $t=1$)", room=0.3)

    # ── step 4: training, the excess objective and the path of w ──
    ax = ax_curve
    epochs = np.arange(len(d["excess_exact"]))
    floor = lambda v: np.maximum(v, 1e-8)
    _band(ax, epochs, floor(d["excess_exact"]), _SIGMA_BAND, label="exact objective")
    for k, (color, run) in enumerate(zip(run_colors, d["excesses"])):
        ax.plot(epochs, floor(run), color=color, lw=2 if k == 0 else 1.4,
                label=f"{name}, {d['budget']} {unit}" + ("" if pcpg else f", {len(d['excesses'])} runs")
                if k == 0 else None)
    ax.set_yscale("log")
    ax.set_xlim(0, epochs[-1])
    ax.set_ylabel("objective minus its optimum", fontsize=12)
    _title(ax, "The objective during training", xlabel="epoch", legend="upper right")

    ax = ax_path
    ax.plot(*d["path_exact"].T, color=_SIGMA_BAND, lw=13, solid_capstyle="round", zorder=1, label="exact")
    for k, (color, path) in enumerate(zip(run_colors, d["paths"])):
        ax.plot(*path.T, "o-", color=color, lw=2 if k == 0 else 1.4, ms=4, zorder=3,
                label=(name if pcpg else f"{name}, {len(d['paths'])} runs") if k == 0 else None)
    ax.plot(1, 0, "*", color=_DEEP["gold"], ms=22, zorder=4, label="$w^*$")
    ax.plot(0, 0, "o", color=_CAT["text"], ms=6, zorder=4)
    ax.annotate("$w=0$", (0, 0), xytext=(8, 8), textcoords="offset points", fontsize=12)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylabel("away from it", fontsize=12)
    _finish(ax, "The path of $w$", xlabel=r"along the direction of $w^*$ (in units of $\|w^*\|$)", room=0.6,
            legend=False)
    ax.legend(loc="upper right")

    _caption(fig, ax_goal, texts["goal_text"])
    _row_caption(fig, [ax_scatter, ax_repeat], texts["step1"], texts["repeat"])
    _row_caption(fig, [ax_loss, ax_error], texts["loss"], texts["loss_error"])
    _row_caption(fig, [ax_slice, ax_slope], texts["slice"], texts["slope"])
    _row_caption(fig, [ax_curve, ax_path], texts["curve"], texts["path"])
    rows = [("Goal", texts["goal_heading"], [ax_goal]),
            ("Step 1", texts["h1"], [ax_scatter, ax_repeat]),
            ("Step 2", texts["h2"], [ax_loss, ax_error]),
            ("Step 3", texts["h3"], [ax_slice, ax_slope]),
            ("Step 4", texts["h4"], [ax_curve, ax_path])]
    _headings(fig, rows, title)
    # a white margin, so the page border of a pdf viewer does not touch the panels
    fig.savefig(out_path, bbox_inches="tight", pad_inches=1)
    plt.close(fig)


# complete data and mean imputation, fit by sklearn and by adam, against the
# number of features: a row per score (objective gap, test accuracy on the
# incomplete and on the complete test points), a column per missing rate.
# results: score -> missing rate -> name -> one value per number of features
@styled
def plot_train_sklearn(features, rates, results, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(THESIS_WIDTH, 5.4), layout="constrained", sharex=True, sharey="row")
    for row, (score, ylabel) in zip(axes, (("gap", "objective gap"), ("accuracy_missing", "acc., missing"),
                                           ("accuracy_complete", "acc., complete"))):
        for ax, rate in zip(row, rates):
            for name, values in results[score][rate].items():
                if name in _SOLID:
                    style = _SOLID[name]
                    ax.plot(features, _positive(values) if score == "gap" else values, color=style["color"],
                            marker=style["marker"], ms=3, lw=1.1, label=f"{name}, Adam")
                else:
                    _method_line(ax, features, _positive(values) if score == "gap" else values, name)
            ax.set_xscale("log")
            if score == "gap":
                ax.set_yscale("log")
        row[0].set_ylabel(ylabel)
    for ax, rate in zip(axes[0], rates):
        ax.set_title(f"{_percent(rate)} missing")
    for ax in axes[-1]:
        ax.set_xlabel("features")
    _top_legend(fig, axes[0, 0], ncol=2)
    fig.savefig(out_path)
    plt.close(fig)
