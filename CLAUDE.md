# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It is a summary; the details are in the files it points to.

## Current focus: the ICML paper

The focus is now the **paper** in `paper/` (ICML 2026 template), not the thesis. The thesis is not written on for now; once the paper is done, the thesis is written based on it. **Read [paper/PAPER.md](paper/PAPER.md) before working on the paper**: layout, build, the framing of the paper, and the pending template migration (a newer ICML template will come out and the paper has to be migrated to it).

In one sentence: a probabilistic circuit (PC) is tractable, but it is usually not combined with a logistic regression because the sigmoid destroys the tractability. Training a joint model and taking the conditional often gives a weak classifier. Our method, **CFI** (characteristic-function inversion), computes E[σ(wᵀx) | x_obs] from the PC's characteristic function by the Gil-Pelaez formula, so PC and LR form one joint model with their own parameters, **PCLR**, trained jointly on incomplete data. The earlier method, PGHS (Pólya-Gamma + Hubbard-Stratonovich, formerly called PCPG), is replaced; `paper/body.tex` still describes it (see PAPER.md, "Status of the paper").

## Repository layout

- `paper/` — the ICML paper. `main.tex` (preamble, front matter) + `body.tex` (main body, max 8 pages) + `appendix.tex` + `references.bib`, build with `cd paper && make pdf`. `paper/template/` is the untouched ICML template, kept as the reference for the formatting instructions. Do not change it.
- `code/` — Python research code (PyTorch). **Read [code/CODE.md](code/CODE.md) before working on the code**: files, the methods (CFI, PGHS, baselines), the `PCLR` model, density models, the experiments `estimate.py` and `train.py`, pitfalls, the current findings and open points. Every plot follows [code/PLOTTING.md](code/PLOTTING.md).
- `thesis/` — LaTeX source of the master's thesis (Karl Voit template), paused. Build with `cd thesis && make pdf` (plain `make` only prints the help; pdflatex + biber). Its chapters are a source of text for the paper; the methods chapter is current (CFI and PCLR), the others still describe PGHS.

Run Python with `~/.pyenv/versions/ml/bin/python` (it has all dependencies), from `code/` (the demos in `code/utils/` from there).

## Preferences

When the user says "remember" something, save it in this repo (in this file, or in `code/CODE.md` / `paper/PAPER.md` if it is about the code or the paper), not in Claude's memory directory.

Writing (paper and thesis):
- Never put explanations beside equations (no `&& \text{...}` annotation columns in `align`) or beside algorithm lines (no `\Comment{...}` or `\hfill` notes in `algorithmic`). Explain the steps in prose before or after.
- Write each equation row on one source line; don't wrap a row over several LaTeX lines.
- When asked to simplify, change only what is needed; keep the existing steps, order and level of detail.
