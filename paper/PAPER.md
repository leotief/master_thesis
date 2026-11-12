# PAPER.md

The paper is the main focus of the project. It is written for ICML in the official template; the thesis is written afterwards, based on the paper.

## Layout

- `main.tex` — preamble, front matter (title, author, abstract) and the required end sections; `\input`s `body.tex`. Built from `template/example_paper.tex`.
- `body.tex` — the whole main body, all sections in one file (the paper is short: at most 8 pages).
- No appendix for now: the focus is the main body (at most 8 pages, figures included). An appendix comes later.
- `references.bib` — bibliography (natbib + `icml2026.bst`, i.e. bibtex, not biber).
- `tables/` — tables of the paper, written by scripts in `code/` (`real_data.py` writes `real_data_log_loss.tex` and `real_data_accuracy.tex`, LaTeX tabulars to `\input` in a `table`/`table*`).
- `figures/` — figures of the paper. Scripts in `code/` that make paper figures write them here (as they write thesis figures to `thesis/figures/`). The style rules of `code/PLOTTING.md` apply, but sized for the ICML column (`\columnwidth` ≈ 3.25 in, `\textwidth` ≈ 6.75 in) instead of the thesis text width.
- `icml2026.sty`, `icml2026.bst`, `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty` — the template's style files. Do not edit them.
- `pghs_alternatives.md` — short notes on why PGHS fails for a wide p(z | x_obs) and how CFI replaces it (the derivation in simple terms and the first numbers); background for the rewrite, not part of the paper.
- `template/` — the untouched template (`example_paper.tex`, its `.bib`, figure and PDF). It is the reference for the formatting rules (8 pages main body, required Impact Statement, anonymous submission, etc.). **Do not change it.**

Section titles say what happens in the section (e.g. "From the Sigmoid to the Characteristic Function"), not generic names like "Background" or "Method". Only Introduction, Experiments and Conclusion keep their generic names. The first draft of `body.tex` is written from the thesis (methods, experimental results, related work), condensed and partly rewritten for the paper; not everything from the thesis is in it. Display equations must fit in one column (≈ 3.25 in): split long ones into short `align` rows. Wide figures are `figure*` and are defined one section before the text that discusses them, so that they are placed near it. `todonotes` is loaded with `disable`.

Build: `cd paper && make pdf` (pdflatex + bibtex, output `main.pdf`); `make clean` removes the auxiliary files.

Submission options in `main.tex`: `\usepackage{icml2026}` is the blind review version; `[preprint]` or `[accepted]` for the other versions. Author information is removed automatically in the blind version.

## Template migration (pending)

The template is **ICML 2026** (`icml2026.sty` of 2025-10-29). A new version of the template will be released; when it is, the paper has to be migrated: replace the style files and the `.bst`, put the new template into `template/`, and diff the new `example_paper.tex` against the old one to carry over changes in the preamble, the front matter and the required end sections into `main.tex`.

## Framing (story of the paper)

1. Probabilistic circuits (PCs) are tractable density models: marginals and conditionals are exact, so missing features can be handled exactly.
2. PCs are usually not used together with a logistic regression (LR), because the sigmoid destroys the tractability: E[σ(wᵀx) | x_obs] under a PC has no closed form.
3. To classify with a PC one usually trains a joint model p(x, y) and computes the conditional p(y | x). In practice this often hurts the classifier's performance compared to a discriminative model.
4. Our approach computes E[σ(wᵀx) | x_obs] from the PC's characteristic function alone, deterministically, differentiably and accurate to float32 precision: **CFI** (characteristic-function inversion). The sigmoid is the CDF of the standard logistic distribution, so the expected prediction is the probability that z − L > 0; the characteristic function of z − L is the PC's times πs/sinh(πs), and the Gil-Pelaez inversion formula turns it into a one-dimensional integral, ½ + ∫₀^∞ Im φ_z(s)/sinh(πs) ds, computed by the trapezoid rule.
5. With it, PC and LR form one joint model with their own parameters, **PCLR**: the PC models p(x), the LR models p(y | x), the likelihood of an incomplete point splits into log p(x_obs) + log p(y | x_obs), and both are trained jointly on incomplete data. Missing features are marginalised exactly, in prediction and in training.

The introduction leads with this framing. The experiments (estimation accuracy and time: `code/estimate.py`; training on incomplete data and the learning curve: `code/train.py`) support it. See `code/CODE.md`, "Status and findings", for the current results.

## Status of the paper (November 2026)

**`body.tex` is stripped down (2026-11-11).** Everything that belonged only to PGHS (the derivation, the quadrature, its figures and old experiments) was removed. What is left: the introduction up to the framing, the setup (PC and LR), the characteristic function of a circuit, the real-data tables (`\input` from `tables/`, as `table*`), and the related work. `% TODO` markers stand where CFI, PCLR, the other experiments and the conclusion have to be written. The current text to build on is the thesis methods chapter (`thesis/chapters/methods.tex`), which was rewritten for CFI and PCLR: core idea (CF of z, sigmoid as a probability, CF of the difference, Gil-Pelaez inversion, full formula), computing the integral (trapezoid rule, truncation, step size), the CF and moments of GMM and PC, the CFI algorithm, PCLR (joint model, objective, density and label terms, gradients, fixed density, algorithm), PGHS as the earlier route, and the direct GMM reference. Background in `thesis/main.tex` is commented out; the thesis experiments, discussion and appendix chapters still describe PGHS.

Why PGHS was replaced: its inner integral carries a factor e^{1/(8γ)} that cancels catastrophically for small γ (and makes the MC variant have infinite variance), and the sum of Gaussian bells approximates the sigmoid badly for large positive z, which the sign flip only fixes when p(z | x_obs) lies on one side of 0. So PGHS breaks down for a wide p(z | x_obs), i.e. many missing features. CFI has neither problem.

**Figures for the new experiments** (in `thesis/figures/`, to be resized for the ICML column):
- `estimate/gmm/comparison_0.5_1000_100_norm.pdf` — CFI vs. MC vs. mean imputation, KL and time against the number of features;
- `estimate/gmm/cfi_frequencies_0.5_1000_100_norm.pdf` — CFI with 8–64 frequencies;
- `train/gmm/training_complete_test_1.0.pdf`, `training_incomplete_test_1.0.pdf` — the training comparison on complete / incomplete test points;
- `train/gmm/weights_1.0.pdf` — the relative error of w against the number of features;
- `train/gmm/convergence_1.0.pdf` — the error of w against the number of training points, on the earlier LR-generated data (removed). Replaced by the learning curve `train/gmm/learning_curve_<INFORMATIVE>_<LAMBDA>.pdf` on `make_classification` (not run yet).

**Related work to cite and discuss:** Khosravi et al., "What to Expect of Classifiers? Reasoning about Logistic Regression with Missing Features" (IJCAI 2019; expected LR prediction is hard even under naive Bayes) and "On Tractable Computation of Expected Predictions" (NeurIPS 2019; moments / Taylor series for classification) — the closest prior work and a natural baseline; Yu, Trapp, Kersting, "Characteristic Circuits" (NeurIPS 2023) and Broadrick et al., "Polynomial Semantics of Tractable Probabilistic Circuits" (UAI 2024) — PCs give the characteristic function tractably; Gil-Pelaez (1951) — the inversion formula; Pirjol (2013) and Monahan & Stefanski — the logistic-normal integral. Frame the result as an error-controlled approximation to float precision, not exact tractability (the exact problem is hard).

**Open before writing:** see "Experiment plan" below; also C chosen per method and the Khosravi baseline (see `code/CODE.md`, "Open points").

## Experiment plan

All experiments use data out of the box: sklearn's `make_classification` and real datasets, with every entry missing independently with probability 10, 30 or 50 % (MCAR). There is no ground truth of the data and no true `w` (the LR is not the true model of `make_classification`), so the training experiments score predictions on held-out labels, as is usual for real data.

1. **Estimation** (`code/estimate.py`): CFI against conditional sampling (MC), the conditional mean and mean imputation, KL to the exact value under the fitted density and time, against the number of features, on raw `make_classification` (the spread of z grows with the features, which CFI handles).
2. **Training** (`code/train.py`): every model trained on incomplete points, log-loss and accuracy on complete and on incomplete test points, against the number of features and the missing rate; and the learning curve, log-loss and accuracy on incomplete test points against the number of training points.
3. **Real data** (`code/real_data.py`, a table per missing rate): MAGIC, Phoneme, Spambase, Banknote, QSAR, Pima, MCAR at 10/30/50 % (every entry independently; 10–30 % is typical in the literature, 50 % a stress test), log-loss, accuracy/AUC, mean ± std over seeds. Competitors: mean imputation, conditional mean, multiple imputation (MICE, sklearn's `IterativeImputer`), the generative classifier (one density with the label inside), if possible SAEM (R package `misaem`).

Still to do: several seeds, possibly MAR missingness, and a PC as the density (everything runs on the GMM so far). Dropped: the consistency claim for w (it needs data from a known LR, `load_lr_data`, which was removed for out-of-the-box data).

## Writing rules

The thesis writing rules in the root `CLAUDE.md` apply to the paper as well. The thesis chapters in `thesis/chapters/` are a source of text and derivations to condense, not to copy: the paper is shorter and two-column.
