# Plotting style

Every figure is drawn with the helpers in `code/side/utils/plots.py`. Read this before writing or changing a plot, and reuse the constants listed here instead of picking new colours.

## General rules

- Plot functions only save the figure and close it with `plt.close()`. They never call `plt.show()` or `print()`; the calling script decides what to output.
- Text is set in LaTeX with Latin Modern (`text.usetex`), so it matches the thesis.
- There is no grid, and there are no top or right spines.
- Legends have no frame and usually sit at the upper left. Give each axis enough room above the data for its legend (`room` in `_finish`).
- Write math without spaces in long explanation texts (`$\sigma(z)\times{}p(z)$`), so that `_wrap` never splits a formula across lines.

## Two sizes

| | Thesis figures | Large step-by-step figures |
|---|---|---|
| rc style | `STYLE`, via the `@styled` decorator | `_BIG_STYLE`, via `plt.rc_context(_BIG_STYLE)` |
| size | `THESIS_WIDTH` = 5.51 in (= `\textwidth`), or `0.75 * THESIS_WIDTH` for one panel (`_thesis_panel`) | 16 in wide, a 6-column grid, one row per step |
| fonts | 9 pt, legend and ticks 7.5 pt (= caption size, no scaling in LaTeX) | 12–15 pt, headings 23 pt |
| line widths | the step-by-step widths × `_THESIS_SCALE` (0.45) | as listed below |
| axes and text | grey `0.3` | Catppuccin text slate `#4c4f69` |

Thesis figures go straight into the LaTeX project: `side/estimate_demo.py` writes them to both `code/side/figures/estimate/` and `thesis/figures/cfi/`.

## Colours

The palette is pastel, built on Catppuccin: the Latte neutrals, and every colour as a pastel (Mocha) and a deep (Latte) version. Every colour has **one role in every figure**, so a reader can follow a piece from step to step.

| Role | Colour | Constants |
|---|---|---|
| σ(z) and the goal area E[σ(z)] | butter yellow | family `gold`, target colour `_BAND["gold"]` = `_SIGMA_BAND` `#ffe794` |
| Densities built from p(z) (the density of z − L) and the GMM components | indigo; as a target (the followed component) mist blue | family `indigo`, target colour `_BAND["indigo"]` `#d6dcff` |
| The pieces that are summed: the sine waves and their frequencies, everything Gauss-Hermite (points, polynomials) | coral (red) | family `coral`, sums `_WAVE_LINES` |
| The fast wave, when it has to be told apart from the slow (coral) one | deep wine red | family `wine` |
| p(z) and the densities the pieces come from (the logistic threshold L, e^{-t²}) | light grey | `_CAT["grey"]` `#dce0e8` |
| Text, axes, ticks | dark slate | `_CAT["text"]` `#4c4f69` |

In the experiment figures of `estimate.py` (error against budget, number of features, …) the methods have fixed colours too (`_METHOD_STYLE`), the line colours `_LINE` of the method figures (as in the step-by-step figure): CFI, the main method, indigo with squares, conditional mean imputation coral with circles, conditional sampling (MC) pastel green with triangles, mean imputation a dashed grey line like p(z). The figure of CFI with different numbers of frequencies (`plot_cfi_frequencies`) uses four shades of indigo, `_tone("indigo", t)` for t = 0.8, 1.2, 1.6, 2.0, from few to many frequencies. The training figures of `train.py` draw the LR on complete data, the reference, as a pale wide butter band, only on complete test points (on incomplete test points it is left out); PCLR is indigo, conditional mean imputation coral with circles, mean imputation dashed grey. The legend sits above the panels (`_top_legend`).

Within a family, the *followed* piece (the one tracked through all steps, e.g. the component with the largest weight) is the darkest tone.

Do not use plain (saturated) green or purple; a pastel green is fine (family `green`). The look should stay pastel but vivid. Keep the colours of the densities (indigo) and the waves (coral) clearly different, and never let σ and the waves share a hue.

### Colours for several lines in one figure

When a figure needs up to four clearly different line colours, use, in this order: butter yellow, coral, indigo, pastel green.

- The butter yellow as a line is the butter `#ffe794` deepened slightly, `_tone("gold", 1.2)` = `#f6d876`, so it stays visible as a thin line and still looks like the yellow of the step-by-step figures. Not `_LINE["gold"]` (`#eab543`), which is too golden, and not `_DEEP["gold"]`.
- The fourth colour is a pastel green (family `green`, `_LINE["green"]`). Not pink, peach, sapphire, sky, teal or wine: they don't fit with the others.
- Lines that are variants of one method (e.g. CFI with different numbers of frequencies) are instead shades of that method's colour, clearly different from light to dark.

### Families and tones

```python
_PASTEL = dict(gold="#ffe794", indigo="#b4befe", coral="#f5a097", wine="#c97b95", green="#a6e3a1")
_DEEP   = dict(gold="#d49a00", indigo="#3b4cca", coral="#e64553", wine="#7a1f3d", green="#40a02b")
```

`_tone(family, t)` mixes a family from white (t = 0) over the pastel (t = 1) to the deep version (t = 2). The fixed tones are:

- `_LINE[f]` = tone 1.55, for lines and markers
- `_FILL[f]` = tone 1.0, for filled areas
- `_tones(f, n)`: n tones from light to dark, for a family of lines

## Targets and estimates

There are two ways to show "an estimate closing in on a target":

A target always has one colour, `_BAND[family]` (butter for σ, mist blue for a GMM component). It is used both for its band and for its area, in every step: the goal fill and the σ × p(z) area of the last step are both that colour.

1. **Only lines** (σ as a sum of sine waves): draw the target as a **pale, wide band** (`_band(ax, x, y, _SIGMA_BAND)`, lw 13, or 13 × 0.45 in the thesis). The converged sum then sits inside it like a line in a halo. Draw the partial sums on top as solid lines that get both **darker and heavier**:

   ```python
   _WAVE_LINES = ["#f7b3a8", "#ef7d70", "#d93a4a", "#5c0f1a"]   # 1, 2, 4, all waves
   _SUM_WIDTHS = (1.4, 1.9, 2.5, 3.2)
   ```

   Legend labels are just "1 wave", "2 waves", "16 waves" (`_count`), with no "sum of".

2. **Lines over areas** (an estimate times p(z) against the target times p(z)): draw the target area in its target colour (`_BAND[target]`), the estimate area in `_FILL[family]` with alpha 0.75 and a thin `_LINE` outline, the target curve **dashed** on top (`_dashed`, so it stays visible where the estimate matches it), and the estimate curve solid in `_LINE[family]`.

Signed areas (e.g. a wave × p(z)) are filled from 0, with a thin zero line in `_CAT["surface"]`.

## Layout of the step-by-step figures

- There's no figure title. Each step is introduced by a centred small uppercase badge (`GOAL`, `STEP 1`, …) on a short thin rule, with a bold centred heading below it (`_headings`).
- There are no cards, outlines or background fills; the page is plain white.
- Every panel has a bold title. Each row (step) has one explanation, centred below the whole row over nearly its full width and wrapped by visible characters (`_row_caption`, `_caption` and `_wrap`), in the text colour. The texts of a two-panel row are joined into one.
- When two things of the same kind are compared (e.g. a slow and a fast wave), they share one panel, in two clearly different tones of their family (coral and wine for the waves), as in the thesis figures.
- The goal panel spans the middle 4 columns, two-panel rows take 3 columns per panel, and progressions take 2. Keep the figure compact: don't make it wider, and show at most three panels in a row.
- Panels that show a progression (1, 2, all pieces; `_first_two_and_all`) sit side by side and share their y-axis. Only the first one carries the legend. In the line-only sums of step 2 there are four sums (1, 2, 4, all).
- The figure is saved with `bbox_inches="tight", pad_inches=1`, so a PDF viewer's page border doesn't touch the panels.
