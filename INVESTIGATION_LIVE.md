# LIVE INVESTIGATION LOG — filter divergence (Sobel/Laplace vs paper)

> **This file is the running context record. Update it as work proceeds.**
> If the session is lost, read this first.

Last updated: 2026-10-21 11:35 +06

---

## 0e. FRESH START ON THE CLARK GENERATOR (2026-10-21 11:30 +06)

User decision: abandon the old-generator branch, start absolutely fresh with
the corrected (verbatim Clark et al.) data pipeline.

- Old results archived to `results_legacy_old_generator_2026-10-21.tar.gz`
  (36 MB, repo root, gitignored). `results/` emptied; only `figures/` recreated.
  The md reports (LOCKED_DATAPOINTS, FILTER_DIVERGENCE_REPORT) remain in git
  history for citation in the final report.
- User CANCELLED the 5 in-flight old-generator kernels (xor-corr-sweep-3..6,
  rigid-white-sweep-5): verified `cancel_acknowledged`.
- PUSHED wave 1 of Clark data-gen (11:31, all `running` as of 11:33):
    phase2-clark-lin-corr, phase2-clark-lin-white, phase2-clark-xor-corr,
    phase2-clark-xor-white, phase2-clark-rigid-corr
  (rigid-white queued for the next free slot).
  Core dataset already refreshed with the clark_xai_tris vendor (11:08).

### Plan from here (the queue)
1. WAIT for the 5 data-gen kernels; `pipeline.py results <phase>` + check
   PASS/90% gate per alpha; record mask_px/Eq13 stats + pattern norms from
   [RESULT] lines.
2. `fetch` outputs; compare training_summary accuracies vs the paper's
   (paper: ~0.9-1.0 across alphas for their MLP).
3. Push phase2-clark-rigid-white when a slot frees.
4. Build 54 Clark sweep kernels: copy the sweep templates, retarget
   `kernel_sources` in config.yaml to the `phase2-clark-*` data kernels, push
   in waves of 5 (order: xor-corr (completes the paper-comparable table
   first), rigid-corr, lin-corr, lin-white, xor-white, rigid-white).
5. Regenerate six-panel EMD/IME curves (scripts/plot_curves.py) and redo the
   ours-vs-paper comparison vs `Baseline Paper/*-datapoints.md`.
6. Final report + EXPERIMENT_LOG.md update; push origin.

### Watch items
- Accuracy gate: Clark's alpha mapping differs; alpha=0.1 may be near/below
  the 90% gate in some scenarios (their own configs used alpha 0.03-0.65).
  If a point fails, document - do not retune.
- RIGID = single tetromino per shipped code (text-vs-code divergence
  documented in the vendor README).
- Old-generator vs new: the archived tarball lets us quantify exactly how much
  of the Fig-4 divergence was generator semantics, if asked.

---

## 0c. BREAKTHROUGH (earlier): original XAI-TRIS generator found PUBLIC

`github.com/braindatalab/xai-tris` (Clark et al.'s own repo, GPL-3.0, updated
2025-05). Contains:
- `data/generate_data` with `data/data_config.json` (64x64 + 8x8 modes)
- scenarios: `linear | xor | translations_rotations` (= LIN/XOR/RIGID)
- backgrounds: `uncorrelated | correlated | imagenet` (= WHITE/CORR)
- `eval/quantitative_metrics` (their EMD/IME implementations)
- also 8x8 mode used by some of their results

**This eliminates the one uncontrolled factor in the divergence attribution**
("our generator reimplemented from prose may differ from theirs"). Path:
1. Clone/port their generator; pixel-level diff vs ours (mask px, geometry,
   smoothing, noise normalization).
2. If difference is a placement/mask subtlety -> align OUR generator to theirs
   (lowest risk; sweep machinery unchanged). Else swap the generator entirely.
3. Regenerate data -> re-run data-gen kernels -> re-run sweeps.
4. Either clean full reproduction, or a bulletproof reproducibility finding.

**HOLD: do not push new sweeps** beyond the 5 in flight until the generator
decision lands (avoid double compute). In-flight: xor-corr 3,4,5,6 +
rigid-white 5. Let them finish.

---

## 0d. GENERATOR DIFF RESOLVED (2026-10-21) — root cause identified + clark kernels built

Their generator (`reference/xai-tris`, cloned today) vs our per-sample-L2
reimplementation — exact diffs, and which matter:

| aspect | theirs (Clark/PAPER Eq.12) | ours | impact |
|---|---|---|---|
| normalization | **GLOBAL** Frobenius over the full 40000-sample matrices (paper Eq. 12 text confirms: "horizontally concatenated into matrices and normalized by the Frobenius norm") | per-sample L2 | **ROOT CAUSE of scale offset** — different alpha<->difficulty mapping; per-sample SNR also varies sample-to-sample in theirs, constant in ours |
| rescale | 1/max\|x\| over whole dataset, then float16 | none, float64 | minor |
| background | N(0, **0.5**) per px (their `var_data` used as sigma) | N(0,1) per px | changes effective alpha scale |
| pattern | 3x2-cell t/l via kron: 24x16 px (scale 8 LIN/XOR); **12x8 px (scale 4 RIGID)** | 3x3-cell T/L, 24x24 px everywhere | mask size/geometry differ everywhere |
| RIGID structure | **single tetromino per sample** (t vs l), random rot90 + random place anywhere | composite aT±aL, quadrant-constrained | paper *text* says composite; their *code* says single. We follow SHIPPED CODE (defensible), document the text-code divergence |
| XOR structure | t@(8,8) & l@(32,40), 4 sign-manips; y=0:(++),(--), y=1:(+-),(-+) — structurally EQUIVALENT to ours | fixed T/L, T±L + global flip | equivalent |
| smoothing | gaussian sigma=1.5, **no 5% support threshold** in generator | same blur + 5% relative threshold | the threshold lives in paper Eq. 13 (mask def), not the generator — ours is right per paper |
| split | StratifiedShuffleSplit 90/5/5, SEED=4000002 | random perm 90/5/5, seed crc32 | equivalent |
| eval nuance | Clark's eval uses absolute-0.05 mask threshold (not rel 5%) and EMD score = 1 - cost/Dmax (similarity) | Eq. 10 raw OT cost | PatternLocal paper redefined the metric; keep Eq.10 |

**Decision: swap generator to theirs, verbatim.** Built this session:
- `src/paper_repro/clark_xai_tris/` — vendored upstream files
  (minimal relative-import patch only, GPL-3.0, README with provenance)
  + `conversion.py` (pickle -> npz, masks per paper Eq. 13/14 relative-5%
  sign-preserving; LIN = Eq.14 union; also stores Clark's abs-0.05 masks +
  raw masks for diagnostics) + `kaggle_datagen.py` (runs their
  `data_generation_process` verbatim, trains the SAME MLP - which matches
  the paper's stated 128/64/32/16 + 25% dropout architecture).
- `templates/phase2_clark_{lin,xor,rigid}_{white,corr}.py` (6 thin kernels)
  + config.yaml phases `phase2_clark_*` (dataset mounts: core only).
- Local smoke of conversion passed (XOR sign-sym masks 432px both classes;
  LIN union 862px; abs05 correctly halves XOR support).

**Next:** (a) `pipeline.py upload-src` to refresh core dataset with the
vendor; (b) when the 5 in-flight old-generator kernels finish, push Clark
data-gen (5 slots: lin-corr, lin-white, xor-corr, xor-white, rigid-corr;
rigid-white follows); (c) verify accuracies + mask stats; (d) point the
sweep templates' kernel_sources at `patternlocal-repro-clark-*` outputs and
re-run all 54 sweeps on the new data.

---

## 0. What is running RIGHT NOW on Kaggle

State at last check (2026-10-21 01:13 +06):

| Kernel | alpha | Status |
|---|---|---|
| `rigid-corr-sweep-1..5` | 0.1-0.5 | **running (CPU)** |
| `rigid-white-sweep-5` | 0.5 | cancelled (no GPU quota) -> re-push on CPU |

### Sweep queue (43 truly missing, 5 in flight)

```
have: xor-corr-sweep-1,2,9; lin-white-sweep-5; rigid-corr-sweep-6(old, INVALID);
      rigid-white-sweep-5 (old, INVALID)
NOTE: rigid-corr-sweep-6 and rigid-white-sweep-5 in results/ are from the OLD
broken generator (acc 0.9535/0.9955) and are INVALID -- excluded everywhere.
```

| scenario | missing parts |
|---|---|
| xor-corr | 3,4,5,6,7,8 |
| xor-white | 1..9 |
| lin-corr | 1..9 |
| lin-white | 1,2,3,4,6,7,8,9 |
| rigid-corr | 1,2,3,4,5,7,8,9 (1-5 in flight) |
| rigid-white | 1,2,3,4,5,6,7,8,9 |

Push order (5 at a time as slots free):
1. xor-corr 3,4,5,6 + rigid-white-sweep-5 (the cancelled TODO)
2. xor-corr 7,8 + lin-corr 1,2,3
3. lin-corr 4..8
4. lin-white 1..4,6..9 + xor-white 1..9
5. rigid-corr 7,8,9 + rigid-white 1..4,6..9

## 0b. Decision log

* 2026-10-21: filter divergence investigation CLOSED (see
  results/FILTER_DIVERGENCE_REPORT.md final sections). Root cause: the paper's
  data has a global EMD floor (~0.043) that ours lacks; affects ALL methods, so
  it is a generator difference, not a filter bug. User approved: document and
  move on, do not tune.
* 2026-10-21: next effort = complete all six curves (user approved).


---

## 1. The problem

Our XOR CORR Sobel/Laplace are **far better than the paper's**, and improve with
alpha where the paper's stay flat.

| alpha | method | ours | paper |
|---|---|---|---|
| 0.1 | Sobel | 0.1223 | 0.140 |
| 0.1 | Laplace | **0.0920** | 0.135 |
| 0.9 | Sobel | **0.0200** | 0.125 |
| 0.9 | Laplace | **0.0162** | 0.080 |

Paper's Remark (line 2006-2007 of the txt):
> "For edge-detector baselines such as Sobel and Laplace, increasing alpha sharpens
> tetromino boundaries and causes the filters to respond outside the true region,
> increasing the EMD/IME."

The paper's filters are **flat at ~0.12-0.14 across all alpha** and never improve.

Paper data points written by hand: `Baseline Paper/xor-corr-datapoints.md`

---

## 2. What has been RULED OUT (do not chase these again)

| Item | Verdict |
|---|---|
| Mask definition (`xai_tris.py` line 247 for XOR, 248-251 for LIN) | **CORRECT.** Matches paper Eq. 13 (support of full signal) and Eq. 14 (union for LIN). Verified: grayscale 1054 px union == `T>0 \| L>0`. |
| `delta_max` = 64x64 diagonal = 89.095 | **CORRECT.** Matches "maximum Euclidean distance between any two pixels". |
| EMD form `OT(\|s\|, F+)/delta_max`, mass-normalized, POT `emd2` | **CORRECT.** Matches Eq. (10) + cited Bonneel/POT. |
| IME `1 - sum_F+\|si\|/sum\|si\|` | **CORRECT.** Matches Eq. (11). |
| EMD **scale** | **FINE.** My earlier "15-20x gap" was a bad PDF tick read. Paper XOR CORR ticks are 0.03-0.15; ours 0.02-0.13. |
| RIGID generator bug | Fixed earlier (accuracy 0.48 -> 0.99). Separate issue. |

---

## 3. The decisive measurement

Recomputed Sobel/Laplace on the **actual saved** XOR CORR alpha=0.9 samples
(`/tmp/b1/xor-corr-sweep-9/alpha0.9_sample_images.npz`):

```
XOR CORR alpha=0.9, sample 0 (of 5):
  Sobel   EMD = 0.0064   mass inside F+ = 88.8%
  Laplace EMD = 0.0021   mass inside F+ = 70.4%
  LIME    EMD = 0.0765
  F+ = 1054 px = 25.7% of image, bbox rows 6-57 x cols 6-57
```

**Interpretation.** F+ is the union of both tetrominoes -> a big blob covering 26%
of the image, bbox spanning nearly the whole frame. Sobel/Laplace fire on every
tetromino boundary, and every one of those boundaries is INSIDE F+. So the filters
get ~88% of their mass counted as hits **for free**, and transport distance is
~0.6 px. That is the whole effect.

`delta_max=89.1` is fixed. So EMD 0.0064 <=> mean transport 0.6 px.
Paper's Sobel 0.125 <=> mean transport ~11 px.

**Same mask. Same metric. Same delta_max.** The difference is that the paper's
filters put substantial mass OUTSIDE the tetromino boundaries; ours do not.

---

## 4. Hypotheses to test (in order)

### H1 — `scipy.ndimage` default differs from the paper's "default implementation"
Paper line 1703: "We run with their default implementations for filter-based methods
(Laplace, Sobel)". Candidates:
- `scipy.ndimage.sobel` (ours) vs `skimage.filters.sobel` vs `cv2.Sobel`.
  Different kernel normalization + border handling.
- `ndimage.laplace` 3x3 kernel is `[[0,1,0],[1,-4,1],[0,1,0]]`.
  `cv2.Laplacian` ksize=1 uses the same; ksize=3 uses a different (8-neighbour) kernel.

### H2 — filter applied to a different quantity
- Applied to the **raw** image `x` (ours) vs to the **signal component only**,
  vs to the **un-smoothed** signal `a`.
- Paper generation: `x = alpha*(R o (H o a)) + (1-alpha)*(G o eta)`.
  "Respond outside the true region as alpha increases" is consistent with the
  filter seeing a *sharpening* signal, i.e. edges bleeding outward.

### H3 — sign / magnitude handling
- We take `|laplace(x)|` and `sqrt(gx^2+gy^2)`.
- Alternative: signed output (negative lobes preserved), or `-laplacian` (which is
  the "blob/edge detector" sign convention), or `max(|gx|,|gy|)`.
- IME's `sum |si| over F+` and the EMD's `|s|` both hint sign structure matters.

### H4 — smoothing / pre-blur
- Pre-blurring the image before filtering changes ridge width and where mass lands.
- `sigma=1.5` is the signal smoothing; maybe the filter sees `sigma=0` (sharp) input.

---

## 5. Success criterion

A variant is a **match** if, on XOR CORR, Sobel stays roughly flat (~0.12-0.14)
across alpha=0.1..0.9 AND Laplace stays at ~0.08-0.135, i.e. both remain clearly
worse than PatternLocal at high alpha and never improve with alpha.

If a variant matches -> it is a genuine bug in `baselines.py` -> affects ALL 54
sweeps -> PAUSE new pushes, fix, re-run. (The in-flight RIGID CORR sweeps still
give valid PatternLocal/LIME numbers, so they are not wasted.)

If no variant matches -> document as an irreproducible paper/implementation
ambiguity and report our numbers with the caveat.

---

## 6. Test harness

Script: `scripts/filter_probe.py`
Uses the 5 saved samples per (scenario, alpha) `.npz` to *recompute* the
explanation variants locally and score them with the real `compute_emd`.
(Only 5 samples, so it is indicative, not exact — but the effect size is huge
(0.006 vs 0.125), so 5 samples is plenty to discriminate.)

---

## 7. Findings so far

### 7.1 Variant sweep result — NONE of the "filter definition" variants fix it

Ran `scripts/filter_probe.py` (12 variants x 6 saved datasets). Representative
XOR CORR rows (F+ = 1054 px = 25.7% of image):

| variant | a=0.1 | a=0.2 | a=0.9 | mass-in-F+ @0.9 |
|---|---|---|---|---|
| sobel_ndimage (OURS) | 0.0884 | 0.0829 | 0.0081 | 88.0% |
| sobel_3x3_kernel | 0.0792 | 0.0765 | 0.0116 | 85.9% |
| sobel_absmax | 0.0778 | 0.0756 | 0.0110 | 86.0% |
| sobel_sumabs | 0.0816 | 0.0779 | 0.0120 | 85.7% |
| sobel_blur1.5 | 0.0900 | 0.0862 | 0.0105 | 78.9% |
| laplace_ndimage (OURS) | 0.0629 | 0.0430 | 0.0024 | 70.3% |
| laplace_signed / neg | 0.0629 | 0.0430 | 0.0024 | 70.3% |
| laplace_8kernel | 0.0335 | 0.0312 | 0.0113 | 64.0% |
| laplace_blur1.5 | 0.0742 | 0.0554 | 0.0087 | 66.9% |
| **paper target** | **0.140** | **0.130** | **0.125** | — |

**Every** variant collapses at a=0.9. Kernel choice, sign, blur, 3x3-vs-separable
all make <0.02 difference. So H1/H3/H4 are **dead**: it is not the filter
implementation.

### 7.2 The mask is the real lever

The mask geometry is the cause. XOR F+ (union of both tetrominoes) is a big blob;
Sobel's edge ridge traces that blob's outline, which lies INSIDE F+. Mass-in-F+ =
88%, so mean transport ~0.6 px vs the paper's ~11 px.

Explicitly tested mask alternatives on XOR CORR a=0.9 (5 samples):

| mask | SOBEL | LIME | PL | px |
|---|---|---|---|---|
| union T|L (ours, = Eq.13) | **0.0081** | **0.0737** | 0.0003 | 1054 |
| T only | 0.1441 | 0.2160 | 0.1008 | 528 |
| L only | 0.1403 | 0.2216 | 0.1258 | 526 |
| **paper target** | **0.125** | **0.080** | **0.048** | — |

Note: the **union** already reproduces the paper's *LIME* (0.074 vs 0.080) but
makes Sobel trivial. "T only" reproduces the paper's *Sobel* (0.144 vs 0.125) but
breaks LIME (0.216 vs 0.080). **No single mask reproduces both.**

This is a genuine tension in the paper's setup that cannot be resolved by any
choice of filter kernel or mask we have tried.

### 7.3 Mask IS literally correct per the paper text

`F+ = support(R o (H o a))` (Eq. 13). `a = aT - aL`; support = union since
overlap = 0 px. Our code produces exactly 1054 px = union. Verified byte-identical.
`smooth_signal` already applies the paper's "5% maximum support threshold" (line
1506). So the mask matches the paper's stated definition.

### 7.4 LIN WHITE check (mask rule unambiguous — Eq. 14 says union)

| | ours (full set, POT) | paper |
|---|---|---|
| Sobel | 0.1150 | ~0.23-0.30 |
| Laplace | 0.1204 | ~0.23-0.30 |
| mass-in-F+ | ~30% | — |

Same direction: our filters are ~2x better than the paper's even in LIN, where
the mask rule is explicitly the union.

### 7.5 Reference implementation

Paper repo `https://github.com/gjoelbye/PatternLocal` ships ONLY the PatternLocal
method (solvers, superpixel simplification). It does **NOT** ship the XAI-TRIS
generator (that is from Clark et al. [8]) and does **NOT** ship the filter
baselines or the EMD/IME evaluation. So the filter+mask+metric path cannot be
verified against their code.

README confirms for image data: `SuperpixelSimplification` with SLIC is required
for image mode; `LIME image mode` uses `num_samples=1000`, `bandwidth=0.25`.

### 7.6 Current verdict

The filter divergence is **NOT** caused by:
- filter kernel / sign / blur choice (ruled out, 7.1)
- `delta_max` (correct, 2)
- EMD/IME formula (correct, 2)
- mask definition per the paper's own Eq. 13 (correct, 7.3)

It **IS** characterized by: our filters put ~88% of mass inside F+, the paper's
must put far less. The paper's Fig-4 numbers for XOR CORR are internally
inconsistent with a single union mask across all methods (Sobel needs a
1-tetromino mask, LIME needs the union).

Open possibilities for the residual gap:
- [ ] The paper's Fig-4 numbers are **visually estimated** (the user's md says
      "visually estimated" from a screenshot) — the ticks may be off.
- [ ] The paper may use a **different F+ for different metrics/methods**.
- [ ] The paper's filters may be evaluated on the **noise-free signal**, not x.
- [ ] The paper's numbers may include the alpha-dependent mask shrink.

**Decision:** document honestly. Do NOT tune to match. Report our numbers with the
caveat that the filter result is not reproducible under the paper's stated
definitions.

### 7.7 DIRECT raster extraction of the paper's Figure 4 (decisive)

Rendered page 8 at 600 dpi and traced the curve pixels programmatically
(`pdftoppm -f 8 -l 8 -r 600`), so this is the paper's **actual plotted data**,
not a visual estimate.

Calibration (XOR EMD panel):
* x-axis frame: 2187..3091 px; 8 alpha labels at 0.2..0.9, spacing 116.5 px
* y-axis frame: y=2970 (0.19) .. y=4318 (0.01); tick 0.13 confirmed at y=3397

Extracted XOR CORR EMD (paper, from raster):

| color | a=0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| orange/brown | 0.013 | 0.013 | 0.012 | 0.012 | 0.013 | 0.014 | 0.016 | ? |
| lightblue | 0.046 | 0.047 | 0.048 | 0.049 | 0.050 | 0.051 | 0.051 | 0.053 |
| blue | 0.173 | 0.169 | 0.167 | 0.166 | 0.166 | 0.166 | 0.166 | 0.167 |

Two clean, unambiguous curves:
* one **rises monotonically 0.046 -> 0.053** (never improves with alpha)
* one is **flat at ~0.167** (never improves with alpha)

Both confirm the paper's Remark. **This is the paper's real behaviour and it is
structurally different from ours** (ours falls to 0.008-0.020 at a=0.9).

### 7.8 The user's hand-transcribed table is inaccurate

`Baseline Paper/xor-corr-datapoints.md` says it is "visually estimated" from a
screenshot. The raster shows the paper's Sobel/Laplace-type curves at
**~0.16-0.17 flat**, whereas the md lists Sobel 0.125-0.140 and Laplace
0.08-0.135. The md also lists highs (0.140 at a=0.1) that the raster does not
support. **Treat the md table as indicative only; the raster is authoritative.**

### 7.9 FINAL ROOT CAUSE

The paper's filters put their mass where F+ is NOT, and this is the *entire*
difference. Our filter mass lands inside F+ (88% at high alpha) because F+ for XOR
is the union blob and the tetromino edges trace its interior/boundary.

The paper's generator is **not shipped** (Clark et al. [8], not in the PatternLocal
repo), and the paper does not state the filter input, so the exact reason their
filters scatter mass outward is **not determinable from the paper**. Candidate
explanations that remain (none testable without their generator):
1. The paper applies Sobel/Laplace to the **raw un-normalised** image at a scale
   where the CORR noise dominates (ours is Frobenius-normalised), so the filter
   responds to *noise structure* (suppressor pixels) rather than the shape.
2. Their F+ for XOR may differ per method.
3. Their tetromino placement/size may leave the shape edges partly OUTSIDE F+.

**Decision: report honestly.** Our reproduction reproduces the paper's *method*
claims (PatternLocal beats LIME, IME collapse) but NOT the *filter* baseline
numbers. Document as a known, characterised non-reproduction.

### 7.10 POT verification (definitive — our metric is NOT the bug)

Installed the **exact** library the paper uses (`POT 0.9.7`) and recomputed the
saved XOR CORR a=0.9 samples:

| method | per-sample POT EMD | Kaggle full-set | paper raster |
|---|---|---|---|
| Sobel | 0.0180-0.0218 | 0.0200 | ~0.167 |
| Laplace | 0.0159-0.0166 | 0.0162 | ~0.167 |
| LIME | 0.099-0.104 | 0.1012 | ~0.05 |

**Our EMD matches Kaggle exactly.** So the metric, the POT call, and the pipeline
are all correct. The divergence is in the DATA / EXPLANATION, not the scoring.

Also: even feeding the **raw input image x** as an "explanation" gives EMD 0.023
against our F+. Our F+ is so large (26%, filled blobs) that anything spatially
near the shapes scores ~0.02. The paper's filters score 0.167 -> their F+ must be
much smaller *relative to where the filters put mass*.

### 7.11 Mask-size test (the real lever, quantified)

On XOR CORR a=0.9, swapping the mask:

| mask | px | Sobel | LIME |
|---|---|---|---|
| union filled (ours, Eq.13) | 1054 | 0.0081 | 0.0737 |
| union outline | 181 | 0.0202 | 0.1021 |
| T filled only | 320 | 0.2395 | 0.2622 |
| T outline only | 90 | 0.2327 | 0.2511 |
| **paper target** | — | **~0.167** | **~0.05** |

**No mask reproduces BOTH** the paper's Sobel (~0.167) and LIME (~0.05)
simultaneously. A small mask fixes Sobel but ruins LIME.

### 7.12 CONCLUSION

The paper's XOR CORR filter numbers are **not reproducible** from the paper's
stated definitions (Eq. 10-14 + default Sobel/Laplace). We have tested and
excluded, with numbers:
- filter kernel/sign/blur (12 variants) — no effect
- delta_max, EMD formula, POT library — verified correct vs the paper's own lib
- mask definition per Eq. 13/14 — verified byte-identical
- separate normalization + alpha mixing (Eq. 12) — verified matching
- mask erosion / outline / single-shape — none reproduces the full pattern

The residual is a **characterised, honest non-reproduction**, most likely because
the paper's XAI-TRIS generator (Clark et al. [8], not shipped) differs from ours
in tetromino placement/size or in how the CORR noise interacts with the shapes.
This must be stated in the report rather than tuned away.

**What DOES reproduce (independent of the filter issue):**
- PatternLocal beats LIME on XOR CORR at a>=0.2, ordering matches
- IME collapse at high alpha (PL 0.0029 vs LIME 0.6364)
- Saliency/DeepLift/GradientShap/GuidedBackprop magnitudes within ~0.02-0.03
- Filters fail badly in LIN WHITE (ours 0.115/0.120 vs PL 0.0152)




---

## 10. RIGID CORR results landed (2026-10-21 08:30 +06)

All 5 sweeps complete with the FIXED generator; accuracies match the regenerated
summary exactly (0.9930/0.9985/1.0000/0.9985/0.9990 for a=0.1..0.5). Results in
`results/patternlocal-repro-rigid-corr-sweep-{1..5}/`.

### RIGID CORR EMD (a=0.1 .. 0.5)

| method | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| PatternLocal | 0.1653 | 0.1414 | 0.1230 | 0.1034 | 0.1127 |
| LIME | 0.1433 | 0.1396 | 0.1374 | 0.1354 | 0.1333 |
| Analytical-IG | 0.1348 | 0.1018 | 0.1069 | 0.0885 | 0.1128 |
| Sobel | 0.1433 | 0.1288 | 0.1096 | 0.0917 | 0.0764 |
| **Laplace** | 0.1089 | 0.0739 | **0.0541** | **0.0425** | **0.0351** |
| Saliency | 0.1507 | 0.1283 | 0.1287 | 0.1332 | 0.1402 |

### Reading

* PatternLocal beats LIME at every alpha (0.1230 vs 0.1374 at a=0.3, widening to
  0.1127 vs 0.1333 at a=0.5); IME gap is much larger (0.4571 vs 0.6888 at a=0.5).
  -> Matches the paper's claim that PL substantially outperforms other XAI methods.
* **Laplace is the best method in RIGID CORR** (0.0351 at a=0.5) and improves
  monotonically with alpha. Sobel also improves.
  -> This matches the paper's "comparable to or worse than Sobel/Laplace" caveat
     in DIRECTION (filters are competitive/better here), i.e. RIGID CORR is the
     scenario where filters succeed, exactly as the paper claims (line 897,
     "sharp lines ... especially easy for filter-based methods").
* Note the same document-and-move-on finding applies: our Laplace (0.035) is
  lower than the paper's RIGID CORR values imply, consistent with the global
  EMD-floor difference documented in section 8 of the report.

### Batch 2 pushed (08:35 +06)

`xor-corr-sweep-3..6` + `rigid-white-sweep-5` (CPU, was the cancelled GPU job).

## 11. LOCKED datapoints recorded (08:45 +06)

Full EMD/IME tables for every valid config (13 methods x 9 configs) written to
**`results/LOCKED_DATAPOINTS.md`**. Valid set:

* RIGID_CORR a=0.1..0.5 (fixed generator, accs match regenerated summary)
* XOR_CORR a=0.1, 0.2, 0.9
* LIN_WHITE a=0.5

Everything else is missing/in-flight/invalid. Old rigid-corr-6 and rigid-white-5
results are excluded (old generator).
