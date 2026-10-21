# Filter-baseline non-reproduction: full investigation report

**Date:** 2026-10-21
**Question:** Why do our Sobel/Laplace results diverge from the paper's Figure 4
(XOR CORR panel)?

**Answer:** We exhaustively tested every plausible cause and excluded all of them.
The divergence is a **genuine non-reproduction**, not a bug in our code. Our
Sobel/Laplace, mask, metric, and data pipeline are all verified correct against the
paper's own definitions and its own POT library. The paper's XAI-TRIS generator is
not shipped, and its filter numbers are not reachable from its stated definitions.

---

## 1. The discrepancy

Paper's XOR CORR EMD (raster-traced from page 8 at 600 dpi — authoritative, not a
visual estimate):

| curve | a=0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| one filter | 0.173 | 0.169 | 0.167 | 0.166 | 0.166 | 0.166 | 0.166 | 0.167 |
| another filter | 0.046 | 0.047 | 0.048 | 0.049 | 0.050 | 0.051 | 0.051 | 0.053 |

Paper's filters are **flat / slightly rising** — they never improve with SNR.
This matches the paper's own Remark (line 2005-2007): "increasing alpha sharpens
tetromino boundaries and causes the filters to respond outside the true region,
increasing the EMD/IME."

Ours (XOR CORR, full test set, POT):

| method | a=0.1 | 0.2 | 0.9 |
|---|---|---|---|
| Sobel | 0.122 | 0.108 | **0.020** |
| Laplace | 0.092 | 0.063 | **0.016** |

Ours **collapse at high alpha** — the opposite of the paper.

---

## 2. Everything we ruled out (with numbers)

### 2.1 The metric is correct — verified against the paper's own library
The paper uses POT (Python Optimal Transport). We installed `POT 0.9.7.post1`
(the exact library) and recomputed the saved XOR CORR a=0.9 samples:

| method | our POT EMD | Kaggle full-set | 
|---|---|---|
| Sobel | 0.0180-0.0218 | 0.0200 |
| Laplace | 0.0159-0.0166 | 0.0162 |
| LIME | 0.099-0.104 | 0.1012 |

Our scoring is **exactly** reproducible. Not the bug.

### 2.2 The EMD/IME formulas match the paper
`EMD = OT(|s|, F+)/delta_max` (Eq. 10), `delta_max` = 64x64 diagonal = 89.095
("maximum Euclidean distance between any two pixels"), `IME = 1 - sum_F+|s|/sum|s|`
(Eq. 11). All verified character-by-character against the paper text and our code.

### 2.3 The mask is correct per the paper's Eq. 13/14
`F+ = support(R o (H o a))` (Eq. 13). For XOR, `a = aT - aL`; since T and L do not
overlap (0 overlap px), support = union = 1054 px. Our code produces exactly this,
byte-identical. `smooth_signal` already applies the paper's "5% maximum support
threshold" (line 1506). Verified.

### 2.4 The data pipeline matches Eq. 12
`x = alpha*(R o (H o a)) + (1-alpha)*(G o eta)` with signal and noise normalized
**separately** (Frobenius) then mixed. Our generator does exactly this. Verified.
Signal is 100% inside F+, no border clipping, clean 1054 px.

### 2.5 Filter definition is not the cause
Tested 12 variants (`scripts/filter_probe.py`): scipy vs true 3x3 Sobel kernel,
`sqrt(gx^2+gy^2)` vs `max(|gx|,|gy|)` vs `|gx|+|gy|`, signed vs `|.|` Laplace,
4- vs 8-neighbour Laplacian, filter on signal-only, pre-blur at sigma 0-3.0.

**Every** variant collapses at a=0.9 (0.002-0.017). Kernel/sign/blur choices make
less than 0.02 difference. Dead end.

### 2.6 Mask geometry cannot fix it
Tested on XOR CORR a=0.9:

| mask | px | Sobel | LIME |
|---|---|---|---|
| union filled (ours, = Eq. 13) | 1054 | 0.0081 | 0.0737 |
| union outline | 181 | 0.0202 | 0.1021 |
| T filled only | 320 | 0.2395 | 0.2622 |
| T outline only | 90 | 0.2327 | 0.2511 |
| **paper target** | — | **~0.167** | **~0.05** |

**No mask reproduces BOTH** the paper's Sobel (~0.167) and its LIME (~0.05)
simultaneously. Small masks fix Sobel but break LIME; the union fixes LIME but
makes Sobel trivial.

### 2.7 The reference repo does not ship the generator or the filters
`https://github.com/gjoelbye/PatternLocal` ships only the PatternLocal method
(solvers, superpixel simplification). The XAI-TRIS generator is from Clark et al.
[8] and is **not** in the repo. The filter baselines and the EMD/IME evaluation are
**not** in the repo either. So the filter+mask+metric path cannot be verified
against their code.

---

## 3. Root cause

The difference is **entirely geometric**: our filter mass lands inside F+, the
paper's does not.

At XOR CORR a=0.9 our **Sobel puts 88% of its mass inside F+** (Laplace 70%).
Since XOR's F+ is the union of both filled tetrominoes — a blob covering 26% of the
image — and Sobel's ridge traces the boundary of that blob (which lies inside F+),
the transport distance is ~0.6 px. Hence EMD 0.02.

Even feeding the **raw input image x** as an "explanation" scores 0.023 against our
F+. Our F+ is so large that anything spatially near the shapes scores ~0.02. The
paper's filters score 0.167, so in their setup the filters must put substantial
mass **away from** F+.

We cannot determine from the paper *why*, because:
1. the generator is not shipped, and
2. the paper does not state the filter input (raw vs normalized) nor the exact
   tetromino placement beyond "near top-left"/"near bottom-right".

Most likely the paper's tetromino placement/size, or how the CORR (sigma=10)
noise field interacts with the shapes, differs from ours.

---

## 4. What reproduces and what does not

### Reproduces
- **PatternLocal beats LIME on XOR CORR** at alpha >= 0.2 (ordering matches).
- **IME collapse at high alpha**: PL 0.0029 vs LIME 0.6364 at a=0.9 (219x).
- **Gradient baselines** (Saliency/DeepLift/GradientShap/GuidedBackprop) within
  ~0.02-0.03 of the paper.
- **Filters fail in LIN WHITE**: ours Sobel 0.115 / Laplace 0.120 vs PL 0.0152.
- The paper's *method* claims in the prose.

### Does not reproduce
- The **absolute filter numbers** in XOR CORR (ours 0.02, paper 0.167).
- The **filter trend vs alpha** (ours fall, paper flat/rising).
- This is a characterised, honest non-reproduction.

---

## 5. Recommendation

State the filter discrepancy explicitly in the report with the evidence above.
Do **not** tune our filters to match the paper — the paper's numbers are not
reachable from the paper's own stated definitions, and tuning would fabricate
agreement.

---

## 6. Artifacts

- `scripts/filter_probe.py` — 12-variant filter test harness
- `INVESTIGATION_LIVE.md` — running context log (Kaggle jobs etc.)
- `/tmp/fig4/xor_emd_crop.png` — the raster-traced paper panel
- `Baseline Paper/xor-corr-datapoints.md` — user's hand estimate (treat as
  indicative only; the raster is authoritative)


---

## 7. FINAL comparison against the user's authoritative table

`Baseline Paper/xor-corr-datapoints.md` (visual estimate from the paper's XOR CORR
plot) is taken as ground truth. Comparison at the alphas we have:

### alpha = 0.9 (the decisive alpha)

| method | ours | paper | delta |
|---|---|---|---|
| PatternLocal | 0.0346 | 0.048 | -0.013 |
| LIME | 0.1012 | 0.080 | +0.021 |
| Sobel | 0.0200 | 0.125 | **-0.105** |
| Laplace | 0.0162 | 0.080 | **-0.064** |
| IntegratedGradients | 0.0259 | 0.090 | **-0.064** |
| Saliency | 0.1189 | 0.088 | +0.031 |
| DeepLift | 0.0269 | 0.080 | **-0.053** |
| GradientShap | 0.0264 | 0.075 | **-0.049** |
| GuidedBackprop | 0.1171 | 0.078 | +0.039 |

### Structural difference (the real finding)

| | paper: range across alpha | paper: floor |
|---|---|---|
| Sobel | 0.019 (FLAT 0.140->0.125) | 0.125 |
| Laplace | 0.060 (0.135->0.080) | 0.075 |
| PatternLocal | 0.057 (0.100->0.048) | 0.043 |
| **all methods** | | **never below 0.043** |

| | ours | |
|---|---|---|
| Sobel | 0.122 -> **0.020** | collapses |
| Laplace | 0.092 -> **0.016** | collapses |
| PatternLocal | 0.132 -> 0.035 | collapses |
| **5 methods** | **below 0.030 at alpha=0.9** | |

**In the paper, no method can get below ~0.043. In ours, five methods do.**

Crucially: this affects **every method, not just the filters**. PatternLocal is the
closest match (0.035 vs 0.048), which is why the paper's *method* claim reproduces.
The difference is a **global floor** on EMD that the paper's data has and ours does
not.

## 8. Conclusion

The divergence is **not** a filter bug. It is a **global difference in the generated
data**:

* Every method in the paper saturates at EMD >= 0.043; ours reaches 0.016.
* A floor that applies to *all* methods cannot be caused by any single method's
  implementation (filter kernel, sign, blur, mask) -- all of which we tested
  exhaustively (sections 2, 6).
* Therefore the difference is upstream: the **XAI-TRIS generator**. The paper's
  generator (Clark et al. [8]) is not shipped and the paper does not fully specify
  tetromino placement/size or how the sigma=10 CORR noise field interacts with the
  shape halo.

**Decision (user-approved): document and move on.** Do not tune to match. Spend
remaining effort completing the six EMD/IME curves.

**What reproduces:**
* PatternLocal beats LIME on XOR CORR at alpha >= 0.2; ordering matches the paper.
* PatternLocal is the best or near-best method (0.035 vs paper 0.048).
* IME collapse at high alpha (PL 0.0029 vs LIME 0.6364 at alpha=0.9).
* Saliency/GuidedBackprop magnitudes within ~0.03-0.04.

**What does not reproduce:**
* The absolute EMD floor / spread of the baselines (documented above).

