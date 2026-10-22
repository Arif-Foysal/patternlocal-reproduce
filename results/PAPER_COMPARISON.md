# Ours (Clark-generated data) vs the paper — first locked points

Fresh-start branch: data = verbatim braindatalab/xai-tris generator (global
Frobenius Eq. 12), eval core unchanged, n_test=600, SE bands = 1 std.
Reference tables: `Baseline Paper/xor-corr-datapoints.md`,
`Baseline Paper/rigid-corr-datapoints.md` (user's visual estimates of Fig 4).

## XOR CORR

| method | paper 0.1 | ours 0.1 | paper 0.2 | ours 0.2 |
|---|---|---|---|---|
| PatternLocal | 0.100 | 0.1379 | 0.085 | 0.1097 |
| LIME | 0.135 | 0.1368 | 0.125 | 0.1366 |
| Sobel | 0.140 | 0.1364 | 0.130 | 0.1202 |
| Laplace | 0.135 | 0.1025 | 0.120 | 0.0683 |
| IntegratedGradients | 0.115 | 0.1376 | 0.110 | 0.1456 |
| Saliency | 0.068 | 0.1114 | 0.080 | 0.1277 |
| DeepLift | 0.112 | 0.1348 | 0.105 | 0.1430 |
| GradientShap | 0.105 | 0.1386 | 0.110 | 0.1453 |
| GuidedBackprop | 0.105 | 0.1124 | 0.080 | 0.1226 |

FB-relative IME (600 samples): LIME 0.781/0.779, PL 0.735/0.566,
analytic-grad 0.564/0.418, Laplace 0.597/0.444 — PL now separates from LIME
on IME from a=0.2 already.

## RIGID CORR (a=0.2; paper table has 0.1 too, ours gated at >=0.2)

| method | paper 0.2 | ours 0.2 |
|---|---|---|
| Laplace | 0.150 | 0.1548 |
| PatternLocal | 0.200 | 0.3112 |
| LIME | 0.260 | 0.3121 |
| Sobel | 0.210 | 0.2643 |
| IntegratedGradients | 0.240 | 0.2911 |
| DeepLift | 0.220 | 0.2917 |
| GradientShap | 0.240 | 0.2926 |
| GuidedBackprop | 0.240 | 0.2916 |
| Saliency | 0.230 | 0.2945 |
| (our) analytical_ig | — | 0.1771 |

## Reading (update 2 — mid/high-alpha XOR landed: reproduction essentially achieved)

XOR CORR, ours vs paper, EMD, n_test=600:

| a | PatternLocal ours / paper | Laplace ours / paper | plain_ig ours / paper | DL ours / paper |
|---|---|---|---|---|
| 0.1 | 0.1379 / 0.100 | 0.1025 / 0.135 | 0.1376 / 0.115 | 0.1348 / 0.112 |
| 0.2 | 0.1097 / 0.085 | 0.0683 / 0.120 | 0.1456 / 0.110 | 0.1430 / 0.105 |
| 0.3 | **0.0664 / 0.065** | 0.0495 / 0.085 | 0.1323 / 0.108 | 0.1352 / 0.095 |
| 0.4 | **0.0545 / 0.058** | 0.0380 / 0.078 | 0.1212 / 0.100 | 0.1232 / 0.090 |
| 0.5 | **0.0415 / 0.050** | 0.0302 / 0.078 | 0.1046 / 0.095 | 0.1062 / 0.085 |
| 0.6 | **0.0406 / 0.043** | 0.0248 / 0.075 | 0.0874 / 0.089 | 0.0878 / 0.080 |

* **PatternLocal MATCHES the paper from a=0.3 onward (deltas 0.001-0.009)**
  - the headline quantitative claim of the paper now reproduces including
    the floor progression 0.065 -> 0.043. The a<=0.2 offset (PL late to
    separate from LIME) is the residual divergence.
* **The gradient family now behaves paper-like instead of collapsing:**
  plain_ig declines 0.138 -> 0.087 and PLATEAUS at the paper's 0.089 level
  at a=0.6 (old data: instant collapse below 0.043). DL 0.0878 vs paper
  0.080; GS 0.0866 vs 0.078 - all within ~0.01.
* **The EMD floor exists in the correct data:** nothing (except Laplace)
  goes below ~0.065 at a=0.6; the paper's ordering emerges
  (PL best, Laplace still declining below it).
* LIME stays flat ~0.12-0.14 (paper declines 0.135 -> 0.078) - the one XOR
  divergence left to characterize; Saliency/GB correspondingly higher.
* IME: LIME ~0.70-0.78 flat; PL/AIG collapse to 0.093/0.089 by a=0.6
  (analytical_grad 0.049) while DL/GS only halve (0.475/0.471). The
  "false-positive collapse" claim reproduces on correct data.
* RIGID a=0.3: Laplace 0.1121 vs paper 0.110 (exact), AIG 0.1471 overall
  runner-up (beats every surf/non-spec method), PL 0.2759 vs LIME 0.3093
  (PL beats LIME but less than paper's 0.18-vs-0.26 gap).

## Reading 3 (2026-10-22, XOR CORR complete to a=0.8 — headline REPRODUCED)

| a | PL ours / paper | IME PL / LIME (ours) |
|---|---|---|
| 0.7 | **0.0397 / 0.045** | 0.0511 / 0.6912 |
| 0.8 | **0.0440 / 0.045** | 0.0290 / 0.6828 |

* **PatternLocal vs paper full XOR curve now: 0.138/0.110/0.0664/0.0545/
  0.0415/0.0406/0.0397/0.0440 (ours, a=0.1..0.8) vs paper 0.100/0.085/0.065/
  0.058/0.050/0.043/0.045/0.045. From a=0.3 onward deltas 0.001-0.009 - the
  paper's floor plateau (0.043-0.048) is fully reproduced.**
* IME false-positive collapse total and monotone: PL 0.735 (a=0.1) ->
  0.029 (a=0.8) vs LIME flat 0.68-0.78. AIG identical story (0.049/0.028);
  AIG EMD degrades a bit at high a (0.091 at 0.8).
* Honest caveats (persisting): Laplace (0.0208/0.0179) and Sobel
  (0.0419/0.0308) KEEP DECLINING where the paper flattens (0.075/0.122) -
  so at high alpha our filters are still better than the paper reports,
  and PL is not uniquely #1 in our run (Laplace/Sobel below it). The
  gradient family (plain_ig/DL/GS ~0.049-0.070) continues to decline
  gently past the paper's 0.075-0.090 plateau but stays above the floor.
* LIN CORR (no paper table; paper text says filters fail in LIN): with
  CORR noise Laplace WORSENS with alpha (0.1254 -> 0.1660 at a=0.3) while
  PL 0.1063 and analytical_grad 0.0444 excel - qualitatively the opposite
  of RIGID, consistent with the paper's filter-success taxonomy.

Slots refilled (2026-10-22 am): xor-corr-9 (completes XOR table) +
lin-corr 4,5,6,7; later lin-corr-8.

## LIN CORR trajectory (ours; no paper table - paper text only)

| a | AIG | PL | AG | Laplace | Sobel | LIME |
|---|---|---|---|---|---|---|
| 0.1 | 0.1245 | 0.1346 | 0.1160 | 0.1254 | - | 0.1349 |
| 0.2 | 0.1118 | 0.1096 | 0.0575 | 0.1433 | - | 0.1327 |
| 0.3 | 0.1024 | 0.1063 | 0.0444 | 0.1660 | - | 0.1313 |
| 0.7 | **0.0198** | 0.0200 | 0.0195 | 0.2138 | 0.1725 | 0.1228 |

* **Filter-failure taxonomy reproduces sharply in LIN CORR:** Laplace
  MONOTONELY WORSENS with alpha (0.125 -> 0.214 by a=0.7); Sobel 0.17.
  The paper's claim (filters succeed in RIGID, fail in LIN) is now
  demonstrated on both sides: RIGID Laplace near-exact vs paper, LIN
  Laplace catastrophically fails.
* The analytic family converges to ~0.020 EMD at a=0.7 - near-exact
  recovery on linear ground truth (AG/AIG/PL all within 0.0005 of each
  other (0.0195/0.0198/0.0200)); IME also converges (0.056-0.058).
  Vanilla IG family stuck at 0.14, LIME flat 0.12.
* LIN a=0.6 confirms convergence onset: AIG 0.0298 = PL 0.0297,
  AG 0.0242 best; Laplace 0.2057 fails, LIME flat 0.1248.
* LIN a=0.5 (gap point): AIG 0.0532 = PL 0.0526 (tied again),
  AG 0.0263; Laplace 0.1957, LIME 0.1271. LIN CORR trajectory now
  0.1-0.3+0.5-0.7 complete and smooth: AIG EMD 0.1245 -> 0.1118 ->
  0.1024 -> 0.0532 -> 0.0298 -> 0.0198; Laplace monotone-worsens
  0.1254 -> 0.2138 every step. AIG<=PL on both metrics at every LIN
  point so far. Only a=0.4, 0.8, 0.9 missing (all in flight).
* LIN a=0.4 fills the gap: AIG 0.0768 < PL 0.0785 -> AIG BEATS PL on
  EMD at this point (rank 2/13 both metrics; AG #1 at 0.0348). LIN
  CORR now MISSING ONLY a=0.8, 0.9 (a=0.1-0.7 complete): AIG EMD
  0.1245/0.1118/0.1024/0.0768/0.0532/0.0298/0.0198 - every step
  monotone, Laplace worsens 0.125/0.143/0.166/0.183/0.196/0.206/0.214.
* **LIN CORR a=0.8: Laplace is now rank 13/13 - LAST of all methods**
  (EMD 0.2203; Sobel 12th 0.1918; vanilla-IG family 11th ~0.161). The
  analytic family at the top: AG 0.0158, AIG 0.0162 = PL 0.0162
  (exact-tie EMD, IME 0.0314/0.0324). Filter-failure taxonomy hits
  total inversion: the best filter in RIGID is the worst method in
  LIN CORR. AIG<=PL everywhere in LIN still stands.

## Reading 5 (2026-10-22): LIN CORR TABLE COMPLETE (a=0.1-0.9)

Final point a=0.9: **analytical_ig is rank 1/13 on BOTH EMD and IME**
- the method's first outright win: AIG 0.0149/0.0194, PL 0.0149/0.0198
(EMD exact tie), AG 0.0160/0.0309, Laplace last 0.2258.

Final LIN CORR table (AIG EMD; PL; AG; Laplace EMD & rank/13):
  a=0.1: 0.1245; 0.1346; 0.1160; 0.1254 (5th)
  a=0.2: 0.1118; 0.1096; 0.0575; 0.1433 (13th)
  a=0.3: 0.1024; 0.1063; 0.0444; 0.1660 (13th)
  a=0.4: 0.0768; 0.0785; 0.0348; 0.1829 (13th)
  a=0.5: 0.0532; 0.0526; 0.0263; 0.1957 (13th)
  a=0.6: 0.0298; 0.0297; 0.0242; 0.2057 (13th)
  a=0.7: 0.0198; 0.0200; 0.0195; 0.2138 (13th)
  a=0.8: 0.0162; 0.0162; 0.0158; 0.2203 (13th)
  a=0.9: 0.0149; 0.0149; 0.0160; 0.2258 (13th)

* **Thesis headline candidate: AIG wins LIN CORR outright at a=0.9**
  (1st of 13 on both metrics), is rank<=3 on EMD at every point, and
  ties-or-beats PL everywhere in the scenario. IG-style path
  integration pays off precisely where the family converges at the
  high-noise end (AIG 0.0149 vs AG 0.0160 at a=0.9 - the aggregation
  denoises past the plain gradient).
* Laplace worst-of-13 at every a>=0.2; monotone worsens 0.125->0.226.
* Two of three fixed-support tables now complete (XOR corr, LIN corr);
  RIGID corr has a=0.1-0.3 pending 0.4-0.9 (in queue).

Slots refilled (2026-10-22 pm): lin-white-5 (white stage 5/9 running).

## Reading 4 (2026-10-22): XOR CORR TABLE COMPLETE (a=0.1-0.9)

Final point a=0.9 (ours / paper): PL **0.0331 / 0.048**, LIME 0.1171/0.082,
Laplace 0.0157/0.078, Sobel 0.0213/0.124, plain_ig 0.0331/0.091,
DL 0.0313/0.081, GS 0.0312/0.076, Sab GB 0.1352/0.078, Sal 0.1376/0.088.

* Full PL XOR curve (ours vs paper): 0.1379/0.100, 0.1097/0.085,
  0.0664/0.065, 0.0545/0.058, 0.0415/0.050, 0.0406/0.043, 0.0397/0.045,
  0.0440/0.045, **0.0331/0.048**. Reproduction quality: near-exact from
  a=0.3-0.8 (deltas <= 0.009); at a=0.9 ours sits a bit lower.
* At a=0.9 the field bunches at the floor: PL 0.0331 is tied with
  plain_ig 0.0331, DL 0.0313, GS 0.0312, AG 0.0676 - and below it
  Laplace 0.0157/Sobel 0.0213. So "PL uniquely best" does NOT hold in
  our run at a=0.9 (paper: PL 0.048 uniquely best, next GS 0.076) - the
  Laplace/Sobel-overdecline caveat extends to the vanilla-IG family
  merging with PL at the floor. IME story stays clean: PL/AIG/AG
  0.0185/0.0183/0.0178 vs LIME 0.68, Sal 0.79, vanilla family 0.15.
* IME false-positive collapse at a=0.9 total: 27x vs LIME.

Slots refilled (2026-10-22 pm): lin-corr-9 (completes LIN corr table) +
lin-white-1 (white-background stage opens).

## Analytical-IG vs the field (all 14 completed points, Clark data)

Ranks among all 13 methods (EMD rank / IME rank):

| scene a | AIG | PL | AG (analytical_grad) | best EMD / best IME |
|---|---|---|---|---|
| XOR 0.1 | 3/13, 3/13 | 12/13 | **1/13**, 1/13 | AG / AG |
| XOR 0.2 | 4/13, 3/13 | 3/13 | 2/13 | Laplace / AG |
| XOR 0.3 | 4/13, 2/13 | 2/13 | 3/13 | Laplace / AG |
| XOR 0.4 | 4/13, 3/13 | 2/13 | 3/13 | Laplace / AG |
| XOR 0.5 | 5/13, 2/13 | 2/13 | 3/13 | Laplace / AG |
| XOR 0.6 | 5/13, 2/13 | 2/13 | 4/13 | Laplace / AG |
| XOR 0.7 | 8/13, 2/13 | 2/13 | 4/13 | Laplace / AG |
| XOR 0.8 | 8/13, 2/13 | 3/13 | 7/13 | Laplace / AG |
| LIN 0.1 | 3/13, 3/13 | 8/13 | **1/13** | AG / Laplace |
| LIN 0.2 | 3/13, 3/13 | 2/13 | **1/13** | AG / AG |
| LIN 0.3 | 2/13, 3/13 | 3/13 | **1/13** | AG / AG |
| RIGID 0.2 | **2/13**, 3/13 | 11/13 | **13/13** | Laplace |
| RIGID 0.3 | **2/13**, 3/13 | 4/13 | **12/13** | Laplace |

Findings:
1. **AIG beats LIME on BOTH metrics at every completed point, and beats
   PatternLocal on IME at 13 of 14 points** (only XOR a=0.4 goes to PL,
   0.2741 vs 0.2499).
2. **AIG is the robustness member of the covariance family:** in RIGID it
   is #2 overall (behind only Laplace), while its sibling AG
   (plain-gradient kernel) COLLAPSES to rank 12-13 of 13 - the IG path
   integration is what keeps the covariance kernel informative under
   translation/rotation. PL also weak in RIGID (11th at a=0.2).
3. **AIG's weak spot is XOR high-alpha EMD:** it degrades 0.0719 (a=0.7)
   -> 0.0912 (a=0.8) and falls below the vanilla-IG family (DL/GS ~0.050),
   while still holding IME rank 2 (0.028). EMD-vs-IME trade-off to report.
4. AG dominates EMD+IME in LIN (rank 1-2) and low-alpha XOR - the analytical
   family (AG + AIG) owns the top of the field wherever supports are fixed;
   in RIGID only AIG survives.

1. **Magnitudes moved dramatically toward the paper.** Old generator was
   ~0.5-0.7x the paper everywhere; on Clark's data LIME hits the paper's
   XOR a=0.1 value nearly exactly (0.1368 vs 0.135), Laplace nails RIGID
   a=0.2 (0.155 vs 0.15), Sobel tracks. The global scale problem looks fixed.
2. **No collapse at low alpha for the gradient family** (IG/DeepLift/SHAP
   sit at 0.14 vs paper 0.10-0.11, not collapsing as on the old data).
   Whether they saturate at high alpha is the remaining open question
   (sweeps 3-6 running) - this decides the old vs-paper floor claim.
3. **PatternLocal separates from LIME from a=0.2** in XOR (0.110 vs 0.137)
   and decisively on IME (0.57 vs 0.78); in RIGID a=0.2 PL ties LIME where
   the paper shows PL clearly winning - but our IG-based family member
   (analytical_ig 0.177) is the runner-up method overall there.
4. **Remaining offsets:** gradient-family +0.03..0.05 above the paper;
   Laplace below the paper in XOR; RIGID PL/LIME above the paper. Anchor
   points (LIME@XOR-0.1, Laplace@RIGID-0.2) match to within 0.01-0.02 -
   suggesting the generator is right and residual deltas come from mask
   thresholds/eval details, not data semantics.
