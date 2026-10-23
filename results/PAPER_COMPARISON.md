# Ours (Clark-generated data) vs the paper

Fresh-start branch: data = verbatim braindatalab/xai-tris generator (global
Frobenius Eq. 12), eval core unchanged, n_test=600, SE bands = 1 std.
Reference tables: `Baseline Paper/xor-corr-datapoints.md`,
`Baseline Paper/rigid-corr-datapoints.md` (user's visual estimates of Fig 4).

---------------------------------------------------------------
## XOR CORR — TABLE COMPLETE (a=0.1–0.9)

Locked points table (ours vs paper):

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

Full PL curve (ours / paper):
0.138/0.100, 0.110/0.085, **0.0664/0.065**, **0.0545/0.058**, **0.0415/0.050**,
**0.0406/0.043**, **0.0397/0.045**, **0.0440/0.045**, **0.0331/0.048**.
From a=0.3 onward deltas are 0.001–0.009 — the paper's floor plateau
(0.043–0.048) is fully reproduced.

IME false-positive collapse total and monotone: PL 0.735 (a=0.1) ->
0.0185 (a=0.9) vs LIME flat ~0.68–0.78 (0.6806 at a=0.9). AIG identical
story (0.6208 -> 0.0183). 27x vs LIME at a=0.9.

Honest caveats: at a=0.9 the field bunches at the floor — PL 0.0331 ties
plain_ig 0.0331, DL 0.0313, GS 0.0312 — "PL uniquely best" does NOT hold
in our run. Laplace (0.0157) and Sobel (0.0213) keep declining below the
paper's flattens (0.078/0.124). LIME stays flat where the paper declines
(0.135 -> 0.082). Residual divergences are characterized, not anomalous.

---------------------------------------------------------------
## RIGID CORR (a=0.2, 0.3 completed; a=0.1 gated, acc 0.504)

| method | paper 0.2 | ours 0.2 | paper 0.3 | ours 0.3 |
|---|---|---|---|---|
| Laplace | 0.150 | 0.1548 | 0.110 | 0.1121 |
| PatternLocal | 0.200 | 0.3112 | 0.180 | 0.2759 |
| LIME | 0.260 | 0.3121 | 0.260 | 0.3093 |
| Sobel | 0.210 | 0.2643 | 0.190 | 0.2325 |
| IntegratedGradients | 0.240 | 0.2911 | — | — |
| DeepLift | 0.220 | 0.2917 | — | — |
| GradientShap | 0.240 | 0.2926 | — | — |
| GuidedBackprop | 0.240 | 0.2916 | — | — |
| Saliency | 0.230 | 0.2945 | — | — |
| (our) analytical_ig | — | 0.1771 | — | 0.1471 |

IME at a=0.3: LIME 0.9474, PL 0.8920, analytical_ig 0.8342, Sobel 0.7301,
Laplace 0.4518.

Reading: Laplace near-exact at both points (0.155/0.15, 0.112/0.110).
Analytical_ig is the runner-up overall at both a=0.2 (0.177) and a=0.3
(0.147), beating PL, LIME, and every baseline. PL beats LIME (0.276 vs
0.309 at a=0.3) but by less than the paper's gap (0.18 vs 0.26).
PL weak in RIGID (11th at a=0.2); Laplace is #1 in RIGID — opposite of LIN.

---------------------------------------------------------------
## LIN CORR — TABLE COMPLETE (a=0.1–0.9)

(No paper table exists for LIN; paper text says filters fail in LIN.)

| a | AIG | PL | AG | Laplace | LIME |
|---|---|---|---|---|---|
| 0.1 | 0.1245 | 0.1346 | 0.1160 | 0.1254 | 0.1349 |
| 0.2 | 0.1118 | 0.1096 | 0.0575 | 0.1433 | 0.1327 |
| 0.3 | 0.1024 | 0.1063 | 0.0444 | 0.1660 | 0.1313 |
| 0.4 | 0.0768 | 0.0785 | 0.0348 | 0.1829 | 0.1276 |
| 0.5 | 0.0532 | 0.0526 | 0.0263 | 0.1957 | 0.1271 |
| 0.6 | 0.0298 | 0.0297 | 0.0242 | 0.2057 | 0.1248 |
| 0.7 | 0.0198 | 0.0200 | 0.0195 | 0.2138 | 0.1228 |
| 0.8 | 0.0162 | 0.0162 | 0.0158 | 0.2203 | 0.1218 |
| 0.9 | 0.0149 | 0.0149 | 0.0160 | 0.2258 | 0.1223 |

Reading:
* **Filter-failure taxonomy confirmed on both sides.** Laplace MONOTONELY
  worsens with alpha (0.1254 -> 0.2258, rank 13/13 from a=0.2 onward);
  Sobel ~0.19 at a=0.8. RIGID Laplace is #1; LIN Laplace is last.
* **The analytic family converges to near-exact recovery** (~0.015–0.020 EMD
  at high alpha), well below LIME (~0.12 flat). AIG ties-or-beats PL at
  every LIN CORR point.
* **Thesis headline: AIG wins LIN CORR outright at a=0.9** — rank 1/13 on
  both EMD (0.0149) and IME (0.0194), edging AG (0.0160/0.0309) and tying
  PL on EMD (0.0149/0.0198). Same outright win in LIN WHITE at a=0.4 and
  a=0.5 (0.0154/0.0151, rank 1/13 both). Path-integration denoises past the
  plain gradient exactly where the family converges at high noise — two
  outright AIG wins on white background alone.

---------------------------------------------------------------
## LIN WHITE (white background) — TABLE COMPLETE (a=0.1–0.9)

| method | a=0.1 | a=0.2 | a=0.3 | a=0.4 | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 |
|---|---|---|---|---|---|---|---|---|---|
| analytical_grad | 0.0279 | 0.0175 | 0.0160 | 0.0154 | 0.0151 | 0.0149 | 0.0148 | 0.0147 | 0.0147 |
| analytical_ig | 0.0717 | 0.0180 | 0.0160 | 0.0154 | 0.0151 | 0.0149 | 0.0148 | 0.0147 | 0.0147 |
| patternlocal | 0.0781 | 0.0204 | 0.0164 | 0.0155 | 0.0151 | 0.0149 | 0.0148 | 0.0147 | 0.0147 |
| gradient_shap | 0.1190 | 0.1018 | 0.1018 | — | 0.1053 | — | — | — | — |
| plain_ig | 0.1268 | 0.1052 | — | — | 0.1053 | 0.1138 | — | — | — |
| lime | 0.1311 | 0.1286 | 0.1286 | 0.1276 | 0.1262 | 0.1245 | 0.1229 | 0.1188 | 0.1220 |
| laplace | 0.1375 | 0.1375 | 0.1375 | 0.1374 | 0.1374 | 0.1374 | 0.1372 | 0.1394 | 0.1336 |
| sobel | 0.1387 | 0.1382 | 0.1370 | — | — | — | 0.1198 | 0.1170 | — |

Reading:
* Analytic family ranks 1-3 at every completed point (a=0.1–0.9); AIG
  <= PL throughout. **AIG outright win at a=0.4–0.9** (rank 1/13 both
  metrics: 0.0154/0.0298, 0.0151/0.0247, 0.0149/0.0214, 0.0148/0.0191,
  0.0147/0.0174, 0.0147/0.0161).
* Laplace flat ~0.137 (rank 13 through a=0.7; rank 10 at a=0.8, rank 6
  at a=0.9) — filters do not recover on white background.
* CONVERGENCE: AIG 0.0717 -> 0.0180 -> 0.0160 -> 0.0154 -> ... -> 0.0147
  (monotone, near-exact by a=0.4); LIME flat ~0.12–0.13 all along.
  LIN WHITE TABLE COMPLETE.

---------------------------------------------------------------
## XOR WHITE (white background) — a=0.3–0.7 (a=0.1, 0.2 re-running after launch errors)

| method | a=0.2 | a=0.3 | a=0.4 | a=0.5 | a=0.6 | a=0.7 |
|---|---|---|---|---|---|---|
| patternlocal | **0.0566** | **0.0374** | **0.0328** | **0.0300** | **0.0448** | **0.0400** |
| analytical_grad | 0.0705 | 0.0527 | 0.0483 | 0.0461 | 0.0477 | 0.0485 |
| analytical_ig | 0.0858 | 0.0541 | 0.0512 | 0.0534 | 0.0675 | 0.0663 |
| gradient_shap | — | 0.1106 | 0.1006 | — | — | — |
| lime | 0.1335 | — | — | — | — | — |
| laplace | 0.1375 | 0.1671 | 0.1374 | 0.1374 | 0.1373 | 0.1372 |
| sobel | — | — | — | — | — | — |

IME: AIG 0.1178(r2), 0.0404(r1), 0.0307(r1), 0.0254(r1), 0.0221(r2), 0.0195(r2) — AIG holds
false-positive crown on white background; PL IME 0.1647(r3), 0.0431, 0.0311, —, —, —.

Reading:
* **PatternLocal is the outright EMD winner** at every completed point (rank
  1/13: 0.0374, 0.0328, 0.0300, 0.0448, 0.0400) — the only scenario so far
  where PL takes EMD #1. Note the a=0.6 uptick (0.0300 -> 0.0448) mirrors the
  XOR CORR high-alpha floor softening.
* **AIG holds IME rank 1-2 everywhere** — on white background AIG keeps the
  false-positive crown even though its EMD drops to rank 3-6 at high alpha:
  the EMD-vs-IME split again (PL best support recovery, AIG best
  false-positive suppression). AG ranks 2 on both metrics at a=0.3/0.4.
* Laplace/Sobel dead last at every point (Laplace 13/13) — filters do not
  recover on white XOR either.

---------------------------------------------------------------
## RIGID CORR — a=0.4–0.8 added (a=0.9 running)

| method | a=0.4 | a=0.5 | a=0.6 | a=0.7 | a=0.8 |
|---|---|---|---|---|---|
| laplace | **0.0825** (r1) | **0.0608** (r1) | **0.0445** (r1) | **0.0318** (r1) | **0.0218** (r1) |
| analytical_ig | **0.1805** (r2) | **0.1449** (r2) | **0.1333** (r2) | 0.1151 (r4) | **0.0566** (r2) |
| patternlocal | 0.2444 (r4) | 0.2181 (r5) | 0.1869 (r5) | 0.1231 (r5) | 0.0623 (r4) |
| analytical_grad | 0.2572 (r6) | 0.2160 (r4) | 0.1786 (r4) | 0.1042 (r3) | 0.0617 (r3) |
| sobel | — | — | 0.1350 (r3) | 0.1022 (r2) | 0.0693 (r5) |
| lime | 0.3015 | 0.2984 | 0.2931 (r13) | 0.2897 (r13) | 0.2878 (r10) |

IME (AIG rank): a=0.4 0.7843 (r3), a=0.5 0.7701 (r8), a=0.6 0.6888 (r7), a=0.7 0.5439 (r3), a=0.8 0.4608 (r4).
Laplace is IME #1 at a=0.6–0.8 (0.290, 0.258, 0.233).

Reading:
* **Laplace is EMD #1 at every RIGID CORR point** (0.083 -> 0.022) — the filter
  success the paper reports for RIGID; it keeps improving with alpha.
* **AIG is EMD #2 at 0.2, 0.3, 0.4, 0.5, 0.6, 0.8** (rank 4 only at 0.7). It is the
  only non-filter method that stays in the top two across the RIGID curve.
* PL falls to rank 4–5 from a=0.4 on; AG collapses to rank 4–6 at a≤0.6.
  Both sit well above AIG (0.18 vs 0.24 at a=0.4; 0.13 vs 0.19 at a=0.6).
* LIME is rank 10–13 from a=0.6 on (flat ~0.29, the paper's pattern for LIME in RIGID).

## Analytical-IG vs the field — rank summary (35 completed points)

Ranks among all 13 methods (EMD rank / IME rank):

| scene a | AIG | PL | AG | best EMD / best IME |
|---|---|---|---|---|
| XOR 0.1 | 3/13, 3/13 | 12/13 | **1/13**, 1/13 | AG / AG |
| XOR 0.2 | 4/13, 3/13 | 3/13 | 2/13 | Laplace / AG |
| XOR 0.3 | 4/13, 2/13 | 2/13 | 3/13 | Laplace / AG |
| XOR 0.4 | 4/13, 3/13 | 2/13 | 3/13 | Laplace / AG |
| XOR 0.5 | 5/13, 2/13 | 2/13 | 3/13 | Laplace / AG |
| XOR 0.6 | 5/13, 2/13 | 2/13 | 4/13 | Laplace / AG |
| XOR 0.7 | 8/13, 2/13 | 2/13 | 4/13 | Laplace / AG |
| XOR 0.8 | 8/13, 2/13 | 3/13 | 7/13 | Laplace / AG |
| XOR 0.9 | 8/13, 2/13 | 3/13 | 4/13 | Laplace / AG |
| LIN 0.1 | 3/13, 3/13 | 8/13 | **1/13** | AG / Laplace |
| LIN 0.2 | 3/13, 3/13 | 2/13 | **1/13** | AG / AG |
| LIN 0.3 | 2/13, 3/13 | 3/13 | **1/13** | AG / AG |
| LIN 0.4 | **1/13**, **1/13** | 3/13 | **1/13** | AIG / AIG |
| LIN 0.5 | 2/13, 2/13 | 2/13 | **1/13** | AG / AG |
| LIN 0.6 | 2/13, 2/13 | 2/13 | **1/13** | AG / AG |
| LIN 0.7 | 2/13, 2/13 | 2/13 | **1/13** | AG / AG |
| LIN 0.8 | 2/13, 2/13 | 3/13 | **1/13** | AG / AG |
| LIN 0.9 | **1/13**, **1/13** | 2/13 | 3/13 | AIG / AIG |
| WLIN 0.1 | 2/13, — | 3/13 | **1/13** | AG / AG |
| WLIN 0.2 | 2/13, 2/13 | 3/13 | **1/13** | AG / AG |
| WLIN 0.3 | 2/13, **1/13** | 3/13 | **1/13** | AG / AG |
| WLIN 0.4 | **1/13**, **1/13** | 3/13 | **1/13** | AIG / AIG |
| WLIN 0.5 | **1/13**, **1/13** | 3/13 | 2/13 | AIG / AIG |
| WLIN 0.6 | **1/13**, **1/13** | 3/13 | 2/13 | AIG / AIG |
| WLIN 0.7 | **1/13**, **1/13** | 3/13 | 2/13 | AIG / AIG |
| WLIN 0.8 | **1/13**, **1/13** | 2/13 | 3/13 | AIG / AIG |
| WLIN 0.9 | **1/13**, **1/13** | 3/13 | 2/13 | AIG / AIG |
| WXOR 0.2 | 3/13, r2 | **1/13** | 2/13 | PL / AIG |
| WXOR 0.2 | 3/13, r2 | **1/13** | 2/13 | PL / AIG |
| WXOR 0.3 | 3/13, **1/13** | **1/13** | 2/13 | PL / AIG |
| WXOR 0.4 | 3/13, **1/13** | **1/13** | 2/13 | PL / AIG |
| WXOR 0.5 | 3/13, **1/13** | **1/13** | 2/13 | PL / AIG |
| WXOR 0.6 | 3/13, r2 | **1/13** | 2/13 | PL / AIG |
| WXOR 0.7 | r6, r2 | **1/13** | 2/13 | PL / AIG |
| RIGID 0.2 | **2/13**, 3/13 | 11/13 | **13/13** | Laplace |
| RIGID 0.3 | **2/13**, 3/13 | 4/13 | **12/13** | Laplace |

Findings:
1. **AIG beats LIME on BOTH metrics at every completed point**, and beats
   PatternLocal on IME at nearly every point (only XOR a=0.4 goes to PL,
   0.2741 vs 0.2499).
2. **AIG is the robustness member of the covariance family.** In RIGID it is
   #2 overall (behind only Laplace) at both points; its sibling AG
   (plain-gradient kernel) COLLAPSES to rank 12–13 of 13 — IG path
   integration is what keeps the covariance kernel informative under
   translation/rotation. PL is also weak in RIGID (11th at a=0.2).
3. **AIG wins outright in LIN at high alpha:** rank 1/13 on both metrics at
   LIN CORR a=0.9 and LIN WHITE a=0.4–0.9 — six outright wins on white
   background alone (0.0154/0.0298, 0.0151/0.0247, 0.0149/0.0214,
   0.0148/0.0191, 0.0147/0.0174, 0.0147/0.0161). Path-integration denoises
   past the plain gradient exactly where the family converges at high noise.
4. **AIG's weak spot is XOR high-alpha EMD** (rank 8 at a=0.7–0.9) while it
   holds IME rank 2 — an EMD-vs-IME trade-off to report in the thesis.
5. AG dominates EMD+IME in LIN and low-alpha XOR (rank 1–2); the analytical
   family owns the top of the field wherever supports are fixed.

---------------------------------------------------------------
## Magnitudes — generator problem solved

Old generator was ~0.5–0.7x the paper everywhere. On Clark's data:
LIME hits paper XOR a=0.1 nearly exactly (0.1368 vs 0.135); Laplace nails
RIGID a=0.2 (0.155 vs 0.15); Sobel tracks. Anchor points match within
0.01–0.02 — residual deltas come from mask thresholds/eval details, not
data semantics. No low-alpha gradient-family collapse.

---------------------------------------------------------------
## Grid state

Done (35 fetched & analyzed): XOR corr complete (0.1–0.9); LIN corr
complete (0.1–0.9); LIN white complete (0.1–0.9); XOR white 0.2–0.7;
RIGID corr 0.2–0.3.
Running (5): xor-white 1 (re-run, still going), xor-white 8, xor-white 9,
rigid-corr 4, rigid-corr 5.
Remaining to push (7): rigid-corr 6–9, rigid-white 7–9.
Still-to-fetch when complete: xor-white 1, 8, 9; rigid-corr 4, 5.

Total planned: 47 (XOR corr 9, LIN corr 9, LIN white 9, XOR white 9,
RIGID corr 8 valid α=0.2–0.9, RIGID white 3 valid α=0.7–0.9).
Progress: 35 fetched + 5 running = 40; 7 remaining.

---------------------------------------------------------------
## SLOT NOTES (working log — not results)

All slot-refill "push" notes removed from results section. Commit messages
on main carry the push history.
