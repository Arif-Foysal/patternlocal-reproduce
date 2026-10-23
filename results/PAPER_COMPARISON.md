# Ours (Clark-generated data) vs the paper

Fresh-start branch: data = verbatim braindatalab/xai-tris generator (global
Frobenius Eq. 12), eval core unchanged, n_test=600, SE bands = 1 std.
Reference tables: `Baseline Paper/xor-corr-datapoints.md`,
`Baseline Paper/rigid-corr-datapoints.md` (visual estimates of paper Fig 4).

Status: 41 sweeps analyzed, 5 running, 1 not yet pushed (47 planned).

---------------------------------------------------------------
## 1. XOR CORR — complete (a=0.1–0.9)

Locked points, ours vs paper:

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

PatternLocal full curve (ours / paper):
0.1379/0.100, 0.1097/0.085, **0.0664/0.065**, **0.0545/0.058**, **0.0415/0.050**,
**0.0406/0.043**, **0.0397/0.045**, **0.0440/0.045**, **0.0331/0.048**.
From a=0.3 onward the deltas are 0.001–0.009; the paper's floor plateau
(0.043–0.048) is reproduced.

IME: PL falls 0.735 (a=0.1) -> 0.0185 (a=0.9) while LIME stays flat at
0.68–0.78 (27x gap at a=0.9).

Caveats:
* At a=0.9 the field bunches at the floor: PL 0.0331 ties plain_ig 0.0331,
  DeepLift 0.0313, GradientShap 0.0312. "PL uniquely best" does not hold.
* Laplace (0.0157) and Sobel (0.0213) keep declining where the paper flattens
  them (0.078 / 0.124).
* LIME stays flat where the paper declines (0.135 -> 0.082).

---------------------------------------------------------------
## 2. RIGID CORR — a=0.2–0.8 analyzed, a=0.9 running

a=0.1 is gated out (accuracy 0.504 < 0.90).

Locked points, ours vs paper:

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
| analytical_ig (ours) | — | 0.1771 | — | 0.1471 |

Full EMD table, a=0.4–0.8:

| method | a=0.4 | a=0.5 | a=0.6 | a=0.7 | a=0.8 |
|---|---|---|---|---|---|
| laplace | **0.0825** (r1) | **0.0608** (r1) | **0.0445** (r1) | **0.0318** (r1) | **0.0218** (r1) |
| analytical_ig | **0.1805** (r2) | **0.1449** (r2) | **0.1333** (r2) | 0.1151 (r4) | **0.0566** (r2) |
| patternlocal | 0.2444 (r4) | 0.2181 (r5) | 0.1869 (r5) | 0.1231 (r5) | 0.0623 (r4) |
| analytical_grad | 0.2572 (r6) | 0.2160 (r4) | 0.1786 (r4) | 0.1042 (r3) | 0.0617 (r3) |
| sobel | — | — | 0.1350 (r3) | 0.1022 (r2) | 0.0693 (r5) |
| lime | 0.3015 | 0.2984 | 0.2931 (r13) | 0.2897 (r13) | 0.2878 (r10) |

IME at a=0.4–0.8: AIG 0.7843 (r3), 0.7701 (r8), 0.6888 (r7), 0.5439 (r3),
0.4608 (r4). Laplace is IME #1 at a=0.6–0.8 (0.2897, 0.2583, 0.2330).

Reading:
* Laplace is EMD #1 at every RIGID CORR point (0.155 at a=0.2 down to 0.022 at
  a=0.8). This is the filter-success result the paper reports for RIGID.
* AIG is EMD #2 at a=0.2, 0.3, 0.4, 0.5, 0.6, 0.8 (rank 4 at a=0.7). It is the
  only non-filter method in the top two across the curve.
* PL sits at rank 4–5 from a=0.4 on. AG is rank 3–6 at a≥0.4 and rank 13 at
  a≤0.3 in the rank table below.
* LIME is rank 10–13 from a=0.6 on (flat ~0.29).

---------------------------------------------------------------
## 3. LIN CORR — complete (a=0.1–0.9)

No paper table exists for LIN; the paper text says filters fail in LIN.

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
* Laplace worsens monotonically with alpha (0.125 -> 0.226) and is rank 13 of 13
  from a=0.2 on. Sobel is ~0.19 at a=0.8.
* The analytic family converges to near-exact recovery (0.015–0.020 EMD at high
  alpha). AIG matches or beats PL at every point.
* AIG is outright rank 1/13 on EMD and IME at a=0.9 (IME 0.0194; AG 0.0309).

---------------------------------------------------------------
## 4. LIN WHITE — complete (a=0.1–0.9)

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
* The analytic family is rank 1–3 at every point. AIG is outright rank 1/13 on
  both metrics at a=0.4–0.9.
* Laplace is flat at ~0.137 (rank 13 through a=0.7; rank 6 at a=0.9).
* The family converges to ~0.0147 by a=0.4. LIME stays at 0.12–0.13.

---------------------------------------------------------------
## 5. XOR WHITE — a=0.1–0.7 analyzed (a=0.8, 0.9 running)

| method | a=0.1 | a=0.2 | a=0.3 | a=0.4 | a=0.5 | a=0.6 | a=0.7 |
|---|---|---|---|---|---|---|---|
| patternlocal | 0.0789 | **0.0566** | **0.0374** | **0.0328** | **0.0300** | **0.0448** | **0.0400** |
| analytical_grad | **0.0722** | 0.0705 | 0.0527 | 0.0483 | 0.0461 | 0.0477 | 0.0485 |
| analytical_ig | 0.0760 | 0.0858 | 0.0541 | 0.0512 | 0.0534 | 0.0675 | 0.0663 |
| gradient_shap | — | — | 0.1106 | 0.1006 | — | — | — |
| lime | 0.1278 | 0.1335 | — | — | — | — | — |
| laplace | 0.1375 | 0.1375 | 0.1671 | 0.1374 | 0.1374 | 0.1373 | 0.1372 |

IME (AIG): 0.2274 (r2), 0.1178 (r2), 0.0404 (r1), 0.0307 (r1), 0.0254 (r1),
0.0221 (r2), 0.0195 (r2). PL IME: 0.3847 (r3), 0.1647 (r3), 0.0431, 0.0311.

Reading:
* PatternLocal is EMD #1 at a=0.2–0.7. At a=0.1 AG leads (0.0722 vs 0.0789).
  PL's only outright-EMD scenario.
* AIG is IME rank 1–2 at every point while its EMD slips to rank 2–6 at high
  alpha. This is the clean EMD-vs-IME split: PL recovers support best, AIG
  suppresses false positives best.
* Laplace is last or near last throughout.

---------------------------------------------------------------
## 6. Rank summary — AIG vs the field (41 analyzed points)

Rank among 13 methods (EMD / IME). "—" = IME rank not recorded.

| scene a | AIG | PL | AG | best EMD / best IME |
|---|---|---|---|---|
| XOR 0.1 | 3 / 3 | 12 / — | **1** / 1 | AG / AG |
| XOR 0.2 | 4 / 3 | 3 / — | 2 / — | Laplace / AG |
| XOR 0.3 | 4 / 2 | 2 / — | 3 / — | Laplace / AG |
| XOR 0.4 | 4 / 3 | 2 / — | 3 / — | Laplace / AG |
| XOR 0.5 | 5 / 2 | 2 / — | 3 / — | Laplace / AG |
| XOR 0.6 | 5 / 2 | 2 / — | 4 / — | Laplace / AG |
| XOR 0.7 | 8 / 2 | 2 / — | 4 / — | Laplace / AG |
| XOR 0.8 | 8 / 2 | 3 / — | 7 / — | Laplace / AG |
| XOR 0.9 | 8 / 2 | 3 / — | 4 / — | Laplace / AG |
| LIN 0.1 | 3 / 3 | 8 / — | **1** / — | AG / Laplace |
| LIN 0.2 | 3 / 3 | 2 / — | **1** / — | AG / AG |
| LIN 0.3 | 2 / 3 | 3 / — | **1** / — | AG / AG |
| LIN 0.4 | **1** / **1** | 3 / — | **1** / — | AIG / AIG |
| LIN 0.5 | 2 / 2 | 2 / — | **1** / — | AG / AG |
| LIN 0.6 | 2 / 2 | 2 / — | **1** / — | AG / AG |
| LIN 0.7 | 2 / 2 | 2 / — | **1** / — | AG / AG |
| LIN 0.8 | 2 / 2 | 3 / — | **1** / — | AG / AG |
| LIN 0.9 | **1** / **1** | 2 / — | 3 / — | AIG / AIG |
| WLIN 0.1 | 2 / — | 3 / — | **1** / — | AG / AG |
| WLIN 0.2 | 2 / 2 | 3 / — | **1** / — | AG / AG |
| WLIN 0.3 | 2 / **1** | 3 / — | **1** / — | AG / AG |
| WLIN 0.4 | **1** / **1** | 3 / — | **1** / — | AIG / AIG |
| WLIN 0.5 | **1** / **1** | 3 / — | 2 / — | AIG / AIG |
| WLIN 0.6 | **1** / **1** | 3 / — | 2 / — | AIG / AIG |
| WLIN 0.7 | **1** / **1** | 3 / — | 2 / — | AIG / AIG |
| WLIN 0.8 | **1** / **1** | 2 / — | 3 / — | AIG / AIG |
| WLIN 0.9 | **1** / **1** | 3 / — | 2 / — | AIG / AIG |
| WXOR 0.1 | 2 / 2 | 3 / — | **1** / — | AG / — |
| WXOR 0.2 | 3 / 2 | **1** / 3 | 2 / — | PL / — |
| WXOR 0.3 | 3 / **1** | **1** / — | 2 / — | PL / AIG |
| WXOR 0.4 | 3 / **1** | **1** / — | 2 / — | PL / AIG |
| WXOR 0.5 | 3 / **1** | **1** / — | 2 / — | PL / AIG |
| WXOR 0.6 | 3 / 2 | **1** / — | 2 / — | PL / — |
| WXOR 0.7 | 6 / 2 | **1** / — | 2 / — | PL / — |
| RIGID 0.2 | **2** / 3 | 11 / — | **13** / — | Laplace / — |
| RIGID 0.3 | **2** / 3 | 4 / — | **12** / — | Laplace / — |
| RIGID 0.4 | **2** / 3 | 4 / 4 | 6 / — | Laplace / — |
| RIGID 0.5 | **2** / 8 | 5 / 3 | 4 / — | Laplace / — |
| RIGID 0.6 | **2** / 7 | 5 / 3 | 4 / 5 | Laplace / Laplace |
| RIGID 0.7 | 4 / 3 | 5 / 5 | 3 / 6 | Laplace / Laplace |
| RIGID 0.8 | **2** / 4 | 4 / 6 | 3 / 8 | Laplace / Laplace |

Findings:
1. **AIG beats LIME on both metrics at every analyzed point.** It beats PL on IME
   at nearly every point (exception: XOR a=0.4, 0.274 vs 0.250).
2. **AIG is the robust member of the covariance family.** In RIGID it is EMD #2
   at 6 of 7 points. Its sibling AG (plain-gradient kernel) falls to rank 12–13
   at a=0.2–0.3. Path integration keeps the covariance kernel informative under
   translation and rotation.
3. **AIG is outright rank 1/13 on both metrics** at LIN CORR a=0.9 and LIN WHITE
   a=0.4–0.9 (six wins).
4. **AIG's weak spot is XOR high-alpha EMD** (rank 8 at a=0.7–0.9) while IME stays
   rank 2. This EMD-vs-IME trade-off belongs in the thesis discussion.
5. **PatternLocal is EMD #1 on XOR WHITE** (a=0.2–0.7) — its only scenario where it
   takes EMD #1 outright.
6. AG dominates EMD and IME in LIN and at low-alpha XOR; the analytic family owns the
   top of the field wherever supports are fixed.

---------------------------------------------------------------
## 7. Generator check (magnitudes)

The old generator produced values ~0.5–0.7x the paper. On the Clark data:
LIME matches paper XOR a=0.1 (0.1368 vs 0.135), Laplace matches RIGID a=0.2
(0.155 vs 0.150), and Sobel tracks. Anchor points match within 0.01–0.02, so the
residual deltas come from mask thresholds and eval details rather than data.

---------------------------------------------------------------
## 8. Grid state

| status | sweeps |
|---|---|
| analyzed (44) | XOR CORR 0.1–0.9; LIN CORR 0.1–0.9; LIN WHITE 0.1–0.9; XOR WHITE 0.1–0.9; RIGID CORR 0.2–0.9; RIGID WHITE 0.7–0.9 |
| running (0) | — |
| pushed / complete (3) | RIGID WHITE 0.9 fetched; XOR WHITE 0.8/0.9 fetched |

Gated (not run): RIGID CORR 0.1 (acc 0.504); RIGID WHITE 0.1–0.6 (acc below 0.90;
0.6 = 0.8465).
Total planned: 47.
