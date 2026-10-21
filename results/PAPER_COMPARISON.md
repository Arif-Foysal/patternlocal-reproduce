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

## Reading (as of 3 points)

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
