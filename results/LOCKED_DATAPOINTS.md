# LOCKED datapoints (valid, reproducible results)

Generated 2026-10-21. These are the only results valid for analysis/plotting.

Excluded: all pre-2026-10-21 RIGID results (old broken generator; acc 0.9535/0.9955
vs regenerated) and quarantined dirs in `results/_stale/`.

n_eval = 570 test instances per config. EMD scored with POT (the paper's library). Bands: std over 570 samples.

## Provenance (are these from the latest code/data?)

**Eval code: YES — all 9 configs were produced by the current 13-method core with
the tuner fix.** Proof from the hyperparameters recorded *inside* each result file
(fields that only exist in the post-refactor core):

| config | pat_k_ratio | k_ig | k_grad | lime_bw |
|---|---|---|---|---|
| XOR_CORR a0.1 | 1.0 | 36000 | 36000 | 30 |
| XOR_CORR a0.2 | 0.05 | 1800 | 5000 | 30 |
| XOR_CORR a0.9 | 1.0 | 36000 | 36000 | 30 |
| LIN_WHITE a0.5 | 1.0 | 36000 | 36000 | 30 |
| RIGID_CORR a0.3 | 1.0 | 36000 | 36000 | 20 |

The `k_ig=36000 = n_train` picks are the tuner fix's floor behaviour
(k_floor = pat_k_ratio * n_train, always includes n_train); grids are
PAT_K=(0.05,0.5,1.0), LIME_BW=(10,20,30). XOR_CORR a=0.1 was additionally
re-run on GPU and produced a bit-identical PatternLocal EMD, ruling out
run-to-run flakiness.

**Training data: YES for RIGID (regenerated 2026-10-21 with the fixed generator;
per-config accs match the regenerated summary exactly). For LIN/XOR the npz files
were generated before the RIGID fix, but the fix is provably inert for them:
`apply_rigid_transform` is called only in the RIGID branch (LIN/XOR consume
`signal_cache` directly), the fix was additive (optional region/jitter params,
default behaviour unchanged), and masks were verified byte-identical before/after
(1054.0 px, std 0.0). LIN/XOR data-gen kernels therefore never needed re-running.**

**Validity boundary:** if the eval core or generator changes again, these points
must be re-run. Batch 2 (xor-corr 3-6, in flight) uses this exact code+data, so
the curves stay consistent.

## XOR_CORR: ours vs the paper's plot (`Baseline Paper/xor-corr-datapoints.md`)

"Locked" means **validated against our own code/data** — it does NOT mean
"matches the paper". Here is the explicit comparison (EMD; ours = full-set POT):

| method | ours a=0.1 -> 0.2 -> 0.9 | paper a=0.1 -> 0.2 -> 0.9 | trend match |
|---|---|---|---|
| PatternLocal | 0.132 -> 0.103 -> 0.035 | 0.100 -> 0.085 -> 0.048 | direction yes; we go BELOW the paper's floor 0.043 |
| LIME | 0.120 -> 0.121 -> 0.101 | 0.135 -> 0.125 -> 0.080 | direction yes; ours improves less |
| Sobel | 0.122 -> 0.108 -> **0.020** | 0.140 -> 0.130 -> 0.125 (flat) | **NO** - ours collapses 6x |
| Laplace | 0.092 -> 0.063 -> **0.016** | 0.135 -> 0.120 -> 0.080 | **NO** - ours collapses 6x |
| IntegratedGradients | 0.133 -> 0.135 -> **0.026** | 0.115 -> 0.110 -> 0.090 | **NO** - ours collapses |
| DeepLift | 0.125 -> 0.133 -> **0.027** | 0.112 -> 0.105 -> 0.080 | **NO** - ours collapses |
| GradientShap | 0.131 -> 0.133 -> **0.026** | 0.105 -> 0.110 -> 0.075 | **NO** - ours collapses |
| Saliency | 0.098 -> 0.113 -> 0.119 | 0.068 -> 0.080 -> 0.088 | direction yes; ours worse at low a (no Saliency dip) |
| GuidedBackprop | 0.094 -> 0.109 -> 0.117 | 0.105 -> 0.080 -> 0.078 | mixed |

Range across alpha, ours vs paper (collapse methods are 3-5x wider):

| method | ours range | paper range |
|---|---|---|
| PatternLocal | 0.098 | 0.057 |
| LIME | 0.020 | 0.057 |
| Saliency | 0.021 | 0.021 |
| GuidedBackprop | 0.023 | 0.030 |
| Sobel | **0.102** | 0.019 |
| Laplace | **0.076** | 0.060 |
| plain IG | **0.109** | 0.026 |
| DeepLift | **0.106** | 0.032 |
| GradientShap | **0.106** | 0.035 |

**The structural failure:** the paper's curves saturate at EMD >= 0.043 for EVERY
method (so at a=0.9 PatternLocal 0.048 stands out as uniquely best, next is 0.075).
In ours, 6 methods (Sobel, Laplace, IG, DeepLift, GradientShap, and PL itself at
0.035) reach or go below that floor, so at a=0.9 PatternLocal is NOT the single
best method in our reproduction - Laplace (0.016), Sobel (0.020), plain IG
(0.026), GradientShap (0.026) and DeepLift (0.027) beat it. In the paper's world
PL is #1 by 0.027; in ours it is ~#6.

Consequences for the reproduction claims:
* REPRODUCED: PL beats LIME clearly at a>=0.2 on both metrics; IME collapse.
* NOT REPRODUCED: PL as uniquely-best method in XOR at high SNR; the paper's
  filter/IG-family baseline magnitudes and flat-vs-alpha trend.

Root-cause status: exhausted implementation space (12 filter variants, mask
geometry, POT verification, Eq. 10-14 all verified character-exact); conclusion is
a data-level difference in the un-shipped XAI-TRIS generator (Clark et al. [8]).
Full analysis: results/FILTER_DIVERGENCE_REPORT.md. Decision (2026-10-21):
document and move on - do not tune to match.

## Sweep status

| scenario | locked | running | missing |
|---|---|---|---|
| XOR_CORR | 0.1, 0.2, 0.9 | 3,4,5,6 | 7,8 |
| LIN_WHITE | 0.5 | - | 1,2,3,4,6,7,8,9 |
| LIN_CORR | - | - | 1-9 |
| XOR_WHITE | - | - | 1-9 |
| RIGID_CORR | 0.1,0.2,0.3,0.4,0.5 | - | 7,8,9 |
| RIGID_WHITE | - | 5 | 1,2,3,4,6,7,8,9 |

## RIGID_CORR (fixed generator)

test acc: a0.1=0.9930, a0.2=0.9985, a0.3=1.0000, a0.4=0.9985, a0.5=0.9990

**EMD**

| method | a=0.1 | a=0.2 | a=0.3 | a=0.4 | a=0.5 |
|---|---|---|---|---|---|
| patternlocal | 0.1653 | 0.1414 | 0.1230 | 0.1034 | 0.1127 |
| analytical_ig | 0.1348 | 0.1018 | 0.1069 | 0.0885 | 0.1128 |
| analytical_grad | 0.1326 | 0.1132 | 0.1201 | 0.1228 | 0.1552 |
| lime | 0.1433 | 0.1396 | 0.1374 | 0.1354 | 0.1333 |
| saliency | 0.1507 | 0.1283 | 0.1287 | 0.1332 | 0.1402 |
| deeplift | 0.1623 | 0.1388 | 0.1336 | 0.1247 | 0.1142 |
| gradient_shap | 0.1452 | 0.1321 | 0.1341 | 0.1162 | 0.1115 |
| guided_backprop | 0.1410 | 0.1128 | 0.1101 | 0.1149 | 0.1301 |
| plain_grad | 0.1507 | 0.1283 | 0.1287 | 0.1332 | 0.1402 |
| plain_ig | 0.1611 | 0.1314 | 0.1291 | 0.1146 | 0.1105 |
| plain_smoothgrad | 0.1527 | 0.1278 | 0.1287 | 0.1332 | 0.1401 |
| sobel | 0.1433 | 0.1288 | 0.1096 | 0.0917 | 0.0764 |
| laplace | 0.1089 | 0.0739 | 0.0541 | 0.0425 | 0.0351 |
**IME**

| method | a=0.1 | a=0.2 | a=0.3 | a=0.4 | a=0.5 |
|---|---|---|---|---|---|
| patternlocal | 0.7383 | 0.6774 | 0.5810 | 0.4976 | 0.4571 |
| analytical_ig | 0.6402 | 0.5762 | 0.5259 | 0.4671 | 0.4162 |
| analytical_grad | 0.6042 | 0.4949 | 0.4645 | 0.4496 | 0.5209 |
| lime | 0.7406 | 0.7320 | 0.7218 | 0.7065 | 0.6888 |
| saliency | 0.7020 | 0.6845 | 0.6779 | 0.6892 | 0.7307 |
| deeplift | 0.7027 | 0.6696 | 0.6299 | 0.5775 | 0.5453 |
| gradient_shap | 0.6852 | 0.6670 | 0.6317 | 0.5670 | 0.5373 |
| guided_backprop | 0.7005 | 0.6810 | 0.6698 | 0.6787 | 0.7208 |
| plain_grad | 0.7020 | 0.6845 | 0.6779 | 0.6892 | 0.7307 |
| plain_ig | 0.7045 | 0.6665 | 0.6260 | 0.5677 | 0.5347 |
| plain_smoothgrad | 0.7041 | 0.6847 | 0.6777 | 0.6890 | 0.7304 |
| sobel | 0.7075 | 0.6325 | 0.5412 | 0.4519 | 0.3746 |
| laplace | 0.6173 | 0.4911 | 0.4179 | 0.3720 | 0.3395 |

## XOR_CORR

test acc: a0.1=1.0000, a0.2=1.0000, a0.9=1.0000

**EMD**

| method | a=0.1 | a=0.2 | a=0.9 |
|---|---|---|---|
| patternlocal | 0.1324 | 0.1029 | 0.0346 |
| analytical_ig | 0.1303 | 0.0991 | 0.0434 |
| analytical_grad | 0.1134 | 0.0883 | 0.0388 |
| lime | 0.1202 | 0.1207 | 0.1012 |
| saliency | 0.0984 | 0.1130 | 0.1189 |
| deeplift | 0.1254 | 0.1333 | 0.0269 |
| gradient_shap | 0.1308 | 0.1333 | 0.0264 |
| guided_backprop | 0.0938 | 0.1086 | 0.1171 |
| plain_grad | 0.0984 | 0.1130 | 0.1189 |
| plain_ig | 0.1333 | 0.1349 | 0.0259 |
| plain_smoothgrad | 0.0978 | 0.1128 | 0.1188 |
| sobel | 0.1223 | 0.1084 | 0.0200 |
| laplace | 0.0920 | 0.0633 | 0.0162 |
**IME**

| method | a=0.1 | a=0.2 | a=0.9 |
|---|---|---|---|
| patternlocal | 0.7032 | 0.5838 | 0.0029 |
| analytical_ig | 0.6435 | 0.5340 | 0.0028 |
| analytical_grad | 0.5970 | 0.4862 | 0.0021 |
| lime | 0.7357 | 0.7359 | 0.6364 |
| saliency | 0.6485 | 0.7130 | 0.7333 |
| deeplift | 0.6736 | 0.7175 | 0.1292 |
| gradient_shap | 0.7013 | 0.7170 | 0.1272 |
| guided_backprop | 0.6355 | 0.6966 | 0.7239 |
| plain_grad | 0.6485 | 0.7130 | 0.7333 |
| plain_ig | 0.7066 | 0.7234 | 0.1274 |
| plain_smoothgrad | 0.6471 | 0.7123 | 0.7327 |
| sobel | 0.6997 | 0.6231 | 0.1223 |
| laplace | 0.6269 | 0.5101 | 0.2980 |

## LIN_WHITE

test acc: a0.5=1.0000

**EMD**

| method | a=0.5 |
|---|---|
| patternlocal | 0.0152 |
| analytical_ig | 0.0152 |
| analytical_grad | 0.0152 |
| lime | 0.1097 |
| saliency | 0.1170 |
| deeplift | 0.0963 |
| gradient_shap | 0.0950 |
| guided_backprop | 0.1096 |
| plain_grad | 0.1170 |
| plain_ig | 0.0950 |
| plain_smoothgrad | 0.1168 |
| sobel | 0.1150 |
| laplace | 0.1204 |
**IME**

| method | a=0.5 |
|---|---|
| patternlocal | 0.0103 |
| analytical_ig | 0.0099 |
| analytical_grad | 0.0103 |
| lime | 0.6779 |
| saliency | 0.7165 |
| deeplift | 0.5442 |
| gradient_shap | 0.5153 |
| guided_backprop | 0.6802 |
| plain_grad | 0.7165 |
| plain_ig | 0.5139 |
| plain_smoothgrad | 0.7154 |
| sobel | 0.6995 |
| laplace | 0.7387 |
