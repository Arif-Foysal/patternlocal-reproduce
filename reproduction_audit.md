# PatternLocal Figure 4 Reproduction Audit

## Repository Audit

### Upstream PatternLocal Package (v2.0.0)
- **Location**: `Baseline Paper/PatternLocal/`
- **Git**: 3 commits, clean tree, remote `gjoelbye/PatternLocal.git`
- **Core algorithm**: `LocalCovarianceSolver` in `patternlocal/solvers/local_covariance.py`
- **Local solver base**: `patternlocal/solvers/local_base.py` — k-NN, kernel weighting, bandwidth

### Algorithm Comparison: Paper Eq. (7) vs. Implementation

**Paper Eq. (7):**
```
a_ℓ2 = Cov_Π[x, ỹ] / (Var_Π[ỹ] + λ)
```
where ỹ = w^T x, expectations are kernel-weighted over local neighborhood.

**Implementation (`local_covariance.py:104`):**
```python
patternlocal_weights = lime_weights @ local_cov
```
This computes `a = C_local * w` (equivalent to Cov_Π[x, ỹ]), but **omits the denominator** `(Var_Π[ỹ] + λ)`.

**Impact**: The denominator is a scalar. After normalization (EMD/IME use |s| normalized to unit mass), the scalar cancels. **Figure 4 results are unaffected.**

**Additional discrepancy**: The implementation uses `shrinkage_intensity` (Ledoit-Wolf covariance shrinkage) which is NOT in Eq. (7). The paper's λ is a ridge regression parameter, not covariance shrinkage.

### Components Status

| Component | In Repo? | Notes |
|-----------|----------|-------|
| XAI-TRIS data generator (App. C.1) | **NO** | Must implement from paper spec |
| MLP/CNN training code (App. D.1) | **NO** | PyTorch 2.6, PyTorch-Lightning 2.5 |
| EMD metric (Eq. 10) | **NO** | Needs POT library |
| IME metric (Eq. 11) | **NO** | Simple formula, easy to implement |
| LIME baseline | **YES** | Via `lime` package |
| PatternLocal (upstream) | **YES** | But omits variance normalization |
| Sobel, Laplace | **NO** | scipy.ndimage |
| Integrated Gradients, Saliency, DeepLift, GradientShap, GuidedBackprop | **NO** | Need PyTorch implementations |
| TPE optimization | **YES** | `hyperopt` in package |
| MRI pipeline | **NO** | Requires external HCP data |
| EEG pipeline | **NO** | Requires PhysioNet data |
| Figure 4 plotting | **NO** | matplotlib |

### Missing Dependencies (not installed locally)
- torch, torchvision (for model training)
- POT (Python Optimal Transport, for EMD)
- scikit-image (for superpixels)
- matplotlib (for plotting)

These are available on Kaggle notebooks.

### Proposed Plan

1. Build `src/paper_repro/` core library with all missing components
2. Build notebook templates for each phase
3. Run XOR toy problem first (smoke test)
4. Run XAI-TRIS data generation + training
5. Run single-config evaluation
6. Document full-mode commands

## Deviations from Paper

1. **Variance normalization omitted in upstream**: Does not affect normalized metrics
2. **Shrinkage vs. ridge**: Will use paper's λ in denominator for faithful implementation
3. **Python version**: Paper uses 3.13, we use available Kaggle Python
4. **Hardware**: Paper uses Intel Xeon E5-2650 v4, Kaggle uses different CPUs
