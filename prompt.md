
You are a rigorous ML research reproduction engineer.

GOAL
Reproduce the main experimental findings(Figure 4) of the NeurIPS 2025 paper:
"Minimizing False-Positive Attributions in Explanations of
Non-Linear Models" using the official PatternLocal repository.

Released code by author: @Baseline\ Paper/PatternLocal/ 
Paper: references/patternlocal_paper.pdf

Do not assume the public repository contains the complete research pipeline. Inspect the real source code before implementing anything.

PHASE 1 — REPOSITORY AND PAPER AUDIT

1. Inspect the repository tree, git history, dependencies, examples, tests, and all existing PatternLocal implementations.
2. Locate the actual implementation of the local surrogate and
   PatternLocal's weighted regression.
3. Compare the implementation against Equations (5)–(7) in the
   paper. Verify locality weighting, simplification, regularization, and normalization. Do not silently replace or modify the algorithm.
4. Identify whether the repository contains:
   - The original XAI-TRIS data generator.
   - The original classifiers and training code.
   - EMD and IME evaluation.
   - Hyperparameter optimization scripts.
   - The artificial-lesion MRI and EEG pipelines.
5. Create reproduction_audit.md documenting every component found, every missing component, version mismatches, and a proposed plan.

PHASE 2 — REPRODUCE FIGURE 4 FIRST

Implement a separate paper_repro/ experiment directory. Preserve the
upstream package and avoid unreviewed changes to its public API.

Reproduce the XAI-TRIS MLP experiment using the paper's specifications:

- 40,000 synthetic 64x64 images.
- LIN, XOR, and RIGID task structures.
- WHITE and CORR noise conditions.
- Signal-to-noise parameter alpha = 0.1, 0.2, ..., 0.9.
- Exact dataset generation and ground-truth masks from Appendix C.1.
- Train/validation/test split of 90/5/5.
- MLP hidden dimensions 128, 64, 32, 16; ReLU, batch
  normalization and 25% dropout as specified.
- Adam, learning rate 1e-4, batch size 128, validation-based
  early stopping, and a maximum of 500 epochs.
- PatternLocal using the identity representation and L2
  regularization for the main Figure 4 comparison.
- LIME and the paper's other baselines, with the appropriate
  original implementation and parameter settings.
- Earth Mover's Distance (EMD) and Importance Mass Error (IME).

Read the complete relevant appendices rather than guessing details.

PHASE 3 — VALIDATION GATES

Before expensive experiments:

1. Test dataset shapes, class balance, generated masks, and split
   isolation.
2. Unit-test EMD and IME against hand-computable examples.
3. Run a small version of the paper's XOR suppressor toy problem.
   Check whether the irrelevant suppressor attribution is reduced.
4. Confirm classifier test accuracy exceeds the paper's 90% criterion.
5. Run one XAI-TRIS configuration using fixed parameters.
6. Compare PatternLocal and LIME explanations visually and
   quantitatively.

If the existing package does not reproduce the paper's method,
document the discrepancy and implement a separate faithful version
in paper_repro/ rather than hiding the difference.

PHASE 4 — TWO EXECUTION MODES

Provide two commands:

QUICK MODE:
One configuration, a small number of samples, and a reduced
hyperparameter search for debugging. Label all results as preliminary.
Do not present them as a full reproduction.

FULL MODE:
The original dataset sizes, complete scenario and alpha grid, paper model training, required baselines, and 200 TPE trials per applicable optimization run.

The full mode must be explicitly requested. Never launch the complete search automatically.

Make the search resumable and deterministic. Cache datasets,
checkpoints and reusable predictions. Save each trial's configuration, validation score, random seed, duration, errors and results.

PHASE 5 — RESULTS AND REPORTING

Create:
- One CLI entry point with quick/full modes.
- Reproducible environment/dependency files.
- Unit tests and smoke tests.
- CSV/JSON experiment logs.
- Figure 4-style EMD and IME curves with uncertainty estimates.
- A report comparing reproduced trends against the paper.
- Exact commands for the MRI and EEG extensions.
- A README explaining how to run every stage.

Never fabricate missing results or claim exact reproduction when
settings differ. Log all deviations from the paper.

First complete PHASE 1 and run the smallest smoke test. Report the audit and estimated full-run cost before starting any expensive search.
