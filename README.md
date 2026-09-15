# WM-minimal-action

**Grounding Only What Matters: Task-Relevant Permutation Cycles in Latent-Action World Models**

A theory-first, CPU-only research project on the interaction needed to turn a latent-effect model into an executable controller. The unknown command-to-effect mapping is a permutation. The task needs only a specified subset of effects.

## Current research status

This version contains a complete working manuscript with self-contained proof appendices, tested reference implementations, and 15,800 executed synthetic calibration runs. It is **not yet independently reviewed or claimed submission-ready**. The paper explicitly identifies the assumptions, strong baselines, failed certificates under misspecification, and conservative robust abstentions. No conference submission or acceptance is implied.

The primary structural result characterizes the hardest decision-changing alternative as a minimum-cost directed permutation cycle touching a required effect. This provides a shortest-cycle separation oracle for optimal experimental design. The likelihood stopping rule and asymptotic adaptive-testing argument specialize established sequential-testing methods; they are not claimed as new generic statistics.

## Main measured result

At nuisance separation `gamma=0.01`, 200 independent calibration runs per method, `delta=0.05`:

| Allocation / stopping | Mean queries | 95% bootstrap interval |
|---|---:|---:|
| Task / task (TDG) | 78.25 | [74.36, 82.29] |
| Full / full | 7816.97 | [7274.03, 8353.36] |
| Full / task | 625.80 | [436.94, 875.40] |
| Uniform / task | 589.97 | [551.18, 629.35] |
| Entropy / task | 64.60 | [61.55, 67.84] |

The approximately 100x comparison is against **the harder full-identification objective**, not every task-grounding baseline. Entropy allocation is faster than TDG at ordinary confidence. At `delta=1e-10`, in the separate two-context sweep, TDG uses 223.99 queries versus 375.46 for entropy/task. TDG has one incorrect task certificate in 200 runs in each ordinary-confidence nuisance setting; the data are not error-free.

When every effect is required, task/task and full/full are identical sample by sample. Under deliberately inconsistent calibration/deployment semantics, uncorrected certificates can all be wrong. The robust correction avoids wrong certificates in that test but abstains heavily. See `docs/RESULTS.md` and the manuscript for the full accounting.

## Reproduce

Python 3.10 or later; no GPU, pretrained-model download, external dataset, or model API is required.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.run_experiments --suite all
python -m experiments.analyze_results
python -m experiments.make_figures
python paper/build.py
```

On Windows, set the three environment variables in the shell or omit them. The statistical code is platform independent; LP tie choices and floating-point differences across SciPy versions can change individual trajectories. Tested dependency versions are recorded in `results/metadata_*.json` and `requirements-tested.txt`.

A quick smoke run is:

```bash
python -m experiments.run_experiments --suite exact --repetitions 10 --cap 500 --output results/smoke
```

The paper build needs a TeX installation with `pdflatex` and either `bibtex` or `bibtex8`. Figures can be regenerated from the committed summary tables without rerunning calibration. The development manuscript uses ICLR 2027 layout with an explicit not-submitted header; final formatting and anonymization need human preflight.

## Files

- `paper/main.tex`, `paper/proofs.tex`, `paper/experimental_details.tex`: manuscript and proofs.
- `paper/references.bib`: verified primary-source bibliography.
- `wm_grounding/core.py`: candidate models, optimal designs, cycle oracle, confidence sets, and common-policy certificate.
- `wm_grounding/simulation.py`: independent calibration replicas, paired random streams, censoring, and exact deployment evaluation.
- `wm_grounding/environments.py`: diagnostic chains and estimated categorical effects.
- `experiments/`: reproduction, aggregation, and figures.
- `tests/`: numerical checks of the cycle characterization, likelihood algebra, and implementation.
- `results/*_summary.csv`: aggregate results; `docs/RESULTS.md` includes intervals and errors.
- `docs/CLAIM_AUDIT.md`: proof status, scope, and remaining submission gates.

The GitHub source release contains source code and aggregate results. The accompanying full research bundle contains all raw run records, model estimates, figures, and the compiled PDF. Running the commands above regenerates those artifacts locally. Do not confuse aggregate tables with unshared additional experiments.

## Scope and limitations

The theory assumes known or bounded-error latent effect kernels, a shared command permutation, known task-required effects, and resettable calibration contexts. It does not solve arbitrary-video latent-action discovery, online navigation to diagnostic states, or continuous robot actuation. The learned-model experiment estimates categorical kernels from **anonymous pure demonstrator IDs**, not a neural encoder. Its robust error radii are computed from simulator truth and are diagnostic oracles, not deployable estimators. The design oracle avoids factorial alternative enumeration, but the reference inference code still enumerates permutations.

## AI involvement

An OpenAI ChatGPT assistant contributed formulation, literature research, proofs, implementation, executed synthetic experiments, analysis, and writing. The numbers come from code execution. Automated tests are not independent mathematical review. Human authors must verify correctness, novelty, citations, reproducibility, and disclosure before submission. The public repository must not be linked directly from a double-blind manuscript.
