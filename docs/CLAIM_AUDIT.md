# Claim and readiness audit

## What has actually been completed

| Claim | Evidence in this version | Scope / caveat |
|---|---|---|
| Task-answer interaction lower bound | Theorem 1; stopped KL proof | A specialization of established change-of-measure methods, not a new general lower-bound technique |
| Relevant permutation-cycle characterization | Theorem 2; full proof; brute-force checks through five effects; LP equality tests | Nonnegative KL costs and a bijective shared mapping; polynomial design separation, not polynomial end-to-end inference |
| Uniform nuisance separation | Theorem 3; explicit four-effect construction and threshold classifier | A constructed multistep family with diagnostic reset access; not an empirical scaling law for video models |
| Anytime task certificate | Proposition 4; likelihood-ratio proof; algebra tests | Finite specified likelihood family; caps are abstentions |
| Asymptotic information optimality | Theorem 5; summable MLE-error and expected-stopping-time proof | Fixed fully identifiable finite instance, delta tending to zero; controlled-testing specialization; not uniform in nuisance separation |
| Component direct-sum cost | Corollary 6; lower and upper proofs | Independent unknown component permutations; supplied exact within-component correspondences |
| Robust likelihood and control certificate | Proposition 7; uniform log-error and coupling proofs | Needs justified radii; oracle radii only in experiments; often conservative |
| Finite fixed-design upper bound | Appendix proposition | Hellinger bound; task-changing alternatives only |
| Numerical separation from full identification | 15,800 recorded calibration runs across five suites | Largest ratio compares different target objectives; entropy/task beats TDG at ordinary confidence |

All proofs have been drafted and checked for consistency by an AI assistant. None has received independent human review or formal proof-assistant verification. Numerical tests corroborate finite cases; they do not establish universal mathematical statements.

## Completed validation

- Randomized cycle versus permutation equality checks and shortest-cycle enumeration tests.
- Cutting-plane versus full-alternative LP objective checks.
- Noiseless/passive-label symmetry and KL checks.
- Task/full allocation and stopping ablations.
- A strong entropy-allocation baseline sharing the same confidence certificate.
- All-effects-required negative control, identical task/full trajectories.
- Explicit right-censoring and incorrect-certificate records.
- Estimated categorical likelihoods with 20 pretraining seeds per sample size.
- Context-semantic mismatch that invalidates uncorrected certificates.
- Conservative robust mode, including widespread abstention.

## Not completed / not claimed

1. Independent expert review of proofs, novelty, and methodology.
2. Neural or video representation-learning validation; the current learned model uses anonymous pure demonstrator anchors.
3. A deployable, statistically justified data-dependent log-error radius and efficient robust allocation.
4. Online navigation cost, unknown task relevance, or learned cross-context correspondence graphs.
5. Scalability of full likelihood inference beyond factorial enumeration.
6. Uniform-in-instance finite-sample optimality for the adaptive TDG rule.
7. Real robot deployment or physical safety certification.

## Novelty positioning

Latent-action alignment already exists in ILPO/LAPO and more recent work. Identifiability up to a permutation is explicitly addressed by Schur (2026). Cross-context effect alignment is studied by Olaf-World. Controlled sensing, Track-and-Stop, and multiple-correct-answer identification already provide broad statistical foundations. The proposed distinct contribution is the task-relevant permutation-cycle formulation and its command-grounding implications. A broader review of structured pure exploration, assignment identification, and optimal experimental design remains necessary before a strong priority claim.

## Submission gates

The manuscript is a complete research draft, not a certified finished submission. Before submission: independently check each proof and reference; assess novelty against structured identification; strengthen at least one representation-learning experiment; review all failure/censoring counts; anonymize artifacts and paths; retrieve the unmodified official style; check current page limits and required AI disclosure; and verify that the human authors can take responsibility for every claim. These are specific unfinished validations, not a statement that the research question cannot be solved.
