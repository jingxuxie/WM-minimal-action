# Measured results

All values below come from executed synthetic CPU experiments. Means include right-censored runs at their caps. Bootstrap intervals are 95% percentile intervals (4,000 resamples).

## Exact-model nuisance sweep

| gamma | method | capped mean [95% CI] | certified / runs | incorrect task certificates |
|---:|---|---:|---:|---:|
| 0.01 | Task design / task stop | 78.25 [74.36, 82.29] | 200/200 | 1 |
| 0.01 | Full design / task stop | 625.79 [436.94, 875.40] | 200/200 | 1 |
| 0.01 | Full design / full stop | 7816.97 [7274.03, 8353.36] | 200/200 | 0 |
| 0.01 | Task design / full stop | 18839.59 [17553.28, 20130.84] | 199/200 | 0 |
| 0.01 | Uniform / task stop | 589.97 [551.18, 629.35] | 200/200 | 0 |
| 0.01 | Entropy / task stop | 64.60 [61.55, 67.84] | 200/200 | 0 |
| 0.02 | Task design / task stop | 77.40 [73.58, 81.50] | 200/200 | 1 |
| 0.02 | Full design / task stop | 1031.03 [891.62, 1175.73] | 200/200 | 0 |
| 0.02 | Full design / full stop | 2258.16 [2096.89, 2429.74] | 200/200 | 0 |
| 0.02 | Task design / full stop | 4528.39 [4169.79, 4898.66] | 200/200 | 0 |
| 0.02 | Uniform / task stop | 611.48 [574.09, 650.44] | 200/200 | 0 |
| 0.02 | Entropy / task stop | 68.56 [65.24, 72.14] | 200/200 | 0 |
| 0.04 | Task design / task stop | 81.60 [77.80, 85.47] | 200/200 | 1 |
| 0.04 | Full design / task stop | 259.90 [229.11, 292.50] | 200/200 | 0 |
| 0.04 | Full design / full stop | 576.40 [540.44, 613.14] | 200/200 | 0 |
| 0.04 | Task design / full stop | 1102.54 [1009.51, 1197.19] | 200/200 | 0 |
| 0.04 | Uniform / task stop | 653.25 [614.81, 696.22] | 200/200 | 0 |
| 0.04 | Entropy / task stop | 70.98 [67.93, 74.23] | 200/200 | 0 |
| 0.08 | Task design / task stop | 95.44 [90.30, 100.88] | 200/200 | 1 |
| 0.08 | Full design / task stop | 123.18 [116.05, 131.05] | 200/200 | 0 |
| 0.08 | Full design / full stop | 179.51 [170.69, 189.06] | 200/200 | 0 |
| 0.08 | Task design / full stop | 202.41 [189.62, 216.74] | 200/200 | 0 |
| 0.08 | Uniform / task stop | 830.62 [777.85, 890.05] | 200/200 | 0 |
| 0.08 | Entropy / task stop | 92.31 [87.20, 97.49] | 200/200 | 0 |

## Interpretation

The roughly 100x comparison at gamma=0.01 is against full mapping recovery, a harder objective; it is not a 100x improvement over all task-directed baselines. Entropy/task uses fewer samples than Task/task at ordinary confidence. At delta=1e-10, Task/task uses fewer samples than Entropy/task. The all-effects-required negative control makes Task/task and Full/full identical sample by sample.

## Learned models and misspecification

Learned-model runs use anonymous pure demonstrator IDs and categorical likelihood estimates. This is not a neural/video representation experiment. Robust radii use simulator truth as an explicit diagnostic oracle; a deployable radius estimator is not evaluated. The task deployment kernels remain known. Robust intervals are conservative and often abstain. Uncorrected certification can become systematically incorrect under context misalignment.

## Data accounting

There are 15800 recorded calibration runs across all five suites. Learned-model runs contain 20 independent pretraining seeds per sample size and 24 calibration episodes per pretrained model. Exact and confidence settings use 200 independent calibration replicas; paired random streams are shared between methods within each setting.
