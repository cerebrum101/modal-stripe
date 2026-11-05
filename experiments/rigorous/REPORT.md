# Controlled spectral-hybrid study

Primary model settings were frozen before fetching the final books. The calibrated-time baseline follows a rule specified before final-score inspection; the local-only control and corrective fixed-window diagnostic are explicitly exploratory additions. Final texts are Frankenstein and Sherlock Holmes. The first table always evaluates reset128-character windows, including for models trained with context512; use the identical-target context table for matched long-context comparisons. Three seeds describe optimization spread, not independent datasets. Quality uses fixed ASCII128 next-character prediction. No claims about useful-scale LLMs or GPU performance follow from this CPU study.

## Final book results with reset128 evaluation

| Phase | Budget | Train context | Model | Parameters | Updates | Training seconds | Mean test nats ± seed SD | Geometric test PPL |
|---|---|---:|---|---:|---:|---:|---:|---:|
| calibrated_time | parameters | 128 | full | 120768 | 2439 | 43.6 | 1.7406 ± 0.0045 | 5.701 |
| equal_time | parameters | 128 | full | 120768 | 4391 | 85.3 | 1.6472 ± 0.0073 | 5.192 |
| main | fixed_ff | 128 | full | 119488 | 1800 | 45.6 | 1.8015 ± 0.0084 | 6.058 |
| main | fixed_ff | 128 | localglobal | 119488 | 1800 | 59.3 | 1.7843 ± 0.0032 | 5.955 |
| main | fixed_ff | 128 | striped | 136913 | 1800 | 66.8 | 1.7060 ± 0.0036 | 5.507 |
| main | parameters | 128 | full | 120768 | 1800 | 64.4 | 1.7996 ± 0.0093 | 6.047 |
| main | parameters | 128 | localglobal | 120768 | 1800 | 77.0 | 1.7857 ± 0.0033 | 5.964 |
| main | parameters | 128 | striped | 120657 | 1800 | 85.3 | 1.7213 ± 0.0016 | 5.592 |
| main | parameters | 512 | full | 120768 | 1800 | 55.2 | 2.0051 ± 0.0060 | 7.427 |
| main | parameters | 512 | localglobal | 120768 | 1800 | 49.5 | 1.8211 ± 0.0115 | 6.179 |
| main | parameters | 512 | striped | 120657 | 1800 | 59.1 | 1.7508 ± 0.0028 | 5.759 |
| posthoc_local | parameters | 128 | localonly | 120768 | 1800 | 41.9 | 1.7676 ± 0.0022 | 5.857 |

## Spectral attribution

Differences below are striped minus localglobal cross-entropy on the same seeds. Negative favors adding spectral memory. Both parameter budgets are reported; fixed_ff permits extra parameters in striped.

| Budget | Train context | Paired nats differences by seed | Mean difference |
|---|---:|---|---:|
| parameters | 128 | -0.0684, -0.0645, -0.0602 | -0.0644 |
| fixed_ff | 128 | -0.0818, -0.0750, -0.0781 | -0.0783 |
| parameters | 512 | -0.0794, -0.0618, -0.0695 | -0.0703 |

## Prefix-plus-continuation extrapolation

The same eight 128-character continuation blocks per final book are evaluated after each prefix. The scored positions run from prefix-1 through prefix+126; the 128-prefix row therefore already exceeds the position range of context128 training. This is a continuation experiment, not an exact fixed-window baseline. This sample is limited; it is not a broad long-context benchmark.

| Training context | Model | Prefix | Mean test nats | Geometric PPL |
|---:|---|---:|---:|---:|
| 128 | full | 128 | 2.4913 | 12.077 |
| 128 | full | 512 | 2.5329 | 12.590 |
| 128 | full | 2048 | 2.7326 | 15.373 |
| 128 | full | 8192 | 2.7069 | 14.983 |
| 128 | localglobal | 128 | 2.1510 | 8.593 |
| 128 | localglobal | 512 | 2.2097 | 9.113 |
| 128 | localglobal | 2048 | 2.5397 | 12.676 |
| 128 | localglobal | 8192 | 2.4606 | 11.712 |
| 128 | striped | 128 | 1.9900 | 7.315 |
| 128 | striped | 512 | 2.0957 | 8.131 |
| 128 | striped | 2048 | 2.2884 | 9.859 |
| 128 | striped | 8192 | 2.2082 | 9.100 |
| 512 | full | 128 | 1.9764 | 7.216 |
| 512 | full | 512 | 2.0863 | 8.055 |
| 512 | full | 2048 | 2.2938 | 9.912 |
| 512 | full | 8192 | 2.2868 | 9.844 |
| 512 | localglobal | 128 | 1.7803 | 5.932 |
| 512 | localglobal | 512 | 1.9452 | 6.995 |
| 512 | localglobal | 2048 | 2.4476 | 11.561 |
| 512 | localglobal | 8192 | 2.3266 | 10.244 |
| 512 | striped | 128 | 1.7052 | 5.503 |
| 512 | striped | 512 | 1.8105 | 6.113 |
| 512 | striped | 2048 | 2.1290 | 8.406 |
| 512 | striped | 8192 | 2.1238 | 8.363 |

## Development learning curves

| Budget | Context | Model | Update | Mean development nats |
|---|---:|---|---:|---:|
| fixed_ff | 128 | full | 300 | 2.1682 |
| fixed_ff | 128 | full | 600 | 2.0313 |
| fixed_ff | 128 | full | 1200 | 1.8890 |
| fixed_ff | 128 | full | 1800 | 1.8410 |
| fixed_ff | 128 | localglobal | 300 | 2.0958 |
| fixed_ff | 128 | localglobal | 600 | 1.9661 |
| fixed_ff | 128 | localglobal | 1200 | 1.8572 |
| fixed_ff | 128 | localglobal | 1800 | 1.8179 |
| fixed_ff | 128 | striped | 300 | 2.0393 |
| fixed_ff | 128 | striped | 600 | 1.8974 |
| fixed_ff | 128 | striped | 1200 | 1.7829 |
| fixed_ff | 128 | striped | 1800 | 1.7438 |
| parameters | 128 | full | 300 | 2.1684 |
| parameters | 128 | full | 600 | 2.0316 |
| parameters | 128 | full | 1200 | 1.8878 |
| parameters | 128 | full | 1800 | 1.8387 |
| parameters | 128 | localglobal | 300 | 2.0946 |
| parameters | 128 | localglobal | 600 | 1.9669 |
| parameters | 128 | localglobal | 1200 | 1.8575 |
| parameters | 128 | localglobal | 1800 | 1.8201 |
| parameters | 128 | striped | 300 | 2.0451 |
| parameters | 128 | striped | 600 | 1.9053 |
| parameters | 128 | striped | 1200 | 1.7942 |
| parameters | 128 | striped | 1800 | 1.7588 |
| parameters | 512 | full | 300 | 2.2072 |
| parameters | 512 | full | 600 | 2.1349 |
| parameters | 512 | full | 1200 | 2.0647 |
| parameters | 512 | full | 1800 | 2.0318 |
| parameters | 512 | localglobal | 300 | 2.1244 |
| parameters | 512 | localglobal | 600 | 1.9945 |
| parameters | 512 | localglobal | 1200 | 1.8906 |
| parameters | 512 | localglobal | 1800 | 1.8518 |
| parameters | 512 | striped | 300 | 2.0708 |
| parameters | 512 | striped | 600 | 1.9330 |
| parameters | 512 | striped | 1200 | 1.8263 |
| parameters | 512 | striped | 1800 | 1.7911 |

## Paired randomized CPU/CUDA timing

All raw repetitions are retained in paired_timing.json. Tables use the median over repeats per seed, then the mean across three seeds. Context128 main models only. Cache is tensor payload, not process peak memory.

| Prefix | Model | Prefill ms | Decode ms/character | Cache KiB |
|---:|---|---:|---:|---:|
| 128 | full | 0.661 | 0.1459 | 160.25 |
| 128 | localglobal | 0.742 | 0.1444 | 84.25 |
| 128 | striped | 1.005 | 0.1774 | 88.25 |
| 512 | full | 2.755 | 0.1597 | 544.25 |
| 512 | localglobal | 2.244 | 0.1523 | 276.25 |
| 512 | striped | 3.092 | 0.1850 | 280.25 |
| 2048 | full | 20.276 | 0.2129 | 2080.25 |
| 2048 | localglobal | 14.241 | 0.1826 | 1044.25 |
| 2048 | striped | 18.006 | 0.2157 | 1048.25 |
| 8192 | full | 201.524 | 0.3268 | 8224.25 |
| 8192 | localglobal | 115.441 | 0.2335 | 4116.25 |
| 8192 | striped | 130.785 | 0.2723 | 4120.25 |

## Corrective exact-window diagnostic

Exploratory, seed0 only, 64 fixed target characters per final book. Each target is predicted from exactly the named number of preceding characters; position of the last input is window-1. Targets are identical across models/windows. The small target sample is descriptive, not a reliable broad quality estimate.

| Training context | Model | Input window | Mean nats | Geometric PPL |
|---:|---|---:|---:|---:|
| 128 | full | 128 | 1.7465 | 5.734 |
| 128 | full | 512 | 2.4223 | 11.271 |
| 128 | full | 2048 | 2.3579 | 10.569 |
| 128 | full | 8192 | 2.4022 | 11.048 |
| 128 | localglobal | 128 | 1.7465 | 5.734 |
| 128 | localglobal | 512 | 2.0467 | 7.742 |
| 128 | localglobal | 2048 | 2.5125 | 12.336 |
| 128 | localglobal | 8192 | 2.2287 | 9.288 |
| 128 | striped | 128 | 1.5258 | 4.599 |
| 128 | striped | 512 | 1.9690 | 7.164 |
| 128 | striped | 2048 | 1.9536 | 7.054 |
| 128 | striped | 8192 | 2.1726 | 8.781 |
| 512 | full | 128 | 1.9357 | 6.929 |
| 512 | full | 512 | 1.9696 | 7.168 |
| 512 | full | 2048 | 2.1772 | 8.822 |
| 512 | full | 8192 | 2.2274 | 9.276 |
| 512 | localglobal | 128 | 1.6663 | 5.292 |
| 512 | localglobal | 512 | 1.6932 | 5.437 |
| 512 | localglobal | 2048 | 2.6979 | 14.849 |
| 512 | localglobal | 8192 | 2.1953 | 8.983 |
| 512 | striped | 128 | 1.6236 | 5.071 |
| 512 | striped | 512 | 1.6455 | 5.184 |
| 512 | striped | 2048 | 2.0909 | 8.092 |
| 512 | striped | 8192 | 1.9276 | 6.873 |
| 128 | localonly | 128 | 1.6564 | 5.240 |
| 128 | localonly | 512 | 2.1542 | 8.621 |
| 128 | localonly | 2048 | 2.0742 | 7.958 |
| 128 | localonly | 8192 | 2.4725 | 11.852 |

## Exploratory local-only efficiency control

This baseline was added after primary test outcomes were inspected. Its learning rate was selected only on development data, but the architecture comparison is explicitly post hoc. It is a diagnostic efficiency baseline, not an untouched confirmatory test.

| Model | Reset128 whole-book PPL | Prefix128 target-block PPL | Prefix8192 target-block PPL |
|---|---:|---:|---:|
| full | 6.047 | 12.077 | 14.983 |
| striped | 5.592 | 7.315 | 9.100 |
| localonly | 5.857 | 7.321 | 11.800 |

Exploratory randomized paired timing, seed0 only:

| Prefix | Model | Median decode ms | Cache KiB |
|---:|---|---:|---:|
| 128 | full | 0.1730 | 160.25 |
| 128 | striped | 0.2116 | 88.25 |
| 128 | localonly | 0.1732 | 8.25 |
| 8192 | full | 0.3716 | 8224.25 |
| 8192 | striped | 0.3224 | 4120.25 |
| 8192 | localonly | 0.1979 | 8.25 |

## Interleaved training-step timing

Three warmup updates, then twenty raw repeats per model/seed. Data sampling is excluded; forward, loss, backward, clipping and AdamW are included. Checkpoint files are unchanged.

| Context | Model | Mean of seed median step ms |
|---:|---|---:|
| 128 | full | 17.289 |
| 128 | localglobal | 19.812 |
| 128 | striped | 24.411 |
| 512 | full | 26.445 |
| 512 | localglobal | 23.439 |
| 512 | striped | 27.903 |

## Verification

All 36 final checkpoints passed the trained-model causality and cached-generation assertions, including an 8192-character prefix. Worst long-prefix logit difference: 6.2e-06.
Exact-block audit: 7669 sampled final-book blocks checked; 0 matched a training substring. This is not semantic deduplication.

## Interpretation limits

Striped retains one full global-attention block: its KV cache still grows linearly and its full-sequence attention work remains quadratic. The cache reduction is a constant-factor saving, not a bounded-memory architecture or an overall O(T log T) model. Local-only attention has a bounded receptive field, so its resource advantage does not establish distant retrieval ability. The legacy recall task used separately trained models. 1800 updates is a finite-budget comparison, not verified convergence. A seed0 learning-rate pilot is limited tuning. All selected rates were 0.002, the upper tested endpoint, so an optimum was not bracketed. Equal-time runs use noisy host wall time with a time-based learning-rate schedule, rather than FLOP matching. Final books supply two transfer domains. Development sources informed the architecture before this study. Long-context losses may expose severe extrapolation failures; always read them alongside speed. Loss values are not directly comparable with legacy experiments because the alphabet, splits and recipe differ. These ablations test the whole modal branch, not the Fourier basis against every alternative mixer. A half-cache result may come from replacing global attention alone; use localglobal to assess spectral attribution.
