# Audit of the text head-to-head

**Audited follow-up:** [rigorous results](../rigorous/REPORT.md) add parameter controls, tuned learning rates, matched-time comparisons, unseen books, paired timing, and corrected long-context diagnostics. This file preserves the earlier stage and should be read alongside that follow-up.

The new independent audit scores every held-out target once, in fixed 128-character chunks, and checks future-token perturbations and cached generation on all 12 trained checkpoints. Results: `headtohead_audit.json`; code: `experiments/audit_headtohead.py`.

## Findings that change how to present the result

1. **Missing attribution control.** The striped block adds both spectral memory and local attention. Local attention already nearly matched spectral variants on the original Tiny Shakespeare experiment. The new result cannot attribute gains to the Fourier/modal component. Add a local-plus-global model with identical mixer order and parameter budget, then compare it with striped.
2. **Equal tokens is not equal compute.** Striped trains slower. A full-attention baseline trained for the same measured compute budget may close the gap. Run equal-compute comparisons and longer learning curves before claiming superior quality or efficiency.
3. **Long-prefix speed is not long-context quality.** Training and quality evaluation used 128-character windows, while performance was measured up to 8192. The fast long-prefix path may generate worse predictions at those lengths. Evaluate held-out cross-entropy at several context lengths before combining quality and long-context cost in one claim.
4. **Timing claims were too strong.** At 8192, mean decode times were 0.618 versus 0.594 ms, but checkpoint-to-checkpoint spreads were 0.109 and 0.078 ms. Treat decode as tied; do not claim a demonstrated decode crossover or speed advantage. Prefill and tensor-cache savings are stronger descriptive results. Save raw repetitions and use interleaved randomized paired timing before precise speed claims.
5. **Repeated holdout use.** These validation tails have been inspected while selecting the architecture. They are development data, not a pristine final test. Reserve unseen documents for final reporting. Three seeds are not three independent datasets.
6. **Parameter matching changes the feed-forward width.** This compares complete architectures at approximately equal parameter counts. It does not isolate one operator. A fixed-width feed-forward ablation would help explain the improvement.
7. **Minor sampler bug.** `len(data)-length-1` omits two valid final window starts. It affects both models equally and is too small to explain the observed gap. Preserve old-run provenance when fixing it and rerun comparisons under the corrected sampler.
8. **Vocabulary/provenance weakness.** Alphabet construction inspects all texts, including validation characters. This exposes character inventory, not validation sequences, but should be declared or replaced with a fixed alphabet. Skip logic checks steps only, not code hashes, data hashes, or full configuration. Save those before publishing a reproducible benchmark.
9. **No convergence or tuning study.** One learning rate and a short schedule may favor one architecture. Show longer curves and a small equal-budget learning-rate sweep before general superiority claims.

## Focused verification

The deterministic holdout audit preserves the striped model's lower perplexity. Trained-model future perturbation and cached-vs-full logits agree to float32 tolerances on the tested inputs. These checks reduce concern about obvious causal leakage or broken decode equivalence; they do not establish general model quality.

## Publishing decision

Suitable as a clearly labeled exploratory engineering blog with released code and limitations. Premature as a methods preprint claiming a novel superior spectral language model. The priority experiments are the local-plus-global ablation, equal-compute longer training, and quality evaluation at the long contexts used for timing. Until those pass, attribute results to the complete hybrid and describe 8k decode as approximately tied.
