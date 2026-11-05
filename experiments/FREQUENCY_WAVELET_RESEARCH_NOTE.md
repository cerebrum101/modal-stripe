# From Zipf skew to adaptive multiresolution memory

**Status:** exploratory research direction, 4 November 2026  
**Evidence:** one controlled synthetic screening experiment; three seeds for the principal comparisons  
**Do not describe this as a demonstrated architecture improvement.**

## The useful new idea

The next branch should not be “add wavelets” or “tell the model which tokens are rare.” Those are too broad and too close to existing token mixers and memory-retention work. The more specific hypothesis is:

> **A causal language model can spend a fixed memory budget adaptively: a cheap coarse spectral state carries diffuse, repeated context, while a small content-addressable detail memory protects surprising or low-frequency key/value events. The query reads detail memory by key match; token frequency controls write/retention priority, not the address used to retrieve a value.**

The physics analogy is adaptive mesh refinement / multiresolution methods. Smooth background fields can be represented coarsely; localized sharp features receive finer resolution. Here, the “sharp feature” is a high-information event or entity association. This analogy suggests *where to spend capacity*. It does not imply that natural-language tokens are literally a smooth physical field.

### Theoretical resource yield to test

For hidden width \(d\), sequence length \(T\), \(R\) modal states and \(M\) explicit detail slots:

- Full causal attention has \(O(T^2d)\) pairwise work and \(O(Td)\) retained KV state per layer at decode.
- The existing modal convolution has \(O(T\log T\,d)\) FFT work in the current implementation and \(O(Rd)\) recurrent state per layer at decode.
- A bounded slot read has \(O(TMd)\) work and \(O(Md)\) stored payload, independent of prompt length once the budget is full. A matrix-valued fast-weight memory instead keeps \(O(d_kd_v)\) state and has per-step feature/update work; its capacity and interference are the tradeoff.

Thus the plausible gain is a fixed memory ceiling and potentially cheaper decoding when \(M\ll T\), while preserving exact bindings in the retained slots. This is a conditional cost argument, not a measured speedup. If the useful facts are more numerous than \(M\), the method must evict something and quality can fall. The practical target should first be the existing hybrid's roughly half-sized cache at short contexts, then measure scaling across longer prompts. No throughput gain should be claimed until the whole implementation, including slot search/update, is timed on the target hardware.

Call this a **frequency-budgeted multiresolution associative memory** as a working name only. It is a hypothesis, not a novelty claim.

## Why the last modification did not work

Let the current modal memory be

\[
z_t=\sum_{j\le t}K_{t-j}\,u_j,
\qquad u_j=W_vh_j\odot\sigma(W_wh_j).
\]

This is a causal shared-kernel sum. The query-conditioned read gate can change how the aggregate is viewed, but there is no explicit operation that compares the current query key against each stored key *before* the values are collapsed together. A frequency scalar can make a write stronger or weaker; it cannot by itself restore key/value binding after unrelated values have been mixed. The observed result fits this mechanism-level limitation.

The minimal associative read has a different structure. For feature maps \(\phi\), maintain

\[
M_t=\sum_{j\le t}\phi(k_j)v_j^\top,
\qquad y_t=M_t^\top\phi(q_t).
\]

This reads by query/key compatibility. A delta-style update can subtract or overwrite an obsolete association before writing. The proposed spectral branch would be complementary: use the modal state for cheap diffuse context, and reserve a bounded number of explicit slots for precise details. Rare/surprising events influence slot admission and retention, while key similarity performs retrieval.

## What we tested

We created 16 key/value records per example. Every key appears once in every context, values are randomly remapped on each example, and the query distribution follows a Zipf prior with exponent 1.1. Training therefore isolates *query supervision imbalance* rather than making rare keys themselves absent from the context. Every model sees 900 updates, batch 32, sequence length 64. Evaluation reports natural Zipf-weighted accuracy and a separate balanced audit with 128 fresh random mappings and every key queried once for each mapping (2,048 balanced queries per seed).

| Model | Zipf micro accuracy | Balanced key accuracy | Common quartile | Rare quartile |
|---|---:|---:|---:|---:|
| Full attention | 78.0 ± 2.3% | 44.2 ± 3.0% | 98.4 ± 0.9% | 8.3 ± 1.6% |
| Spectral memory | 41.9 ± 2.8% | 13.9 ± 2.2% | 41.7 ± 6.8% | 4.3 ± 0.7% |
| Spectral + inverse-root loss weights | 6.3 ± 0.2% | 6.4 ± 0.2% | 5.4 ± 0.6% | 6.9 ± 0.8% |
| Spectral + rarity side channel | 42.6 ± 3.8% | 14.2 ± 2.0% | 41.8 ± 8.1% | 5.1 ± 1.3% |

Chance is 6.25%. Values are mean ± sample SD across three seeds, not confidence intervals. The combined weighted-plus-rarity branch and local attention were only screened at seed 0; they are omitted from this three-seed table. The principal result is not that attention is “solved”: even full attention has only 8.3% balanced recall on the rare quartile at this short training budget. Rather, the static spectral memory learns a stronger frequent-key shortcut, and the two simple frequency interventions do not repair it. Reweighting collapses to chance in all three seeds. Rarity conditioning changes balanced accuracy by only 0.3 percentage points on average versus plain spectral, with no visible rare-tail recovery.

Raw per-seed records, checkpoints and the screening code are in `experiments/results/zipf_*.json`, `experiments/results/zipf_*.pt`, and `experiments/zipf_wavelet_branch.py`. These are short CPU screening runs. They do not establish convergence, language-model quality, or hardware speed.

## Bounded tree search

```text
Goal: preserve distant exact recall with less growing KV storage
|
+-- 1. Change corpus/query frequency only
|   +-- scalar rarity feature -> tested; no rare-tail recovery (prune)
|   +-- inverse-frequency loss -> tested; chance performance (prune as default)
|
+-- 2. Change temporal basis only
|   +-- more Fourier modes -> existing 32-mode check did not rescue recall (deprioritize)
|   +-- wavelet token mixer alone -> already a studied token-mixing direction (not a novelty claim)
|
+-- 3. Add queryable content memory, keep spectral background
|   +-- unbounded growing key/value cache -> accurate reference, misses memory objective
|   +-- fixed associative state with delta update -> test; efficient candidate
|   +-- bounded explicit detail slots + coarse modal state -> lead experiment
|       +-- admission: contextual surprisal / entity novelty, calibrated with token counts
|       +-- retrieval: query-key match, never unigram frequency as the address
|       +-- eviction: learned utility under fixed byte budget; use frequency only as prior
|
+-- 4. If slot memory succeeds, ask if frequency allocation adds anything
    +-- same memory bytes, uniform admission vs surprisal admission vs Zipf-prior admission
    +-- test under stationary Zipf, shifted Zipf, and uniform query distributions
```

The next experiment should keep only the lead candidate and controls: full attention, the current spectral model, a fixed-budget associative memory (no spectral branch), and spectral + fixed-budget associative detail slots. Then ablate uniform vs surprisal/frequency-informed admission while holding slot count, hidden size, optimizer updates and inference budget fixed. Use the existing unique-value MQAR-style control, the Zipf-skew test here, a shifted-frequency test where the rank prior changes between train and evaluation, and one natural-text rare-entity/retrieval task. Report per-frequency-bin recall, macro recall, micro recall, bytes retained, and measured end-to-end latency.

**Falsification criterion:** if rarity-aware admission fails to improve rare-bin recall over uniform admission at the same memory budget on both stationary and shifted priors, remove corpus-frequency allocation from the method. If the associative detail memory fails to close the binding gap before the admission ablation, stop adding wavelet/FNO complexity; the bottleneck is the memory read/update rule.

## Relation to prior work and novelty boundary

- **Wavelet token mixing exists.** WNet replaces encoder attention with DWT-scale token mixers and studies gated scale fusion, but its reported setup is masked-language-model encoder pretraining and GLUE fine-tuning, not causal exact retrieval. Therefore “wavelet mixer for text” is occupied; our candidate is about queryable detail storage and frequency-budgeted admission, which still needs a direct literature search before any novelty assertion. [WNet](https://arxiv.org/abs/2610.04720)
- **Recall failure in efficient sequence models is established.** Zoology formalizes multi-query associative recall and finds a substantial recall gap for gated convolutions, then studies sparse-attention hybrids. Our benchmark is a small Zipf-skewed diagnostic, not a replacement for MQAR or a new general benchmark claim. [Zoology](https://arxiv.org/abs/2312.04927)
- **Targeted state updates exist.** Gated DeltaNet combines adaptive forgetting with delta-rule writes for associative memory; this is a direct baseline and a likely implementation guide, not a component we can relabel as novel. [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
- **Frequency-aware cache selection exists.** FASA uses dominant RoPE frequency chunks to predict salient tokens for query-aware KV eviction. Its “frequency” means positional RoPE bands, not unigram/Zipf frequency, but it makes broad frequency-aware retention claims unsafe. [FASA](https://arxiv.org/abs/2602.03152)

The possible contribution, if experiments support it, is the measured interaction between *corpus-frequency/surprisal-guided fixed-budget detail admission* and a causal spectral background under shifted Zipf priors. It is not yet established as new or useful. Compare explicitly against query-aware eviction and delta-rule memory before drafting a preprint around it.

## Research decision

There is now enough for a short **negative-results / hypothesis memo** and a concrete next experiment. There is not yet enough for a credible methods preprint. The discriminating result would be a repeatable rare-bin recall gain at matched memory bytes and latency, surviving shifted frequency priors and a natural retrieval task, without losing common-token language-model quality. Only then would it make sense to expand into a paper and make a careful novelty claim.
