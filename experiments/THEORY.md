# Expected yield before interpreting training results

This note distinguishes mathematical properties, resource estimates, and empirical questions. It accompanies the executable experiments; it is not a quality prediction.

## 1. Causal operator and equivalent execution

For tokenwise projected value v, write a, and read b:

    u_t = sigmoid(a_t) * v_t
    z_t,d = Σ_{j≤t} K_{t-j,d} u_j,d
    out_t = W_out(sigmoid(b_t) * z_t)
    K_l,d = Re Σ_r c_r,d p_r^l, |p_r| < 1

The same operator has recurrence s_t,r,d = p_r s_(t-1),r,d + u_t,d, followed by z_t,d = Re Σ_r c_r,d s_t,r,d. Expanding the recurrence gives the convolution directly for zero initial state. Therefore a zero-padded FFT computes the parallel form while recurrence computes sequential decoding. Tokenwise gates preserve this equivalence. Arbitrary input-dependent poles would destroy the stationary convolution equivalence.

The implementation uses real sine/cosine kernel construction for training and complex states at decode. It does not truncate frequency bins. An identity residual preserves a direct token path; local attention supplies short-range interactions. The implementation is an operator-learning-inspired gated convolution / state-space hybrid, not an unmodified FNO.

## 2. Memory yield

At equal scalar precision, standard multi-head attention caches about 2LD real scalars per layer for keys and values (ignoring allocator overhead and GQA compression). A modal bank with R complex states per channel uses 2RD real scalars; local attention with window w caches at most 2wD. Thus:

    hybrid cache / full KV cache ≈ (R + w) / L

For D=64, R=8, w=8, L=1024: full KV is 131,072 float32 scalars (512 KiB) per layer; hybrid state plus local KV is 2,048 float32-equivalent scalars (8 KiB), a 64-fold reduction in this narrow component. The shared two-token stem additionally caches D scalars once per model in all variants. This is not a 64-fold reduction in total process memory; weights, activations, framework runtime, and workspaces remain. GQA and lower-precision KV would change the comparison.

For a hypothetical D=4096, R=32, w=128, L=32768, the corresponding ratio is 204.8. This is a scenario, not a measured large-model result.

## 3. Compute yield and break-even

Full attention mixing costs O(L²D) arithmetic. The proposed hybrid mixing costs O(LD log L + LwD), plus O(LRD) for the straightforward modal kernel construction. Both model families still have O(LD²) projections and feed-forward computation. The spectral branch adds projections and gates; FFTs incur launches, intermediate complex tensors, and memory traffic. Small L can therefore favor full attention.

The benchmark must include backward, kernel rebuilding, gradient clipping and optimizer updates, not merely an isolated FFT. Parameter matching approximately matches dense arithmetic in this small implementation but reallocates capacity: the hybrid has a narrower feed-forward layer to pay for its second mixer. Equal parameters do not guarantee equal function class or equal runtime.

During cached generation, full attention mixing grows with context, while modal updates are O(RD) and local attention O(wD). Actual implementation constants, cache movement, and projections decide the measured crossover. Use a preallocated full-attention cache to avoid charging the baseline for concatenating the entire prefix at every step. The implementation reserves 32 extra positions during prefill and grows geometrically only if needed; the 32-token benchmark does not trigger growth. No accelerator-speed conclusion follows from CPU measurements.

## 4. What cannot be free

At fixed finite precision the recurrent state has finitely many distinguishable configurations. It cannot losslessly encode an arbitrary number of independent key/value assignments. With 2RD real state slots at b stored bits per slot, a single modal state bank has at most b·2RD stored bits before accounting for numerical reliability. Across layers, states and the local cache contribute additional capacity. This is a storage upper bound, not a usable-capacity or retrieval guarantee.

Stationary modes also share the same decay/oscillation laws across contexts. Writes and reads are content-dependent, but pairwise coefficients factor as b_t,d K_(t-j),d a_j,d. This is more restricted than freely computed query-key interactions. Increasing modes may improve recall but raises state and kernel construction costs. If input-dependent forgetting is essential, test a selective recurrence, accepting that a single shared FFT no longer applies.

## 5. Falsifiable predictions

1. Increasing L eventually favors spectral mixing asymptotically, but the measured end-to-end crossover may be outside the tested range.
2. The local-only baseline cannot solve a fresh random key/value query whose record lies outside its total receptive field. With two blocks, window 8, and the shared previous-token stem, a position sees at most 15 earlier offsets. The earliest possible queried value in the default task is 40 tokens before the final position. Population accuracy is therefore 1/16; finite evaluation samples fluctuate.
3. Spectral/hybrid success on that task demonstrates information traveling outside local receptive fields, but not unlimited memory or general language reasoning.
4. Good character prediction can coexist with weak exact retrieval. Evaluate both; do not infer one from the other.
5. Longer-length and more-pair recall tests can expose reliance on fixed training positions and state interference. Failure there limits extrapolation even if in-distribution accuracy is high.

## 6. Practical protocol

- CPU, native arm64 Python, float32. The host has 8 GB RAM. Use two PyTorch threads and sequential jobs; record installed versions in results.
- Four variants: full, local, spectral, hybrid. Two blocks, width 64, eight modes, local window eight. Per-block parameter budget is approximately 50,000. All variants share sinusoidal positions, a causal previous-token stem, normalization and the prediction head.
- Approximately parameter-matched, equal training tokens, identical per-seed training batches and final evaluation batches. Run seeds 0, 1, 2. This is an equal-token comparison, not an equal-FLOP or equal-time training comparison. Separate isolated benchmarks measure cost; a compute-matched learning comparison remains future work.
- Retrieval: random distinct keys, random values with 16 possibilities; four adjacent key/value records at the beginning, distractor tokens in the middle, and a key query at the end. Length 48, batch 32, 1,000 updates after the shared-stem control succeeded. Each example has a fresh mapping. Final test: 1,600 independent generated examples per seed. Transfer checks vary pair count or distance. Loss is applied only to the answer; this is a retrieval diagnostic, not language-model training. A later all-query control uses unique values to separate binding from query-blind guessing.
- Text: Tiny Shakespeare, character-level next-token objective, first 90% training and last 10% validation, independent windows within each split, context 128, batch 16. Vocabulary is defined from the full corpus; only the character inventory is exposed by that choice. File hash and source URL are saved. Character perplexity is not comparable to word/subword perplexity from large-model papers. Repeated random validation windows overlap; their token count is not an independent-sample count.
- Three seeds quantify observed training variability; they do not establish a broad benchmark claim. No hyperparameter search is declared. A preliminary retrieval baseline without the shared stem failed to learn binding at 3,000 steps (~29% accuracy versus 6.25% uniform chance), so its artifact is retained and the common stem is tested before accepting the diagnostic. A failed attention control prevents a strong negative conclusion about spectral retrieval.
- The initial acceptance targets from REALIZATION.md (3% perplexity gap, 5 percentage-point recall gap, 20% measured utility improvement) remain screening targets. Passing a tiny task cannot establish those targets at useful language-model scale.
