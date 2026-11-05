# Full attention vs. the striped spectral hybrid

**Audited follow-up:** [rigorous results](../rigorous/REPORT.md) add parameter controls, tuned learning rates, matched-time comparisons, unseen books, paired timing, and corrected long-context diagnostics. This file preserves the earlier stage and should be read alongside that follow-up.

**Question:** on ordinary next-character prediction, how much quality do we give up for the spectral/attention “Frankenstein,” and does it actually run faster?  
**Answer from this prototype:** no observed quality penalty in these short tests; about half the KV cache; slower training and short-context decode; long-context prefill improves, and decode is roughly even by 8k characters. This is a promising systems/blog result, not evidence that the model is generally better or faster.

## Models

- **Full:** two causal global-attention blocks.
- **Striped:** one block combines the causal modal spectral memory with window-8 local attention; the other is a full global-attention block. This is the branch that retained exact retrieval in the earlier diagnostic.

Both are two layers, width 64, and have nearly equal parameter counts (114,752 full; 114,641 striped, a 0.1% difference). The text model is character-level with a shared 81-character vocabulary normalized from the three sources. Each comparison uses the same seed, batch stream, optimizer, training-token budget, and 600 updates: 1,228,800 input characters. Results are over three seeds; `±` is sample standard deviation, not a confidence interval.

## Text prediction quality

We split each source contiguously: first 90% for training and final 10% for validation. In the mixed experiment, windows are sampled from Hamlet, *Alice’s Adventures in Wonderland*, and *Moby-Dick* in proportion to the number of available training windows. We score 102,400 randomly sampled held-out next-character targets per source and seed. For Hamlet alone this revisits overlapping positions because the held-out tail is only 17,810 characters; treat the small-data estimates accordingly.

| Training data | Validation text | Full attention PPL | Striped PPL | Striped change |
|---|---|---:|---:|---:|
| Hamlet only | Hamlet | 7.425 ± 0.129 | 6.754 ± 0.072 | 9.0% lower |
| Mixed 3 works | Hamlet | 9.105 ± 0.097 | 8.448 ± 0.039 | 7.2% lower |
| Mixed 3 works | Alice | 8.236 ± 0.067 | 7.507 ± 0.034 | 8.8% lower |
| Mixed 3 works | Moby-Dick | 8.420 ± 0.039 | 7.641 ± 0.024 | 9.3% lower |

The existing Tiny Shakespeare experiment points the same way after its 600-step budget: full attention 8.36 ± 0.01 vs. striped (reported there as “hybrid + global block”) 7.70 ± 0.04 character perplexity. These are tiny character models trained briefly. Lower character perplexity is a next-character prediction result, not a direct measure of human judgments, semantic ability, or token-level LLM performance.

## Speed and memory

Trained checkpoints are evaluated with a 32-character continuation, after a decode warmup. Each timing is the mean ± sample SD of the five-repeat medians across the two training corpora and three seeds (six checkpoints per model); these spreads are not confidence intervals. Prefill includes cache construction. The host is the project’s Apple M1 CPU; these are unfused PyTorch reference paths.

| Prefix length (characters) | Full prefill (ms) | Striped prefill (ms) | Full decode (ms/char) | Striped decode (ms/char) | Full cache (KiB) | Striped cache (KiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4.86 ± 0.67 | 6.06 ± 0.53 | 0.365 ± 0.030 | 0.433 ± 0.025 | 544 | 280 |
| 4,096 | 86.08 ± 1.33 | 65.97 ± 2.27 | 0.431 ± 0.007 | 0.472 ± 0.015 | 4,128 | 2,072 |
| 8,192 | 322.26 ± 33.68 | 209.31 ± 19.30 | 0.618 ± 0.109 | 0.594 ± 0.078 | 8,224 | 4,120 |

At 512, the striped model uses 49% less cache but prefill is 25% slower and decode is 18% slower. By 4,096, prefill is 23% faster and cache remains 50% smaller, while per-character decode is still 10% slower. At 8,192, prefill is 35% faster, decode is approximately tied within measurement spread, and cache is 50% smaller. The benchmark therefore has a context-length crossover: at small contexts the extra spectral/local work is overhead; at long prompts it reduces the cost of prefill and retained KV storage. It does not show a large decode-throughput win. Quality was tested on 128-character windows; the 4k–8k prefixes measure resource use only, not long-context prediction quality.

The measured 600-step training wall times also favor full attention: 19.2 ± 5.9 s vs. 23.2 ± 4.0 s on Hamlet, and 19.2 ± 7.1 s vs. 21.1 ± 3.3 s on the mixed corpus. That timing has noticeable host variance; use it as a rough sign that the current implementation is not a training-speed win. A separate length-1,024 batch-4 microbenchmark in the original report found 61.91 ms/step full vs. 65.44 ms/step striped.

Cache payload counts include retained K/V or recurrent state tensors and the shared previous-token state. They exclude weights, allocator overhead, temporary workspaces, and runtime. These small models do not predict GPU behavior or production kernels.

## Does this merit a blog post?

**Yes, if framed as an honest prototype tradeoff.** The clearest claim supported here is: *on three small character-level text sources, the striped model matched or improved held-out next-character perplexity after the same number of training characters, cut long-context cache payload about in half, improved prefill at 4k–8k characters, and did not improve training speed; decode only caught up at the longest tested prefix.*

That is a credible engineering blog because it answers the practical question with measurements and shows where the hybrid helps and hurts. Do not title it as a new state-of-the-art language model or a generally faster attention replacement. The next step for a stronger post is a GPU benchmark with optimized kernels and a larger tokenizer/model, plus longer training to see whether the short-run perplexity gap survives convergence.

## Reproduction and data

Run from the project root:

```sh
.venv-arm/bin/python experiments/corpus_headtohead.py --steps 600 --seeds 3
```

The script saves normalized public-domain texts in `experiments/data/` and per-seed metrics, checkpoints, prefill/decode times, and cache bytes in `experiments/results/headtohead_*.json` and `.pt`. Data comes from Project Gutenberg: [Hamlet, ebook 1524](https://www.gutenberg.org/ebooks/1524), [Alice, ebook 11](https://www.gutenberg.org/ebooks/11), and [Moby-Dick, ebook 2701](https://www.gutenberg.org/ebooks/2701). The exact downloaded raw hashes and normalization rule are stored in each result record.
