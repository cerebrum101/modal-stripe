# Reproduce the first experiment

The audited follow-up is in [rigorous/README.md](rigorous/README.md), with the full controlled results in [rigorous/REPORT.md](rigorous/REPORT.md). Use that framework for new comparisons; the scripts below retain the original exploratory runs.

Run from the project root with native arm64 Python 3.13. The current environment is `.venv-arm/`; exact installed packages are in `requirements-lock.txt`.

```sh
.venv-arm/bin/python experiments/run.py check
.venv-arm/bin/python experiments/suite.py
.venv-arm/bin/python experiments/branches.py
.venv-arm/bin/python experiments/zipf_wavelet_branch.py --steps 900 --seeds 3
.venv-arm/bin/python experiments/corpus_headtohead.py --steps 600 --seeds 3
.venv-arm/bin/python experiments/run.py check
.venv-arm/bin/python experiments/run.py benchmark
.venv-arm/bin/python experiments/run.py decode
.venv-arm/bin/python experiments/diagnose.py
.venv-arm/bin/python experiments/summarize.py
```

Run these sequentially for isolated timing. The suite skips a run only when a saved result exists with its requested step count; it does not verify source hashes. To reproduce after code changes, use a separate copy of the experiment directory or move the affected saved run records aside first. Outputs use deterministic names and replace earlier files of the same configuration. Training checkpoints contain model weights, not resumable optimizer/RNG state.

`suite.py` runs four initial variants, three seeds, retrieval then text. `branches.py` follows the retrieval failure with 32-mode single-seed checks and a three-seed striped hybrid. Every text model sees 1,228,800 input positions and every retrieval model sees 1,536,000 input positions. The stem, embeddings, positional encoding, prediction head, optimizer schedule, and per-seed example streams are common. Approximately matched parameter counts are achieved by adjusting feed-forward width, so this is not a pure mixer-only ablation.

`zipf_wavelet_branch.py` is a separate 900-step screening experiment for Zipf-skewed key queries. It compares full/local attention and spectral variants with inverse-frequency loss weighting and an explicit rarity side channel. It reports both Zipf-weighted accuracy and an equally weighted per-key audit. See [FREQUENCY_WAVELET_RESEARCH_NOTE.md](FREQUENCY_WAVELET_RESEARCH_NOTE.md) for results and the proposed next branch. Its 900-step runs are not directly comparable to the longer original suite.

`corpus_headtohead.py` compares full attention with the striped model (one spectral + local block, one full-attention block) on Hamlet alone and on Hamlet + Alice + Moby-Dick. It uses equal training-token budgets, three seeds, held-out tail windows, and warmed prefill/decode measurements at multiple context lengths. See [results/TEXT_HEAD_TO_HEAD.md](results/TEXT_HEAD_TO_HEAD.md). It saves the public-domain text snapshots and per-seed checkpoints/results under `experiments/data/` and `experiments/results/`.

`THEORY.md` explains the expected resource benefits and limits. `results/REPORT.md` gives the findings, with underlying runs, trained weights, checks, diagnostics, timing measurements, source metadata, and illustrative samples in the same directory. Training wall times in run histories are not isolated benchmarks; use `benchmark.json` and `decode.json` for timing claims.

The source corpus is fetched only if missing. Its SHA-256 is recorded in every text result. The data is a character-level corpus; its perplexity cannot be compared directly to token-level results in language-model papers. No external account or paid compute is used.
