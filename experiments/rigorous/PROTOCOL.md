# Frozen small-model study protocol

Prepared before the new training and final-book evaluation. This is a bounded CPU/CUDA prototype study, not a large-model scaling study.

## Comparisons

Three complete architectures: two global-attention blocks (`full`); window-8 local attention followed by global attention (`localglobal`); spectral + window-8 local attention followed by global attention (`striped`). Two budget controls: approximately matched total parameters, and identical 256-wide feed-forward layers. The latter permits parameter counts to differ and isolates feed-forward allocation. All use width64, two blocks, the same two-token causal stem, fixed ASCII128 vocabulary, optimizer, batches and data streams.

## Data and selection

Train on first80% of each original text (Hamlet/Alice/Moby Dick); development uses 80–90%. These sources have already informed the architecture and are development sources. Corrected uniform valid-window sampling never crosses document or split boundaries. Character normalization is recorded in code; source snapshots and hashes are retained.

For each architecture, budget and training context, select learning rate from 0.0005/0.001/0.002 using a 400-step seed0 pilot. Score is equally weighted development cross-entropy over the three texts. The same tuning budget applies to each architecture. Pilots cannot establish a robust universal optimal rate.

Main runs use seeds0/1/2 and 1800 updates at 2048 characters per update, with quality checkpoints300/600/1200/1800. Training contexts128 and512 use batches16 and4. Fixed-feed-forward control uses context128; parameter-matched controls use both contexts. Models at different checkpoints share one 1800-update cosine schedule; compare like checkpoints. Do not call 1800 updates convergence.

Full attention additionally trains for the striped model's measured accumulated training-step wall time for each seed at context128. Its learning-rate schedule decays by elapsed time; actual steps, characters and time are reported. This is a practical wall-time comparison subject to host load, not a FLOP-matched proof.

Freeze all selected configurations before downloading final books: Frankenstein (PG84), Sherlock Holmes (PG1661). Never use final-book outcomes for selection. Evaluate every held-out character once with reset128-character windows. Also evaluate identical target blocks with prefix lengths128/512/2048/8192, reporting context extrapolation explicitly. These final tests measure transfer to unseen works; they are not within-source final tails. Eight target blocks per book limit inference and are descriptive, not a broad long-context benchmark.

## Reliability

Independent sampler alignment/last-window, future-token perturbation, future-gradient, cached-vs-full equivalence including cache growth, dense-reference local attention, and spectral finite-difference gradient checks run before training. Full source code, configuration, source hashes, device/library versions and checkpoint hashes identify each completed run. Writes are atomic for JSON; incomplete runs restart, completed runs are reused only for the identical configuration/code/data hash. Store immutable source copies with runs. Seeds describe optimization spread; do not treat overlapping characters or seeds as independent datasets.

Record all pilot outcomes, selected rates, full learning curves and parameter counts. No claim that spectral memory helps is allowed unless striped beats localglobal under these controls. No speed/quality claim at a long context without reporting quality at that context. No GPU claim from CPU measurements. Timing must save raw repeats and randomize/interleave architecture order after warmup, synchronizing CUDA when applicable. Tensor cache payload is separate from process/device peak memory.
