# Reproducible prototype audit

Read PROTOCOL.md before interpreting results. Original experiment files are preserved; model.py freezes the prior architecture implementation.

From the project root:

```
.venv-arm/bin/python experiments/rigorous/study.py --checks-only
.venv-arm/bin/python experiments/rigorous/study.py
.venv-arm/bin/python experiments/rigorous/measure.py
```

Python requires PyTorch and NumPy (already installed locally). The same scripts choose CUDA automatically if available, otherwise CPU. They do not use the Apple MPS backend. On a Colab GPU runtime, upload this folder, install PyTorch if necessary, then run `python study.py` and `python measure.py`. Performance results must be labeled by the recorded backend; do not merge CPU and GPU timing.

No paid compute is requested automatically. Study JSONs, checkpoints and frozen source copies live under content-derived `runs/` directories. Completed runs are verified and reused only for identical code/data/configurations. Incomplete runs restart. Keep these directories to reproduce the report. Model training is deterministic where the backend permits; device/library details are recorded. This does not promise bitwise reproducibility across platforms.

The final books are downloaded after selection is frozen. Do not tune using their outcomes. If the study script changes, the hash changes and training creates new run directories. measure.py checks all trained checkpoints then saves every randomized paired timing repetition, rather than overwriting measurements with a hand-picked best number.

Additional checks and final reporting:

```
.venv-arm/bin/python experiments/rigorous/test_framework.py
.venv-arm/bin/python experiments/rigorous/audit_data.py
.venv-arm/bin/python experiments/rigorous/report.py
```

Run audit_data.py after study.py finishes. It checks sampled nonoverlapping final-book blocks for exact matches anywhere in the training texts; it is a limited overlap audit, not a semantic deduplication guarantee. measure.py also verifies cached-vs-full logits at 8192-prefix length on every trained checkpoint. Its raw timing file records each timed checkpoint's identity and the measurement script hash. GPU synchronization is implemented; process peak memory is not measured. Tensor cache payload is the memory metric.

The host-load issue discovered during training is addressed by a pre-specified supplemental comparison in CALIBRATION_PLAN.md. After measure.py, run `python experiments/rigorous/calibrate.py` from the project root (or `python calibrate.py` inside this directory), then generate the report. It selects a full-attention update budget using only the paired training timing, not final-book scores. It preserves the primary final evaluation file and adds its own configurations, checks and results. All phases remain visible in REPORT.md.

Recommended full pipeline: `.venv-arm/bin/python experiments/rigorous/run_all.py`. It archives existing evidence before rerunning and archives the final evidence afterward. Training directories are content-derived and verified; timing and summary files are regenerated. The archives include source snapshots, input texts, raw JSONs and checkpoints with a SHA-256 manifest, and exclude transient logs/cache files. Individual scripts can replace timing/summary files, so use the full pipeline when preserving an existing study.

The local-only control is explicitly exploratory and was added after the primary results were inspected; see LOCAL_CONTROL_PLAN.md. exact_window_audit.py corrects an interpretation problem in the original continuation test: it predicts one fixed target from exactly the named number of preceding characters. It uses seed0 and64 targets/book as a diagnostic; do not call that a broad long-context benchmark. The original prefix-plus-continuation results remain visible.
