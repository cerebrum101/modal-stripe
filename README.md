# ModalStripe

Modal memory followed by local attention in the first block of a small character language model, with global attention retained in the second block.

**Author:** Atay Kim · Nazarbayev University · atay.kim@nu.edu.kz

[Read the paper](modal-stripe.pdf) · [Single-file LaTeX](main.tex) · [Equations](docs/FULL_EQUATIONS.md) · [Protocol](experiments/rigorous/PROTOCOL.md) · [Audited results](experiments/rigorous/REPORT.md)

## What the experiment shows

At approximately 121k parameters, context-128 training and an equal character budget, final-book character perplexity is 6.047 for full attention, 5.964 for local + global attention, and 5.592 for ModalStripe. Giving attention the timing-calibrated 2,439 updates improves it to 5.701. Literal elapsed-time matching reverses the ranking, but changing host load confounds that comparison.

These are tiny character models, three optimization seeds and two final books, measured on an Apple M1 CPU. The evidence does not establish GPU speed, large-model scaling or reliable long-context retrieval. Most cache reduction also occurs in the local-plus-global control. Local-only and corrected context diagnostics are exploratory. The complete hybrid retains a growing global KV cache and quadratic full-sequence attention work.

## Contents

- `main.tex`: self-contained paper, with all four current figures, tables, references and equations embedded. No external figure, bibliography or table files are needed to compile it.
- `modal-stripe.pdf`: compiled v0.6 paper.
- `experiments/rigorous/`: controlled study, raw timing/evaluation records, 66 completed training runs with checkpoints and frozen source snapshots, and five original corpus snapshots. There are 36 final evaluated checkpoints; the other runs are tuning pilots.
- `experiments/`: earlier exploration, including recall, Zipf/frequency and head-to-head runs. These are historical experiments, not interchangeable with the controlled study or evidence for every paper claim.
- `figures/`: reviewed schematics, PGFPlots sources, and numerical plot generation.
- `docs/`: full equations, concise diagram key and the internal conceptual review behind the schematic redesign.
- `paper/`: editable manuscript template, table generation and single-file assembly.
- `validation/`: release integrity and operator checks.

Rejected decorative images, old paper drafts, duplicated evidence ZIPs and transient output are excluded.

## Compile the paper

With Tectonic installed, from the repository root:

```sh
mkdir -p output
tectonic -o output main.tex
```

The result is `output/main.pdf`. `main.tex` can also be copied into an otherwise empty folder and compiled there. It uses standard LaTeX packages including TikZ and PGFPlots; Tectonic may fetch missing packages on its first run. LaTeX math and Helvetica schematic text are rendered directly, without generated illustrations.

To regenerate the standalone source after editing the modular sources:

```sh
python paper/assemble.py
```

## Verify the saved release

```sh
python validation/verify_release.py
```

This standard-library check verifies the release manifest, saved checkpoints, data/source identities, primary record consistency and the single-file manuscript. It does not train, download data, replace measurements or require PyTorch.

For model and framework checks, use Python 3.13 with the measured dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/rigorous/study.py --checks-only
python experiments/rigorous/test_framework.py
python experiments/rigorous/verify_results.py
python validation/verify_math.py
```

The exact measured environment is retained in `experiments/requirements-lock.txt` and each run's provenance. Checkpoints store trained weights, not resumable optimizer/RNG state. Cross-platform bitwise reproduction is not promised.

## Repeat experiments

Read the protocol before running training. The scripts use CUDA when available, otherwise CPU; the archived paper results are CPU measurements. Run a copied checkout when comparing new hardware so the original measurements remain available.

```sh
python experiments/rigorous/run_all.py
```

This archives existing evidence, verifies/reuses compatible completed runs, trains missing conditions and regenerates timing and summary records. It can take substantial time. Legacy commands and their limitations are documented in `experiments/README.md`; replace its historical `.venv-arm/bin/python` path with your own Python executable.

## Data and attribution

The corpus files retain the original Project Gutenberg text and license notices. See [DATA.md](DATA.md) for source identifiers and uses. Existing code provenance and inactive parameter accounting are documented in the paper. No software license has been selected for this release; the source is published for inspection and a license can be added by the author.

OpenAI Codex assisted with implementation, audits, figure production and writing. The internal design review is not external peer review or an endorsement by Yann LeCun. This repository is a small-model research artifact, not an arXiv acceptance or publication announcement.
