"""Aggregate actual saved runs and render reproducible scientific figures."""
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
ORDER = ['full', 'local', 'spectral', 'hybrid', 'striped']
LABELS = {'full': 'Full attention', 'local': 'Local attention', 'spectral': 'Spectral memory',
          'hybrid': 'Spectral + local', 'striped': 'Hybrid + global block'}
COLORS = dict(zip(ORDER, ['#111111', '#969690', '#73736D', '#404040', '#1F6F5F']))
STYLES = dict(zip(ORDER, ['-', ':', '--', '-.', '-']))
MARKERS = dict(zip(ORDER, ['o', 's', '^', 'v', 'D']))


def stats(values):
    return {'mean': statistics.mean(values), 'sd': statistics.stdev(values) if len(values)>1 else None,
            'values': values, 'n': len(values)}


records = [json.loads(p.read_text()) for p in OUT.glob('*_s*_r8.json')]
summary = {}
for task in ['recall', 'text']:
    summary[task] = {}
    for kind in ORDER:
        runs = sorted([r for r in records if r['task'] == task and r['kind'] == kind], key=lambda r: r['seed'])
        assert len(runs) == 3 and [r['seed'] for r in runs] == [0, 1, 2], (task, kind)
        assert all(r['steps'] == (1000 if task == 'recall' else 600) for r in runs)
        assert len({r['training_tokens'] for r in runs}) == 1
        summary[task][kind] = {metric: stats([r['final'][metric] for r in runs])
                               for metric in ['accuracy', 'perplexity', 'loss_nats']}
        summary[task][kind]['parameters'] = runs[0]['parameters']
        summary[task][kind]['initial_loss'] = stats([r['initial']['loss_nats'] for r in runs])
        summary[task][kind]['training_tokens'] = runs[0]['training_tokens']
        if task == 'recall':
            summary[task][kind]['transfer'] = {condition: stats([r['transfer'][condition]['accuracy'] for r in runs])
                                              for condition in runs[0]['transfer']}
bench = json.loads((OUT/'benchmark.json').read_text())
decode = json.loads((OUT/'decode.json').read_text())
diagnostics = json.loads((OUT/'diagnostics.json').read_text())
summary['diagnostics'] = {kind: {
    metric: stats([r[metric] for r in diagnostics if r['kind'] == kind and r['modes'] == 8])
    for metric in ['unique_value_all_query_accuracy', 'contexts_with_query_dependent_predictions']}
    for kind in ORDER}
summary['mode_branches'] = [json.loads(p.read_text()) for p in OUT.glob('recall_*_s0_r32.json')]
(OUT/'summary.json').write_text(json.dumps(summary, indent=2))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#969690', 'axes.linewidth': .7,
                     'text.color': '#111111', 'axes.labelcolor': '#404040',
                     'xtick.color': '#404040', 'ytick.color': '#404040',
                     'svg.fonttype': 'none', 'figure.facecolor': 'white'})
spec = {'claim': 'Compare retrieval, character prediction, and measured CPU costs without inferring large-model performance',
        'source': 'results/*json, summarized in summary.json',
        'visual_form': 'quality: paired horizontal dot plots; costs: ordered line plots',
        'order': ORDER, 'uncertainty': 'show all three training seeds; timing min-max over repeated measurements',
        'dimensions_inches': [12, 5.6], 'formats': ['svg', 'png'],
        'references': ['OAI-VIS-002', 'OAI-VIS-011', 'OAI-VIS-012', 'OAI-VIS-004', 'OAI-VIS-095'],
        'precision': 'accuracy 1 decimal percent; character perplexity 2 decimals; latency 3 significant figures'}
(OUT/'visual-spec.json').write_text(json.dumps(spec, indent=2))


def export(fig, name):
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png', dpi=200)
    # Separate internal grayscale QA preview; never replaces the color artifact.
    image = Image.open(OUT/f'{name}.png')
    ImageOps.grayscale(image).save(OUT/f'{name}_grayscale.png')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
fig.subplots_adjust(left=.18, right=.96, top=.81, bottom=.28, wspace=.62)
for axis, task, metric, scale, title, xlabel in [
    (axes[0], 'recall', 'accuracy', 100, 'Global block restores recall', 'Recall accuracy (%) · higher is better'),
    (axes[1], 'text', 'perplexity', 1, 'Held-out character prediction', 'Character perplexity · lower is better')]:
    for i, kind in enumerate(ORDER):
        values = summary[task][kind][metric]['values']
        axis.scatter([v*scale for v in values], i+np.linspace(-.08, .08, len(values)),
                     color=COLORS[kind], marker=MARKERS[kind], s=28, zorder=3)
        mean = statistics.mean(values)*scale
        axis.scatter([mean], [i], color=COLORS[kind], marker='|', s=160, linewidth=2, zorder=4)
    axis.set_yticks(range(len(ORDER)), [LABELS[k] for k in ORDER])
    axis.invert_yaxis()
    axis.set_title(title, loc='left', pad=14)
    axis.set_xlabel(xlabel, labelpad=12)
    axis.set_ylim(4.55, -.55)
    if task == 'recall':
        axis.set_xlim(-3, 105)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.axvline(6.25, color='#B6B6B0', linestyle=':', linewidth=1)
    else:
        axis.tick_params(axis='y', labelleft=False)
        axis.set_yticks(range(len(ORDER)))
fig.text(.18, .96, 'Small models: retrieval and text prediction measure different abilities', fontsize=14, weight='semibold', va='top')
fig.text(.18, .035, 'Three seeds shown individually; vertical ticks mark means. Approximately 110K parameters; equal tokens within each task.\nRecall: 1,600 fresh examples per seed. Text: Tiny Shakespeare held-out split, context 128; short 600-step training.',
         fontsize=10, color='#5E6462')
export(fig, 'quality')

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
fig.subplots_adjust(left=.085, right=.97, top=.80, bottom=.35, wspace=.28)
for kind in ORDER:
    training = [r for r in bench['rows'] if r['kind'] == kind]
    decoding = [r for r in decode['rows'] if r['kind'] == kind]
    for axis, data, xkey, ykey, low, high in [
        (axes[0], training, 'length', 'median_ms', 'min_ms', 'max_ms'),
        (axes[1], decoding, 'context', 'decode_median_ms_per_token', 'decode_min_ms_per_token', 'decode_max_ms_per_token')]:
        xs = [r[xkey] for r in data]
        ys = [r[ykey] for r in data]
        axis.plot(xs, ys, STYLES[kind], color=COLORS[kind], marker=MARKERS[kind],
                  label=LABELS[kind], linewidth=1.8 if kind=='striped' else 1.2, markersize=4)
        axis.fill_between(xs, [r[low] for r in data], [r[high] for r in data], color=COLORS[kind], alpha=.08)
axes[0].set_title('Complete training step', loc='left', pad=12)
axes[0].set_xlabel('Sequence length (tokens)')
axes[0].set_ylabel('Median step time (ms) · lower is better')
axes[1].set_title('Cached generation', loc='left', pad=12)
axes[1].set_xlabel('Prefix length (tokens)')
axes[1].set_ylabel('Median ms/token · lower is better')
for axis in axes:
    axis.set_ylim(bottom=0)
    axis.set_xscale('log', base=2)
axes[0].set_xticks([64, 128, 256, 512, 1024], ['64', '128', '256', '512', '1,024'])
axes[1].set_xticks([64, 256, 1024, 4096], ['64', '256', '1,024', '4,096'])
fig.text(.085, .96, 'CPU timing: cheaper asymptotics do not guarantee faster execution', fontsize=14, weight='semibold', va='top')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .13), ncol=3, frameon=False, fontsize=10)
fig.text(.085, .025, 'Apple M1 CPU · 2 threads · float32. Training: batch 4, 10 repeats; decode: batch 1, 32 tokens, 5 repeats.\nShading shows observed repeat min–max, not confidence intervals. Full-attention KV caches are preallocated; x-axes use log₂ spacing.',
         fontsize=9, color='#5E6462')
export(fig, 'timing')


def fmt(task, kind, metric, scale=1, digits=2):
    value = summary[task][kind][metric]
    return f"{value['mean']*scale:.{digits}f} ± {value['sd']*scale:.{digits}f}"


table = ['| Model | Parameters (text) | Recall (%) ↑ | Character perplexity ↓ |',
         '| --- | ---: | ---: | ---: |']
for kind in ORDER:
    table.append(f"| {LABELS[kind]} | {summary['text'][kind]['parameters']:,} | {fmt('recall', kind, 'accuracy', 100, 1)} | {fmt('text', kind, 'perplexity')} |")
timing_table = ['| Model | Training step at L=1,024 (ms) ↓ | Decode at prefix 4,096 (ms/token) ↓ | Cache at prefix 4,096 (KiB) ↓ |',
                '| --- | ---: | ---: | ---: |']
for kind in ORDER:
    train = next(r for r in bench['rows'] if r['kind']==kind and r['length']==1024)
    dec = next(r for r in decode['rows'] if r['kind']==kind and r['context']==4096)
    timing_table.append(f"| {LABELS[kind]} | {train['median_ms']:.2f} | {dec['decode_median_ms_per_token']:.3f} | {dec['cache_bytes_before_decode']/1024:.2f} |")
control_table = ['| Model | Unique-value recall (%) ↑ | Contexts where changing query changes prediction (%) |',
                 '| --- | ---: | ---: |']
for kind in ORDER:
    diag = summary['diagnostics'][kind]
    control_table.append(f"| {LABELS[kind]} | {diag['unique_value_all_query_accuracy']['mean']*100:.1f} | {diag['contexts_with_query_dependent_predictions']['mean']*100:.1f} |")
transfer_table = ['| Model | 2 pairs, L=48 (%) | 8 pairs, L=48 (%) | 4 pairs, L=96 (%) | 4 pairs, L=192 (%) |',
                  '| --- | ---: | ---: | ---: | ---: |']
for kind in ORDER:
    conditions = ['L48_pairs2','L48_pairs8','L96_pairs4','L192_pairs4']
    values = [summary['recall'][kind]['transfer'][c]['mean']*100 for c in conditions]
    transfer_table.append('| '+LABELS[kind]+' | '+' | '.join(f'{v:.1f}' for v in values)+' |')
mode_table = ['| Exploratory branch (seed 0 only) | Modes | Recall (%) |', '| --- | ---: | ---: |']
for record in summary['mode_branches']:
    mode_table.append(f"| {LABELS[record['kind']]} | {record['modes']} | {record['final']['accuracy']*100:.1f} |")
full_ppl = summary['text']['full']['perplexity']['mean']
gaps = {k: (summary['text'][k]['perplexity']['mean']/full_ppl-1)*100 for k in ORDER}
data = next(r['data'] for r in records if r['task']=='text')
checks = json.loads((OUT/'checks.json').read_text())
report = f'''# First training and hardware results

## Decision within this experiment

Pure spectral variants learned short-context character prediction but failed key/value binding: their unique-value recall stayed near the 25% query-blind baseline. Increasing modes from eight to 32 did not fix the original retrieval task in the exploratory runs. Keeping one global-attention block restored 100% default recall and roughly halved retained decode-cache storage, but did not beat full attention on the measured training-step or per-token decode times. This is a useful memory tradeoff, not a demonstrated throughput win at matched retrieval quality.

The spectral operation is trainable and causal. The two-block striped model contains one spectral-plus-local block and one full-attention block; it retains quadratic attention in half its blocks. Its result does not show that the modal memory learned exact retrieval.

## From theoretical yield to execution

[THEORY.md](../THEORY.md) derives the causal FFT/recurrence equivalence, fixed-cache benefit, compute terms, and finite-state limitations. The global branch uses damped modes without hard low-pass truncation. Actual experiments use approximately 110K parameters, two blocks, width 64, eight modes, and a local window of eight. A causal previous-token stem is shared by every model.

The baseline without that stem did not learn binding at 3,000 steps. That pilot is retained in `pilot_without_stem/`. The shared stem then let the attention control solve the diagnostic in 1,000 steps. We fixed that common stem and 1,000-step budget before comparing the other variants. This is protocol development on a synthetic diagnostic, not a pre-registered benchmark.

## Prediction and retrieval

{chr(10).join(table)}

Values are means ± sample standard deviation over seeds 0, 1, 2. These are seed spreads, not confidence intervals. Each model sees identical training data per seed: 1,536,000 input tokens for retrieval and 1,228,800 input tokens for text. The retrieval loss supervises one answer per example; the text loss supervises every next character. Parameter counts differ by less than 0.3%; the hybrid pays for its second mixer with a narrower feed-forward layer.

![Quality comparison](quality.png)

Relative mean character-perplexity changes versus full attention: {', '.join(f"{LABELS[k]} {gaps[k]:+.1f}%" for k in ORDER if k!='full')}. These are short training runs on a single small corpus. The learning curves have not established convergence or scaling behavior. They cannot establish a general language-model advantage.

## Is recall actually query-dependent?

Uniform value guessing scores 6.25%. A query-blind oracle that knows the four stored values and guesses the most frequent one averages 33.7158% on the original task. Therefore roughly 34% original-task accuracy is not evidence of binding keys to values.

The stronger control uses four distinct stored values and asks all four keys for each fixed context. A query-blind prediction then scores exactly 25%. Each seed uses 512 contexts and 2,048 queries.

{chr(10).join(control_table)}

Query-dependent predictions alone are insufficient: a wrong answer can also change. Interpret that column alongside answer accuracy. The unique-value condition is common under the original training distribution; it is not a wholly new key/value format.

## Follow the failed branch

The 32-mode branches are exploratory single-seed checks after observing weak eight-mode recall. They retain the same approximate parameter budget by reducing feed-forward width; this tests a parameter-allocation choice, not a pure capacity increase.

{chr(10).join(mode_table)}

The striped fallback retains a direct global retrieval mechanism. Its observed success justifies testing reduced attention frequency; it does not show that the modal memory itself learned exact retrieval.

## Transfer beyond the trained configuration

All retrieval models train with four pairs and length 48. The following values are means over three seeds, using fresh generated mappings. They change pair count or context length without additional training.

{chr(10).join(transfer_table)}

Length extrapolation and additional stored pairs are distinct stresses. A failure limits the result to the trained regime even if default accuracy is perfect. Fixed absolute record positions and sinusoidal positions are possible sources of length sensitivity.

## Measured CPU costs

{chr(10).join(timing_table)}

Training numbers include forward, loss, backward, clipping, and AdamW; spectral kernels are rebuilt as weights change. Ten measured steps follow three warmups. Decode numbers are medians of five repeated 32-token continuations after one warmup, with cached execution of the entire model. Prefix lengths are starting lengths; full attention grows by another 32 tokens during each continuation. Benchmarks are run sequentially after training, with no intentional competing experiment jobs. Random model weights are used for timing; these are architecture timings, not quality-matched deployment results.

Cache numbers count retained tensor payload, including complex modal states, unused reserved KV capacity and the common previous-token stem, but exclude weights, allocator overhead, attention workspaces and the runtime. Full-attention KV buffers are preallocated with 32 spare positions; none of the measured continuations reallocates or concatenates a growing global cache. Small local-window buffers still concatenate only their bounded window. These are Python/PyTorch reference paths rather than fused deployment kernels. Neither this implementation nor these Apple M1 CPU timings establish a GPU advantage.

![Timing comparison](timing.png)

Theoretical fixed-memory benefits are useful even if this FFT implementation is slower. They are different claims. A striped model still retains one growing global KV cache and an O(L²) training component.

## Checks and provenance

- All five model variants pass future-perturbation, prefix consistency, and future-gradient checks at float32 tolerances. Complete-forward and cached-generation logits agree on tested inputs. Trained retrieval checkpoints are also checked by `diagnose.py`.
- Spectral input gradients pass double-precision finite-difference checking; selected decay, frequency and residue derivatives are checked independently. This is a focused implementation check, not exhaustive numerical verification.
- Bounded local attention agrees with a dense masked reference. The bounded implementation does not materialize a full L×L attention score matrix.
- Host: {checks['environment']}. Timings are local measurements on this machine, with no extrapolation to accelerator hardware.
- Text source: [Tiny Shakespeare]({data['url']}); SHA-256 `{data['sha256']}`. First {data['train_characters']:,} characters train, final {data['validation_characters']:,} validate. The 65-character inventory is built from the full file; no validation token sequence is used for training. Validation samples are repeated overlapping windows, so 102,400 scored next-character positions per seed are not 102,400 independent observations.
- `samples.txt` contains fixed-prompt character generation for illustration; it is not a quality metric. Raw curves, evaluation seeds, environment, final weights and per-run records are retained beside this report.

## Scope and next decision

This is a completed small-scale screening experiment. It tests causal trainability, short-budget text prediction, a controlled retrieval diagnostic, and CPU timing. It does not test useful-scale pretraining, GPU kernels, broad language understanding, long-context text quality, equal-quality large-model throughput, or convergence. The original 3% perplexity / 5 percentage-point recall / 20% utility thresholds cannot be interpreted as general acceptance from this corpus alone.

Use the retrieval controls to decide whether to keep the memory mechanism. If it remains at the query-blind baseline, the next meaningful branch is explicit content-addressable memory or selective state updates, rather than simply calling its global mixture “semantic context.” Keep the full-attention control and the successful striped fallback visible in any next experiment.

Two primary-source connections guide that next branch. [Structured state-space duality / Mamba-2](https://arxiv.org/abs/2405.21060) connects structured state-space operators and variants of attention through structured matrix representations. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) combines adaptive forgetting with targeted associative-memory updates and tests retrieval. These papers motivate testing a queryable key/value state rather than adding more fixed temporal modes. They do not prove the cause of this prototype's failure or establish that a proposed modification will work. Input-dependent memory updates generally require a scan or chunked algorithm instead of this single stationary FFT convolution.
'''
(OUT/'REPORT.md').write_text(report)
manifest = {'source_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
            'records': len(records), 'data_sha256': data['sha256'],
            'review': 'Inspect final PNG and grayscale renders; verify report values against summary.json before delivery.'}
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'quality': {task: {k: summary[task][k]['perplexity' if task=='text' else 'accuracy'] for k in ORDER}
                            for task in ['recall', 'text']}, 'relative_ppl_gap_percent': gaps}, indent=2))
