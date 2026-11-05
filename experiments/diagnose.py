"""Post-training controls: query dependence, unique values, trained streaming."""
import json
from pathlib import Path

import torch
from run import LM, ROOT, load_text, recall_batch

torch.set_num_threads(2)
rows = []
for path in sorted((ROOT/'results').glob('recall_*_r*.json')):
    record = json.loads(path.read_text())
    if record['steps'] != 1000:
        continue
    model = LM(record['kind'], 41, modes=record['modes']).eval()
    model.load_state_dict(torch.load(path.with_suffix('.pt'), weights_only=True))
    gen = torch.Generator().manual_seed(9182)
    x, _ = recall_batch(gen, batch=512)
    # Distinct values isolate binding: a query-blind context guess scores 25%.
    values = torch.rand(512, 16, generator=gen).argsort(-1)[:, :4]+16
    x[:, 1:8:2] = values
    queries = x[:, :8:2]
    expanded = x[:, None].expand(-1, 4, -1).clone()
    expanded[:, :, -1] = queries
    predictions = []
    with torch.no_grad():
        for batch in expanded.reshape(-1, 48).split(64):
            predictions.append(model(batch)[:, -1].argmax(-1))
        pred = torch.cat(predictions).reshape(512, 4)
        answer_accuracy = (pred == values).float().mean().item()
        query_changes_answer = (pred != pred[:, :1]).any(-1).float().mean().item()
        _, cache = model.prefill(x[:2, :24])
        streamed = []
        for token in x[:2, 24:].unbind(1):
            out, cache = model.step(token, cache)
            streamed.append(out)
        error = (model(x[:2])[:, 24:]-torch.stack(streamed, 1)).abs().max().item()
        assert error < 1e-4, (path, error)
    row = {'kind': record['kind'], 'seed': record['seed'], 'modes': record['modes'],
           'unique_value_all_query_accuracy': answer_accuracy,
           'contexts_with_query_dependent_predictions': query_changes_answer,
           'contexts': 512, 'queries': 2048, 'query_blind_baseline_accuracy': .25,
           'trained_prefill_decode_max_logit_error': error}
    rows.append(row)
    print(json.dumps(row), flush=True)

# Text samples are illustrative, not an additional quality benchmark.
_, _, vocab, metadata = load_text()
chars = metadata['vocab']
lookup = {c: i for i, c in enumerate(chars)}
samples = []
for kind in ['full', 'local', 'spectral', 'hybrid', 'striped']:
    path = ROOT/f'results/text_{kind}_s0_r8.pt'
    if not path.exists():
        continue
    model = LM(kind, vocab).eval()
    model.load_state_dict(torch.load(path, weights_only=True))
    tokens = torch.tensor([[lookup[c] for c in 'First Citizen:\n']])
    gen = torch.Generator().manual_seed(17)
    with torch.no_grad():
        logits, cache = model.prefill(tokens)
        logits = logits[:, -1]
        continuation = []
        for _ in range(300):
            token = torch.multinomial((logits/.8).softmax(-1), 1, generator=gen).squeeze(-1)
            continuation.append(chars[token.item()])
            logits, cache = model.step(token, cache)
    samples.append(f'[{kind}; seed 0; temperature 0.8]\nFirst Citizen:\n'+''.join(continuation))
(ROOT/'results/diagnostics.json').write_text(json.dumps(rows, indent=2))
(ROOT/'results/samples.txt').write_text('\n\n'.join(samples))
