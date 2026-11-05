"""Same-token-budget text comparison: global attention vs spectral+global hybrid."""
import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

import torch
from torch.nn import functional as F

try:
    from .run import LM, ROOT, environment
except ImportError:  # direct execution from this directory
    from run import LM, ROOT, environment

DATA = ROOT/'data'
OUT = ROOT/'results'
SOURCES = {
    'hamlet': ('https://www.gutenberg.org/cache/epub/1524/pg1524.txt', 'pg1524.txt'),
    'alice': ('https://www.gutenberg.org/cache/epub/11/pg11.txt', 'pg11.txt'),
    'moby_dick': ('https://www.gutenberg.org/cache/epub/2701/pg2701.txt', 'pg2701.txt'),
}
CONTEXT, BATCH = 128, 16


def get_text(name):
    url, filename = SOURCES[name]
    path = DATA/filename
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    raw = path.read_bytes()
    source = raw.decode('utf-8', errors='replace')
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if 'START OF THE PROJECT GUTENBERG' in line.upper()), 0)
    end = next((i for i in range(start+1, len(lines))
                if 'END OF THE PROJECT GUTENBERG' in lines[i].upper()), len(lines))
    text = '\n'.join(lines[start+1:end]).strip()
    replacements = {'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
                    '\u2014': '--', '\u2013': '-', '\u2026': '...', '\u00a0': ' '}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    return text, {'url': url, 'sha256_raw': hashlib.sha256(raw).hexdigest(),
                  'raw_bytes': len(raw), 'normalized_characters': len(text),
                  'normalization': 'strip Gutenberg wrapper; map smart quotes/dashes; ASCII'}


def encode(text, lookup):
    return torch.tensor([lookup[c] for c in text], dtype=torch.long)


def sample_batch(data_by_text, generator, batch=BATCH, length=CONTEXT):
    # Select a valid window uniformly from all available training windows.
    limits = [len(x)-length-1 for x in data_by_text]
    cumulative = torch.tensor(limits).cumsum(0)
    picks = torch.randint(int(cumulative[-1]), (batch,), generator=generator)
    rows = []
    for pick in picks.tolist():
        corpus = int(torch.searchsorted(cumulative, torch.tensor(pick), right=True))
        offset = pick - (0 if corpus == 0 else int(cumulative[corpus-1]))
        rows.append(data_by_text[corpus][offset:offset+length+1])
    seq = torch.stack(rows)
    return seq[:, :-1], seq[:, 1:]


@torch.no_grad()
def evaluate(model, data, seed, batches=50):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    total = 0
    for _ in range(batches):
        x, y = sample_batch([data], gen)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
        total += y.numel()
    model.train()
    return {'loss_nats': statistics.mean(losses), 'perplexity': math.exp(statistics.mean(losses)),
            'evaluated_characters': total}


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


@torch.no_grad()
def benchmark_decode(model, prefix, repeats=5, continuation=32):
    model.eval()
    x = prefix[:1]
    warm_logits, warm_cache = model.prefill(x)
    warm_token = warm_logits[:, -1].argmax(-1)
    for _ in range(continuation):
        warm_logits, warm_cache = model.step(warm_token, warm_cache)
        warm_token = warm_logits.argmax(-1)
    rows = []
    for _ in range(repeats):
        start = time.perf_counter()
        logits, cache = model.prefill(x)
        prefill_ms = (time.perf_counter()-start)*1000
        cache_size = tensor_bytes(cache)
        token = logits[:, -1].argmax(-1)
        start = time.perf_counter()
        for _ in range(continuation):
            logits, cache = model.step(token, cache)
            token = logits.argmax(-1)
        decode_ms = (time.perf_counter()-start)*1000/continuation
        rows.append((prefill_ms, decode_ms, cache_size))
    return {'prefill_ms_median': statistics.median(x[0] for x in rows),
            'decode_ms_per_token_median': statistics.median(x[1] for x in rows),
            'cache_bytes': int(statistics.median(x[2] for x in rows)),
            'repeats': repeats, 'continuation_tokens': continuation}


def train(corpus, kind, seed, steps, vocab_size, lookup, train_parts, valid_parts, metadata):
    torch.manual_seed(seed)
    model = LM(kind, vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    gen = torch.Generator().manual_seed(2000+seed)
    train_data = [encode(text, lookup) for text in train_parts]
    valid_data = [encode(text, lookup) for text in valid_parts]
    histories = []
    started = time.perf_counter()
    for step in range(1, steps+1):
        x, y = sample_batch(train_data, gen)
        lr = .001*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/steps)))
        opt.param_groups[0]['lr'] = lr
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError((corpus, kind, seed, step))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step in (steps//2, steps):
            histories.append({'step': step, 'train_loss': loss.item(),
                              'train_seconds': time.perf_counter()-started})
        if step % 100 == 0 or step == steps:
            print(json.dumps({'corpus': corpus, 'kind': kind, 'seed': seed,
                              'step': step, 'loss': loss.item()}), flush=True)
    train_seconds = time.perf_counter()-started
    final = {}
    for i, data in enumerate(valid_data):
        final[metadata[i]['name']] = evaluate(model, data, 70000+seed+i)
    result = {'corpus': corpus, 'kind': kind, 'seed': seed, 'steps': steps,
              'training_tokens': steps*BATCH*CONTEXT,
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'train_seconds': train_seconds, 'history': histories, 'validation': final,
              'decode': {}, 'source_metadata': metadata, 'environment': environment()}
    path = OUT/f'headtohead_{corpus}_{kind}_s{seed}.json'
    path.write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), path.with_suffix('.pt'))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    texts, source_meta = {}, {}
    for name in SOURCES:
        texts[name], source_meta[name] = get_text(name)
    chars = sorted(set(''.join(texts.values())))
    lookup = {c: i for i, c in enumerate(chars)}
    jobs = []
    hamlet = texts['hamlet']
    hamlet_cut = int(.9*len(hamlet))
    jobs.append(('hamlet', [hamlet[:hamlet_cut]], [hamlet[hamlet_cut:]],
                 [source_meta['hamlet'] | {'name': 'hamlet'}]))
    mixed_train, mixed_valid, mixed_meta = [], [], []
    for name in SOURCES:
        text = texts[name]
        cut = int(.9*len(text))
        mixed_train.append(text[:cut])
        mixed_valid.append(text[cut:])
        mixed_meta.append(source_meta[name] | {'name': name})
    jobs.append(('mixed3', mixed_train, mixed_valid, mixed_meta))
    for corpus, train_parts, valid_parts, metadata in jobs:
        for seed in range(args.seeds):
            for kind in ('full', 'striped'):
                out = OUT/f'headtohead_{corpus}_{kind}_s{seed}.json'
                if out.exists():
                    previous = json.loads(out.read_text())
                    if previous.get('steps') == args.steps:
                        print(f'skip existing {out.name}', flush=True)
                        continue
                train(corpus, kind, seed, args.steps, len(chars), lookup,
                      train_parts, valid_parts, metadata)

    # Measure inference after all training jobs have ended, avoiding overlap
    # with training and using the same Hamlet prefix for every checkpoint.
    hamlet_valid = encode(hamlet[hamlet_cut:], lookup)
    for corpus, _, _, _ in jobs:
        for seed in range(args.seeds):
            for kind in ('full', 'striped'):
                path = OUT/f'headtohead_{corpus}_{kind}_s{seed}.json'
                checkpoint = path.with_suffix('.pt')
                if not path.exists() or not checkpoint.exists():
                    continue
                record = json.loads(path.read_text())
                model = LM(kind, len(chars))
                model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                model.eval()
                record['decode'] = {
                    str(length): benchmark_decode(model, hamlet_valid[:length].unsqueeze(0))
                    for length in (128, 512, 1024, 2048, 4096, 8192)}
                path.write_text(json.dumps(record, indent=2))
                print(json.dumps({'benchmark': True, 'corpus': corpus, 'kind': kind,
                                  'seed': seed, 'decode': record['decode']}), flush=True)


if __name__ == '__main__':
    main()
