"""Zipf-skewed recall screening: loss reweighting vs rarity-conditioned memory."""
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import LM, ROOT, environment

OUT = ROOT/'results'
KINDS = ['full', 'local', 'spectral', 'spectral_weighted', 'spectral_rarity', 'spectral_both']
KEYS, VALUES, LENGTH, PAIRS = 16, 16, 64, 16
PROB = torch.arange(1, KEYS+1, dtype=torch.float32).pow(-1.1)
PROB /= PROB.sum()
RARITY = (-torch.log(PROB)).float()
RARITY = (RARITY-RARITY.mean())/RARITY.std()
FREQUENCY = torch.zeros(48)
FREQUENCY[:KEYS] = RARITY
DISTRACTOR_WEIGHT = torch.tensor([1., .5, 1., 0., 1., .5, 0., 1., 1., 0., .5, 1., 0., 1., .5, 1.])


def batch(gen, n, query_mode='zipf'):
    x = torch.randint(33, 41, (n, LENGTH), generator=gen)
    # Every example receives a new random key-to-value assignment.
    values = torch.rand(n, KEYS, generator=gen).argsort(-1)+KEYS
    x[:, :KEYS] = torch.arange(KEYS)
    x[:, KEYS:2*KEYS] = values
    if query_mode == 'zipf':
        query = torch.multinomial(PROB.expand(n, -1), 1, generator=gen).squeeze(-1)
    elif query_mode == 'uniform':
        query = torch.randint(KEYS, (n,), generator=gen)
    else:
        raise ValueError(query_mode)
    x[:, -2], x[:, -1] = 32, query
    f = torch.zeros(n, LENGTH)
    f[:, :KEYS] = RARITY
    f[:, -1] = RARITY[query]
    target = values.gather(1, query[:, None]).squeeze(-1)
    return x, f, target, query


@torch.no_grad()
def evaluate(model, seed, batches=32):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    key_correct = torch.zeros(KEYS)
    key_total = torch.zeros(KEYS)
    for _ in range(batches):
        x, f, target, query = batch(gen, 64)
        logits = model(x, f if model.kind in ('spectral_rarity', 'spectral_both') else None)[:, -1]
        correct = (logits.argmax(-1) == target).float()
        key_correct.scatter_add_(0, query, correct)
        key_total.scatter_add_(0, query, torch.ones_like(correct))
    zipf_per_key = key_correct/key_total.clamp_min(1)
    # A separate, exactly balanced audit prevents rare keys from disappearing
    # in finite evaluation. Each random mapping is queried once for every key.
    eval_values = torch.rand(batches, KEYS, generator=gen).argsort(-1)+KEYS
    x = torch.randint(33, 41, (batches, LENGTH), generator=gen)
    x[:, :KEYS] = torch.arange(KEYS)
    x[:, KEYS:2*KEYS] = eval_values
    x = x[:, None, :].expand(-1, KEYS, -1).reshape(-1, LENGTH).clone()
    queries = torch.arange(KEYS).repeat(batches)
    x[:, -2], x[:, -1] = 32, queries
    f = torch.zeros_like(x, dtype=torch.float32)
    f[:, :KEYS] = RARITY
    f[:, -1] = RARITY[queries]
    targets = eval_values.gather(1, queries.view(batches, KEYS)).reshape(-1)
    logits = model(x, f if model.kind in ('spectral_rarity', 'spectral_both') else None)[:, -1]
    correct = (logits.argmax(-1) == targets).float().view(batches, KEYS)
    per_key = correct.mean(0)
    model.train()
    ranks = torch.arange(1, KEYS+1)
    # Rank 1 is most frequent; rare is the least frequent quartile.
    common, rare = per_key[:4].mean().item(), per_key[-4:].mean().item()
    return {'micro_zipf_accuracy': (key_correct.sum()/key_total.sum()).item(),
            'zipf_per_key_accuracy': zipf_per_key.tolist(),
            'macro_key_accuracy': per_key.mean().item(), 'common_quartile_accuracy': common,
            'rare_quartile_accuracy': rare, 'rare_minus_common': rare-common,
            'per_key_accuracy': per_key.tolist(), 'per_key_count': [batches]*KEYS,
            'evaluated_queries': int(key_total.sum().item())}


def train(kind, seed, steps=1800):
    torch.manual_seed(seed)
    model = LM(kind if kind in ('full', 'local') else 'spectral', 48)
    model.kind = kind
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    gen = torch.Generator().manual_seed(10000+seed)
    rows = []
    start = time.perf_counter()
    for step in range(1, steps+1):
        x, f, target, query = batch(gen, 32)
        opt.param_groups[0]['lr'] = .002*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/steps)))
        opt.zero_grad(set_to_none=True)
        logits = model(x, f if kind in ('spectral_rarity', 'spectral_both') else None)[:, -1]
        losses = F.cross_entropy(logits, target, reduction='none')
        if kind in ('spectral_weighted', 'spectral_both'):
            weights = PROB[query].rsqrt()
            weights = weights/weights.mean()
            loss = (weights*losses).mean()
        else:
            loss = losses.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError((kind, seed, step))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step in (600, 1200, steps):
            rows.append({'step': step, 'train_seconds': time.perf_counter()-start,
                         **evaluate(model, 70000+seed)})
    final = evaluate(model, 90000+seed, batches=128)
    result = {'branch': 'zipf_query_frequency', 'kind': kind, 'seed': seed,
              'steps': steps, 'parameters': sum(p.numel() for p in model.parameters()),
              'frequency_distribution': PROB.tolist(), 'rarity_feature': 'z-scored negative log Zipf prior for key-token positions',
              'loss_weight': 'inverse square-root query prior, batch-renormalized' if kind in ('spectral_weighted','spectral_both') else 'uniform per-example',
              'training': rows, 'final': final, 'environment': environment(),
              'tokens_seen': steps*32*LENGTH}
    path = OUT/f'zipf_{kind}_s{seed}.json'
    path.write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), path.with_suffix('.pt'))
    print(json.dumps({'kind': kind, 'seed': seed, **final}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--kinds', nargs='*', default=KINDS)
    args = parser.parse_args()
    for seed in range(args.seeds):
        for kind in args.kinds:
            path = OUT/f'zipf_{kind}_s{seed}.json'
            if path.exists():
                continue
            train(kind, seed, steps=args.steps)
