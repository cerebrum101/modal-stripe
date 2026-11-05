"""Small, reproducible CPU experiments. No claims about GPU or large-model speed."""
import argparse
import hashlib
import json
import math
import platform
import statistics
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
KINDS = ['full', 'local', 'spectral', 'hybrid', 'striped']


class Attention(nn.Module):
    def __init__(self, d, window=None):
        super().__init__()
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.window = window
        self.heads = 4

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        if self.window is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            w = min(t, self.window)
            # Materialize only T*w interactions, never a masked T*T matrix.
            kw = F.pad(k, (0, 0, w-1, 0)).unfold(2, w, 1).transpose(-1, -2)
            vw = F.pad(v, (0, 0, w-1, 0)).unfold(2, w, 1).transpose(-1, -2)
            score = (q.unsqueeze(-2) * kw).sum(-1) / math.sqrt(d//self.heads)
            valid = torch.arange(w, device=x.device)[None, :] >= (w-1-torch.arange(t, device=x.device)[:, None])
            score = score.masked_fill(~valid[None, None], float('-inf'))
            y = (score.softmax(-1).unsqueeze(-1) * vw).sum(-2)
        return self.out(y.transpose(1, 2).reshape(b, t, d))

    def cache(self, x):
        b, t, d = x.shape
        _, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        if self.window is None:
            kb = k.new_empty(b, self.heads, t+32, d//self.heads)
            vb = v.new_empty(b, self.heads, t+32, d//self.heads)
            kb[:, :, :t], vb[:, :, :t] = k, v
            return {'k': kb, 'v': vb, 'length': t}
        return k[:, :, -self.window:].contiguous(), v[:, :, -self.window:].contiguous()

    def step(self, x, state=None):
        b, d = x.shape
        q, k, v = self.qkv(x).reshape(b, 3, self.heads, d//self.heads).permute(1, 0, 2, 3)
        q, k, v = q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)
        if self.window is None:
            if state is None:
                state = {'k': k.new_empty(b, self.heads, 32, d//self.heads),
                         'v': v.new_empty(b, self.heads, 32, d//self.heads), 'length': 0}
            length = state['length']
            if length == state['k'].shape[2]:
                kb = k.new_empty(b, self.heads, 2*length, d//self.heads)
                vb = v.new_empty(b, self.heads, 2*length, d//self.heads)
                kb[:, :, :length], vb[:, :, :length] = state['k'], state['v']
                state = {'k': kb, 'v': vb, 'length': length}
            state['k'][:, :, length:length+1] = k
            state['v'][:, :, length:length+1] = v
            state['length'] = length+1
            y = F.scaled_dot_product_attention(q, state['k'][:, :, :length+1],
                                               state['v'][:, :, :length+1]).reshape(b, d)
            return self.out(y), state
        if state is not None:
            k = torch.cat((state[0], k), 2)
            v = torch.cat((state[1], v), 2)
        if self.window is not None:
            k, v = k[:, :, -self.window:].contiguous(), v[:, :, -self.window:].contiguous()
        y = F.scaled_dot_product_attention(q, k, v).reshape(b, d)
        return self.out(y), (k, v)


class Spectral(nn.Module):
    def __init__(self, d, modes=8):
        super().__init__()
        self.proj = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.log_decay = nn.Parameter(torch.linspace(math.log(1/256), math.log(1/2), modes))
        self.omega = nn.Parameter(torch.linspace(0, math.pi, modes))
        self.residue = nn.Parameter(torch.randn(modes, d, 2) * .1/math.sqrt(modes))
        self.frequency_gain = nn.Parameter(torch.zeros(()))

    def kernel(self, length):
        lag = torch.arange(length, device=self.omega.device)[:, None]
        decay = torch.exp(-lag * self.log_decay.exp()[None])
        phase = lag * self.omega[None]
        return ((decay * phase.cos()) @ self.residue[:, :, 0]
                - (decay * phase.sin()) @ self.residue[:, :, 1])

    def forward(self, x, frequency=None):
        value, write, read = self.proj(x).chunk(3, -1)
        if frequency is not None:
            write = write + self.frequency_gain * frequency.unsqueeze(-1)
        u = value * write.sigmoid()
        t = x.shape[1]
        n = 1 << (2*t-2).bit_length()
        kernel = self.kernel(t)
        z = torch.fft.irfft(torch.fft.rfft(u, n=n, dim=1)
                            * torch.fft.rfft(kernel, n=n, dim=0)[None], n=n, dim=1)[:, :t]
        return self.out(z * read.sigmoid())

    def step(self, x, state=None):
        value, write, read = self.proj(x).chunk(3, -1)
        pole = torch.polar(torch.exp(-self.log_decay.exp()), self.omega)
        if state is None:
            state = torch.zeros(x.shape[0], len(pole), x.shape[-1], device=x.device, dtype=torch.complex64)
        state = pole[None, :, None] * state + (value * write.sigmoid())[:, None]
        residue = torch.view_as_complex(self.residue.contiguous())
        z = (state * residue[None]).sum(1).real
        return self.out(z * read.sigmoid()), state

    def cache(self, x):
        value, write, _ = self.proj(x).chunk(3, -1)
        u = value * write.sigmoid()
        pole = torch.polar(torch.exp(-self.log_decay.exp()), self.omega)
        lag = torch.arange(x.shape[1], device=x.device)
        basis = pole[None] ** lag[:, None]
        return torch.einsum('tr,btd->brd', basis, u.flip(1).to(basis.dtype))


class Block(nn.Module):
    def __init__(self, kind, d, modes, window, budget=50000):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mixers = nn.ModuleList()
        if kind in ('spectral', 'hybrid'):
            self.mixers.append(Spectral(d, modes))
        if kind in ('full', 'local', 'hybrid'):
            self.mixers.append(Attention(d, None if kind == 'full' else window))
        mixer_params = sum(p.numel() for p in self.mixers.parameters())
        hidden = max(8, round((budget - mixer_params - 4*d)/(2*d)))
        self.ff = nn.Sequential(nn.Linear(d, hidden, bias=False), nn.GELU(), nn.Linear(hidden, d, bias=False))

    def forward(self, x, frequency=None):
        for mixer in self.mixers:
            normalized = self.norm(x)
            update = (mixer(normalized, frequency) if isinstance(mixer, Spectral)
                      else mixer(normalized))
            x = x + update
        return x + self.ff(self.norm2(x))


class LM(nn.Module):
    def __init__(self, kind, vocab, d=64, layers=2, modes=8, window=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.frequency_embedding = nn.Parameter(torch.zeros(d))
        # A shared causal two-token stem exposes key/value adjacency to every mixer.
        self.previous = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([Block('full' if kind == 'striped' and i % 2 else
                                          'hybrid' if kind == 'striped' else kind,
                                          d, modes, window) for i in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, tokens, frequency=None):
        x = self.embed(tokens)
        x = x + self.previous(F.pad(x[:, :-1], (0, 0, 1, 0)))
        t, d = x.shape[1:]
        pos = torch.arange(t, device=x.device)[:, None]
        angle = pos * torch.exp(torch.arange(0, d, 2, device=x.device) * (-math.log(10000)/d))
        pe = torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)
        x = x + pe[None]
        if frequency is not None:
            x = x + self.frequency_embedding * frequency[:, :, None]
        for block in self.blocks:
            x = block(x, frequency)
        return self.head(self.norm(x))

    def prefill(self, tokens):
        raw = self.embed(tokens)
        x = raw + self.previous(F.pad(raw[:, :-1], (0, 0, 1, 0)))
        t, d = x.shape[1:]
        angle = torch.arange(t, device=x.device)[:, None] * torch.exp(
            torch.arange(0, d, 2, device=x.device) * (-math.log(10000)/d))
        x = x + torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)[None]
        states = []
        for block in self.blocks:
            block_states = []
            for mixer in block.mixers:
                normalized = block.norm(x)
                block_states.append(mixer.cache(normalized))
                x = x + mixer(normalized)
            x = x + block.ff(block.norm2(x))
            states.append(block_states)
        return self.head(self.norm(x)), {'previous': raw[:, -1].clone(), 'states': states, 'position': t}

    def step(self, token, cache=None):
        raw = self.embed(token)
        previous = torch.zeros_like(raw) if cache is None else cache['previous']
        position = 0 if cache is None else cache['position']
        x = raw + self.previous(previous)
        d = x.shape[-1]
        angle = position * torch.exp(torch.arange(0, d, 2, device=x.device) * (-math.log(10000)/d))
        x = x + torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)[None]
        states = []
        for i, block in enumerate(self.blocks):
            block_states = []
            for j, mixer in enumerate(block.mixers):
                state = None if cache is None else cache['states'][i][j]
                update, state = mixer.step(block.norm(x), state)
                x = x + update
                block_states.append(state)
            x = x + block.ff(block.norm2(x))
            states.append(block_states)
        return self.head(self.norm(x)), {'previous': raw, 'states': states, 'position': position+1}


def recall_batch(generator, batch=32, length=48, pairs=4):
    assert 2*pairs < length-16
    x = torch.randint(33, 41, (batch, length), generator=generator)
    keys = torch.rand(batch, 16, generator=generator).argsort(-1)[:, :pairs]
    values = torch.randint(16, 32, (batch, pairs), generator=generator)
    x[:, :2*pairs:2] = keys
    x[:, 1:2*pairs:2] = values
    query = torch.randint(pairs, (batch,), generator=generator)
    x[:, -2] = 32
    x[:, -1] = keys[torch.arange(batch), query]
    return x, values[torch.arange(batch), query]


def load_text():
    path = ROOT/'data/shakespeare.txt'
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    chars = sorted(set(text))
    lookup = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([lookup[c] for c in text], dtype=torch.long)
    split = int(.9*len(data))
    return data[:split], data[split:], len(chars), {
        'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'characters': len(data),
        'train_characters': split, 'validation_characters': len(data)-split,
        'vocab': chars, 'split': 'contiguous first 90% train; final 10% validation'}


def text_batch(data, generator, batch=16, length=128):
    starts = torch.randint(len(data)-length, (batch,), generator=generator)
    seq = data[starts[:, None]+torch.arange(length+1)[None]]
    return seq[:, :-1], seq[:, 1:]


def loss_for(model, x, y, task):
    logits = model(x)
    return F.cross_entropy(logits[:, -1] if task == 'recall' else logits.flatten(0, 1),
                           y if task == 'recall' else y.flatten())


@torch.no_grad()
def evaluate(model, task, val, seed, batches=20, length=None, pairs=4):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses, correct, count = [], 0, 0
    for _ in range(batches):
        x, y = (recall_batch(gen, length=length or 48, pairs=pairs) if task == 'recall'
                else text_batch(val, gen))
        logits = model(x)
        selected = logits[:, -1] if task == 'recall' else logits.flatten(0, 1)
        target = y if task == 'recall' else y.flatten()
        losses.append(F.cross_entropy(selected, target).item())
        correct += (selected.argmax(-1) == target).sum().item()
        count += target.numel()
    model.train()
    return {'loss_nats': statistics.mean(losses), 'accuracy': correct/count,
            'evaluated_targets': count, 'perplexity': math.exp(statistics.mean(losses))}


def train(args):
    train_data = val = metadata = None
    vocab = 41
    if args.task == 'text':
        train_data, val, vocab, metadata = load_text()
    torch.manual_seed(args.seed)
    model = LM(args.kind, vocab, modes=args.modes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002 if args.task == 'recall' else .001,
                                 weight_decay=.01)
    gen = torch.Generator().manual_seed(args.seed+1000)
    initial = evaluate(model, args.task, val, 12345)
    history = []
    train_seconds = 0.
    for step in range(1, args.steps+1):
        start = time.perf_counter()
        x, y = recall_batch(gen) if args.task == 'recall' else text_batch(train_data, gen)
        # Identical token budget; cosine decay is shared across variants.
        lr = (.002 if args.task == 'recall' else .001) * (.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, x, y, args.task)
        assert torch.isfinite(loss), (step, loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        train_seconds += time.perf_counter()-start
        if step == 1 or step % 100 == 0 or step == args.steps:
            metrics = evaluate(model, args.task, val, 12345)
            row = {'step': step, 'train_loss': loss.item(), 'train_seconds': train_seconds, **metrics}
            history.append(row)
            print(json.dumps({'task': args.task, 'kind': args.kind, 'seed': args.seed, **row}), flush=True)
    final = evaluate(model, args.task, val, 54321, batches=50)
    transfer = {}
    if args.task == 'recall':
        for length, pairs in [(48, 2), (48, 8), (96, 4), (192, 4)]:
            transfer[f'L{length}_pairs{pairs}'] = evaluate(model, args.task, val, 54321, length=length, pairs=pairs)
    checkpoint = ROOT/f'results/{args.task}_{args.kind}_s{args.seed}_r{args.modes}.pt'
    torch.save(model.state_dict(), checkpoint)
    result = {'task': args.task, 'kind': args.kind, 'seed': args.seed, 'steps': args.steps,
              'modes': args.modes, 'parameters': sum(p.numel() for p in model.parameters()),
              'initial': initial, 'final': final, 'transfer': transfer, 'history': history,
              'train_seconds': train_seconds, 'training_tokens': args.steps*(32*48 if args.task == 'recall' else 16*128),
              'environment': environment(), 'data': metadata, 'checkpoint': str(checkpoint)}
    checkpoint.with_suffix('.json').write_text(json.dumps(result, indent=2))


def environment():
    return {'torch': torch.__version__, 'python': platform.python_version(),
            'machine': platform.machine(), 'device': 'cpu', 'dtype': 'float32',
            'threads': torch.get_num_threads(), 'platform': platform.platform()}


def checks():
    torch.manual_seed(7)
    results = {}
    # Double-precision gradcheck on a genuinely differentiable FFT operation.
    spectral = Spectral(4, 3).double()
    x = torch.randn(1, 7, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(spectral, (x,), eps=1e-6, atol=1e-4)
    results['input_gradcheck'] = True
    # Check learned filter-parameter gradients against finite differences too.
    value = spectral(x).square().sum()
    value.backward()
    gradient_errors = {}
    for name in ['log_decay', 'omega', 'residue']:
        p = getattr(spectral, name)
        index = (0,)*p.ndim
        analytic = p.grad[index].item()
        with torch.no_grad():
            original = p[index].item()
            p[index] = original+1e-6
            plus = spectral(x).square().sum().item()
            p[index] = original-1e-6
            minus = spectral(x).square().sum().item()
            p[index] = original
        numeric = (plus-minus)/2e-6
        gradient_errors[name] = abs(analytic-numeric)
        assert abs(analytic-numeric) < 1e-5
    results['parameter_gradient_absolute_errors'] = gradient_errors
    # Independent dense reference for the bounded local-attention implementation.
    local = Attention(16, 4)
    xlocal = torch.randn(2, 11, 16)
    q, k, v = local.qkv(xlocal).reshape(2, 11, 3, 4, 4).permute(2, 0, 3, 1, 4)
    position = torch.arange(11)
    lag = position[:, None]-position[None, :]
    mask = (lag >= 0) & (lag < 4)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    reference = local.out(reference.transpose(1, 2).reshape(2, 11, 16))
    local_error = (reference-local(xlocal)).abs().max().item()
    assert local_error < 1e-6
    results['local_attention_dense_reference_error'] = local_error
    for kind in KINDS:
        model = LM(kind, 41)
        tokens = torch.randint(41, (2, 31))
        changed = tokens.clone()
        changed[:, 16:] = torch.randint(41, (2, 15))
        with torch.no_grad():
            original = model(tokens)
            error = (original[:, :16]-model(changed)[:, :16]).abs().max().item()
            prefix_error = (original[:, :16]-model(tokens[:, :16])).abs().max().item()
            cache = None
            streamed = []
            for token in tokens.unbind(1):
                logits, cache = model.step(token, cache)
                streamed.append(logits)
            stream_error = (original-torch.stack(streamed, 1)).abs().max().item()
            _, cache = model.prefill(tokens[:, :16])
            prefixed = []
            for token in tokens[:, 16:].unbind(1):
                logits, cache = model.step(token, cache)
                prefixed.append(logits)
            prefill_error = (original[:, 16:]-torch.stack(prefixed, 1)).abs().max().item()
            assert max(stream_error, prefill_error) < 2e-5, (kind, stream_error, prefill_error)
            growth_error = None
            if kind in ('full', 'striped'):
                longer = torch.randint(41, (1, 97))
                expected = model(longer)[:, 16:]
                _, growth_cache = model.prefill(longer[:, :16])
                grown = []
                for token in longer[:, 16:].unbind(1):
                    logits, growth_cache = model.step(token, growth_cache)
                    grown.append(logits)
                growth_error = (expected-torch.stack(grown, 1)).abs().max().item()
                assert growth_error < 2e-5, (kind, growth_error)
        assert max(error, prefix_error) < 2e-5, (kind, error, prefix_error)
        embedded = model.embed(tokens).detach().requires_grad_(True)
        out = embedded
        for block in model.blocks:
            out = block(out)
        grad = torch.autograd.grad(out[:, :16].square().sum(), embedded)[0]
        future_gradient = grad[:, 16:].abs().max().item()
        assert future_gradient < 2e-5
        results[kind] = {'future_perturbation': error, 'prefix_error': prefix_error,
                         'future_gradient': future_gradient,
                         'streamed_logits_error': stream_error, 'prefill_decode_error': prefill_error,
                         'cache_growth_logits_error': growth_error,
                         'parameters': sum(p.numel() for p in model.parameters())}
    spectral = Spectral(64)
    x = torch.randn(2, 127, 64)
    with torch.no_grad():
        expected = spectral(x)
        state = None
        outputs = []
        for token in x.unbind(1):
            y, state = spectral.step(token, state)
            outputs.append(y)
        error = (expected-torch.stack(outputs, 1)).abs().max().item()
    assert error < 2e-5, error
    results['float32_fft_vs_recurrence'] = error
    results['environment'] = environment()
    (ROOT/'results/checks.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def benchmark():
    results = []
    # End-to-end model training step, including gradient clipping and AdamW.
    for length in [64, 128, 256, 512, 1024]:
        for kind in KINDS:
            torch.manual_seed(0)
            model = LM(kind, 65)
            opt = torch.optim.AdamW(model.parameters(), lr=.001)
            x = torch.randint(65, (4, length))
            y = torch.randint(65, (4, length))
            times = []
            for iteration in range(13):
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss_for(model, x, y, 'text').backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                elapsed = time.perf_counter()-start
                if iteration >= 3:
                    times.append(elapsed)
            row = {'scope': 'full_training_step', 'kind': kind, 'length': length, 'batch': 4,
                   'median_ms': 1000*statistics.median(times), 'min_ms': 1000*min(times),
                   'max_ms': 1000*max(times), 'repeats': len(times)}
            results.append(row)
            print(json.dumps(row), flush=True)
    (ROOT/'results/benchmark.json').write_text(json.dumps({'environment': environment(), 'rows': results}, indent=2))


@torch.no_grad()
def decode_benchmark():
    def cache_bytes(item):
        if torch.is_tensor(item):
            return item.numel()*item.element_size()
        if isinstance(item, dict):
            return sum(cache_bytes(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(cache_bytes(v) for v in item)
        return 0
    rows = []
    for length in [64, 256, 1024, 4096]:
        for kind in KINDS:
            torch.manual_seed(0)
            model = LM(kind, 65).eval()
            prefix = torch.randint(65, (1, length))
            continuation = torch.randint(65, (1, 32))
            decode_times, prefill_times = [], []
            for repeat in range(6):
                start = time.perf_counter()
                _, cache = model.prefill(prefix)
                prefill_time = time.perf_counter()-start
                before = cache_bytes(cache)
                start = time.perf_counter()
                for token in continuation.unbind(1):
                    _, cache = model.step(token, cache)
                elapsed = time.perf_counter()-start
                if repeat:
                    decode_times.append(1000*elapsed/continuation.shape[1])
                    prefill_times.append(1000*prefill_time)
            row = {'kind': kind, 'context': length, 'batch': 1, 'decode_tokens': 32,
                   'decode_median_ms_per_token': statistics.median(decode_times),
                   'decode_min_ms_per_token': min(decode_times),
                   'decode_max_ms_per_token': max(decode_times),
                   'prefill_median_ms_including_cache_construction': statistics.median(prefill_times),
                   'cache_bytes_before_decode': before, 'repeats': len(decode_times)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    (ROOT/'results/decode.json').write_text(json.dumps({'environment': environment(), 'rows': rows,
        'cache_policy': 'preallocated full KV cache, +32 slots at prefill; grows geometrically if needed; local window cache; complex64 modal state; includes shared stem; tensor payload including unused capacity'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'train', 'benchmark', 'decode'])
    parser.add_argument('--task', choices=['recall', 'text'], default='recall')
    parser.add_argument('--kind', choices=KINDS, default='hybrid')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ROOT.joinpath('results').mkdir(exist_ok=True)
    ROOT.joinpath('data').mkdir(exist_ok=True)
    {'check': checks, 'benchmark': benchmark, 'decode': decode_benchmark, 'train': lambda: train(args)}[args.action]()
