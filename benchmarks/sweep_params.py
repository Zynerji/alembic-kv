#!/usr/bin/env python3
"""Sweep write_alpha and write_temp for optimal codebook quality."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from alembic_kv.hf_cache import AlembicHFCache

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B', torch_dtype=torch.bfloat16,
    device_map='auto', trust_remote_code=True)
model.eval()

print("Loading data...")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = [t for t in dataset['text'] if len(t) > 100][:20]
print(f"  {len(texts)} samples")


def compute_ppl(**kwargs):
    total_loss, total_tok = 0.0, 0
    for text in texts:
        inp = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=1024).to('cuda')
        ids = inp['input_ids']
        if ids.shape[1] < 2:
            continue
        cache = AlembicHFCache(**kwargs)
        with torch.no_grad():
            out = model(ids, past_key_values=cache, use_cache=True)
        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tok += targets.numel()
    return torch.exp(torch.tensor(total_loss / max(total_tok, 1))).item()


# Baseline
print("\nBaseline (no cache):")
ppl_std = compute_ppl(budget=9999)  # budget > any seq → lossless
print(f"  Standard: {ppl_std:.4f}")

# Sweep at budget=128
print(f"\n{'alpha':>8s} {'temp':>8s} {'gamma':>8s} {'ppl':>10s} {'vs_std':>10s}")
print("-" * 50)

best_ppl = float('inf')
best_params = {}

for alpha in [0.01, 0.05, 0.1, 0.2, 0.5]:
    for temp in [0.05, 0.1, 0.5, 1.0]:
        for gamma in [0.0, 0.01, 0.05]:
            p = compute_ppl(budget=128, write_alpha=alpha,
                           write_temp=temp, qarc_gamma=gamma)
            delta = (p / ppl_std - 1) * 100
            marker = " ***" if p < best_ppl else ""
            print(f"{alpha:>8.2f} {temp:>8.2f} {gamma:>8.2f} {p:>10.2f} {delta:>+9.1f}%{marker}")
            if p < best_ppl:
                best_ppl = p
                best_params = {'alpha': alpha, 'temp': temp, 'gamma': gamma}

print(f"\nBest: ppl={best_ppl:.4f} ({(best_ppl/ppl_std-1)*100:+.1f}%)")
print(f"  Params: {best_params}")
