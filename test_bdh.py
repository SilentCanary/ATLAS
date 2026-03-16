#!/usr/bin/env python
"""Quick test for BDH small config."""
from bdh.bdh import BDH, CodeBDHConfigSmall

config = CodeBDHConfigSmall()
print(f"Config: {config.n_embd}d, {config.n_head} heads, {config.n_layer} layers")
print(f"Block size: {config.block_size}")

N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
print(f"Neurons per head: {N}")
print(f"Total neurons: {N * config.n_head}")

model = BDH(config)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
print(f"Memory estimate: ~{params * 4 / 1e6:.0f}MB")
print("SUCCESS")
