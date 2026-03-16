# BDH (Baby Dragon Hatchling) adapted for code intelligence
# Original: https://github.com/pathwaycom/bdh (MIT License, Pathway Technology Inc.)
# Adapted for ATLAS: subword tokenization, code-specific config, state extraction

import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    """Default config matching original BDH paper."""
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    block_size: int = 512


@dataclasses.dataclass
class CodeBDHConfig(BDHConfig):
    """Config adapted for Python code understanding."""
    n_layer: int = 8              # more iterations for deeper code reasoning
    n_embd: int = 512             # larger embedding for richer code vocabulary
    dropout: float = 0.1
    n_head: int = 8               # more heads for parallel concept tracking
    mlp_internal_dim_multiplier: int = 64  # N = 64*512/8 = 4096 neurons
    vocab_size: int = 8192        # BPE code vocabulary
    block_size: int = 1024        # longer context for code


@dataclasses.dataclass
class CodeBDHConfigSmall(BDHConfig):
    """Smaller config for CPU training (~3M params)."""
    n_layer: int = 4              # fewer layers
    n_embd: int = 128             # smaller embedding
    dropout: float = 0.1
    n_head: int = 4               # fewer heads
    mlp_internal_dim_multiplier: int = 32  # N = 32*128/4 = 1024 neurons
    vocab_size: int = 8192        # BPE code vocabulary
    block_size: int = 256         # shorter context for memory


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert K is Q  # Q=K constraint (Hebbian self-reference)
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs

        QR = self.rope(r_phases, Q)
        KR = QR

        # Linear causal attention (no softmax) — strictly causal
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    """
    Baby Dragon Hatchling model.

    Key properties:
    - Shared weights across all layers (layers = timesteps of dynamical system)
    - ReLU sparse positive activations (~5% non-zero)
    - Linear attention in high neuron dimension N
    - Monosemantic synapses emerge naturally
    - State lives on edges (synaptic weights), not nodes
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_activations=False):
        """
        Forward pass.

        Args:
            idx: input token ids (B, T)
            targets: target token ids for loss (B, T), optional
            return_activations: if True, also return per-layer sparse activations
                for synapse analysis

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
            activations: list of (x_sparse, y_sparse) per layer (if return_activations)
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        layer_activations = []

        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N — sparse positive

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # multiplicative gating

            if return_activations:
                layer_activations.append({
                    "x_sparse": x_sparse.detach(),
                    "y_sparse": y_sparse.detach(),
                    "gated": xy_sparse.detach(),
                })

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_activations:
            return logits, loss, layer_activations
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_neuron_count(self):
        """Total neuron particles in the model."""
        C = self.config
        N = C.mlp_internal_dim_multiplier * C.n_embd // C.n_head
        return N * C.n_head

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
