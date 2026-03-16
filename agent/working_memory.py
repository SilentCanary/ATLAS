"""
BDH Working Memory — maintains evolving synaptic state across multi-step tasks.

BDH's sigma state (linear attention accumulates outer products) naturally
strengthens synapses between co-occurring concepts. This module exposes
that as working memory for the agent loop.

Example: When processing a multi-step task:
  Step 1: "create user model" → ORM pattern synapses activate
  Step 2: "add authentication" → auth synapses activate + ORM persists
  Step 3: "connect to API"    → all three concept clusters active simultaneously
"""

from typing import List, Dict, Tuple, Optional

import torch


class BDHWorkingMemory:
    """Manages BDH's accumulated state across reasoning steps."""

    def __init__(self, bdh_model, tokenizer, concept_map, device: str = "cpu"):
        self.model = bdh_model
        self.tokenizer = tokenizer
        self.concept_map = concept_map
        self.device = device
        self.model.eval()

        # Accumulated fingerprints across steps
        self.step_fingerprints: List[torch.Tensor] = []
        self.step_texts: List[str] = []

    @torch.no_grad()
    def process_step(self, text: str) -> List[Tuple[str, float]]:
        """
        Process one reasoning step, accumulating working memory.

        Returns currently active concepts (union of all steps).
        """
        ids = self.tokenizer.encode(text)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        _, _, activations = self.model(idx, return_activations=True)
        fingerprint = activations[-1]["gated"].mean(dim=(0, 1, 2))

        self.step_fingerprints.append(fingerprint)
        self.step_texts.append(text)

        return self.get_active_concepts()

    def get_active_concepts(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get currently active concepts from accumulated working memory.

        Combines fingerprints from all steps with exponential decay
        (recent steps weighted more heavily).
        """
        if not self.step_fingerprints:
            return []

        # Exponentially weighted average: recent steps matter more
        n = len(self.step_fingerprints)
        weights = [0.7 ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        combined = torch.zeros_like(self.step_fingerprints[0])
        for fp, w in zip(self.step_fingerprints, weights):
            combined += fp * w

        # Map to concepts
        scores = {}
        for concept, synapses in self.concept_map.items():
            if not synapses:
                continue
            acts = [combined[idx].item() for idx, _ in synapses[:20]]
            scores[concept] = sum(acts) / len(acts) if acts else 0.0

        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_concept_keywords(self, threshold: float = 0.3) -> List[str]:
        """Get concept names above threshold for query enhancement."""
        return [name for name, score in self.get_active_concepts() if score > threshold]

    def reset(self):
        """Clear working memory for a new task."""
        self.step_fingerprints.clear()
        self.step_texts.clear()

    def get_state_summary(self) -> str:
        """Human-readable summary of current working memory."""
        concepts = self.get_active_concepts(top_k=8)

        lines = [
            f"Working Memory ({len(self.step_fingerprints)} steps accumulated)",
            ""
        ]

        if not concepts:
            lines.append("  (empty)")
            return "\n".join(lines)

        for name, score in concepts:
            bar_len = int(score * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(f"  {name:25s} [{bar}] {score:.3f}")

        lines.append("")
        lines.append("Steps processed:")
        for i, text in enumerate(self.step_texts):
            lines.append(f"  {i+1}. {text[:80]}...")

        return "\n".join(lines)
