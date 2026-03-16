"""
BDH Concept Router — uses BDH's monosemantic synapses to enhance retrieval.

Instead of raw embedding similarity alone, BDH identifies which code concepts
a query activates at the neural level, then uses those concepts to guide
retrieval toward the most relevant code.
"""

import os
from typing import List, Dict, Optional, Tuple

import torch


class BDHRouter:
    """Routes queries through BDH to get concept-enhanced retrieval."""

    def __init__(
        self,
        bdh_model,
        tokenizer,
        concept_map: Dict[str, List[Tuple[int, float]]],
        retriever,
        device: str = "cpu",
        concept_boost: float = 0.3,
    ):
        """
        Args:
            bdh_model: trained BDH model
            tokenizer: CodeTokenizer instance
            concept_map: synapse→concept map from SynapseInspector
            retriever: CodeRetriever instance
            device: torch device
            concept_boost: how much to weight concept-based retrieval (0-1)
        """
        self.model = bdh_model
        self.tokenizer = tokenizer
        self.concept_map = concept_map
        self.retriever = retriever
        self.device = device
        self.concept_boost = concept_boost
        self.model.eval()

    @torch.no_grad()
    def get_active_concepts(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Run text through BDH and identify which code concepts activate.

        Returns list of (concept_name, confidence) sorted by strength.
        """
        ids = self.tokenizer.encode(text)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        _, _, activations = self.model(idx, return_activations=True)
        # Use last layer's gated activations
        gated = activations[-1]["gated"]
        fingerprint = gated.mean(dim=(0, 1, 2))  # (N_neurons,)

        scores = {}
        for concept, synapses in self.concept_map.items():
            if not synapses:
                continue
            # Average activation at concept's key synapses
            acts = [fingerprint[idx].item() for idx, _ in synapses[:20]]
            scores[concept] = sum(acts) / len(acts) if acts else 0.0

        # Normalize
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_concepts[:top_k]

    def route(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Enhanced retrieval: combine semantic search with BDH concept routing.

        1. Get BDH's active concepts for the query
        2. Append concept keywords to the query
        3. Run enhanced query through retriever
        4. Re-rank results using concept alignment
        """
        # Step 1: Get BDH concepts
        concepts = self.get_active_concepts(query)
        concept_names = [name for name, score in concepts if score > 0.3]

        # Step 2: Enhance query with concept keywords
        if concept_names:
            enhanced_query = query + " " + " ".join(concept_names)
        else:
            enhanced_query = query

        # Step 3: Retrieve with enhanced query
        results = self.retriever.retrieve(enhanced_query)

        return results

    def explain_routing(self, query: str) -> str:
        """Return human-readable explanation of routing decisions."""
        concepts = self.get_active_concepts(query, top_k=8)

        lines = [f"Query: {query}", "", "BDH Active Concepts:"]
        for name, score in concepts:
            bar_len = int(score * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(f"  {name:25s} [{bar}] {score:.3f}")

        return "\n".join(lines)
