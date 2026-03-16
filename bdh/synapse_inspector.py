"""
Synapse Inspector for Code-BDH.

Analyzes which synapses (neuron pairs) activate for specific code concepts.
BDH's monosemantic property means individual synapses map to specific concepts.

Usage:
    inspector = SynapseInspector(model, tokenizer)
    concept_map = inspector.build_concept_map(code_samples)
    inspector.save_map(concept_map, "bdh/concept_map.json")
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch


class SynapseInspector:
    """Analyzes BDH synapse activations to build concept maps."""

    # Code concept categories with representative samples
    CONCEPT_PROBES = {
        "import_statement": [
            "import os\nimport sys\nimport json",
            "from typing import List, Dict, Optional",
            "from pathlib import Path",
        ],
        "function_definition": [
            "def process_data(input_list):\n    result = []\n    return result",
            "def calculate_average(numbers):\n    return sum(numbers) / len(numbers)",
            "def validate_input(data, schema):\n    pass",
        ],
        "class_definition": [
            "class UserModel:\n    def __init__(self, name, email):\n        self.name = name",
            "class DatabaseConnection:\n    def __init__(self, host, port):\n        self.conn = None",
            "class APIHandler(BaseHandler):\n    pass",
        ],
        "error_handling": [
            "try:\n    result = risky_operation()\nexcept ValueError as e:\n    logger.error(e)",
            "try:\n    data = json.loads(raw)\nexcept (json.JSONDecodeError, KeyError):\n    return None",
            "raise ValueError(f'Invalid input: {value}')",
        ],
        "loop_pattern": [
            "for item in collection:\n    process(item)",
            "while not done:\n    step()\n    done = check()",
            "results = [f(x) for x in data if x > 0]",
        ],
        "decorator_pattern": [
            "@app.route('/api/users', methods=['GET'])\ndef get_users():\n    pass",
            "@staticmethod\ndef helper():\n    pass",
            "@property\ndef name(self):\n    return self._name",
        ],
        "string_formatting": [
            "message = f'Hello, {name}! You have {count} items.'",
            "query = 'SELECT * FROM {} WHERE id = %s'.format(table)",
            "log_line = '{}: {} - {}'.format(timestamp, level, msg)",
        ],
        "file_io": [
            "with open(path, 'r', encoding='utf-8') as f:\n    data = json.load(f)",
            "with open(output, 'w') as f:\n    f.write(content)",
            "os.makedirs(directory, exist_ok=True)",
        ],
        "data_structure": [
            "config = {'host': 'localhost', 'port': 8080, 'debug': True}",
            "users = [{'name': 'Alice', 'role': 'admin'}, {'name': 'Bob'}]",
            "result = set(a) & set(b)",
        ],
        "async_pattern": [
            "async def fetch_data(url):\n    async with aiohttp.get(url) as resp:\n        return await resp.json()",
            "await asyncio.gather(*tasks)",
            "async for item in stream:\n    yield process(item)",
        ],
    }

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def get_activations(self, text: str) -> List[Dict[str, torch.Tensor]]:
        """Run text through model and return per-layer activations."""
        ids = self.tokenizer.encode(text)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        _, _, activations = self.model(idx, return_activations=True)
        return activations

    def get_synapse_fingerprint(self, text: str, layer: int = -1) -> torch.Tensor:
        """
        Get a synapse fingerprint for a piece of code.

        Returns the mean gated activation (x_sparse * y_sparse) across all
        tokens and heads at a given layer. This represents which synapses
        fired for this code.
        """
        activations = self.get_activations(text)
        layer_act = activations[layer]
        # gated shape: (1, n_heads, seq_len, N_neurons)
        gated = layer_act["gated"]
        # Mean across batch, heads, and sequence → (N_neurons,)
        return gated.mean(dim=(0, 1, 2))

    def build_concept_map(
        self,
        custom_probes: Optional[Dict[str, List[str]]] = None,
        top_k_synapses: int = 50,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Build synapse→concept mapping by running concept probes.

        For each concept category, runs probe code samples and identifies
        which synapses activate most strongly and consistently.

        Returns:
            Dict mapping concept name → list of (synapse_idx, avg_activation)
        """
        probes = custom_probes or self.CONCEPT_PROBES
        concept_map = {}

        for concept, samples in probes.items():
            fingerprints = []
            for sample in samples:
                try:
                    fp = self.get_synapse_fingerprint(sample)
                    fingerprints.append(fp)
                except Exception as e:
                    print(f"  Warning: probe failed for {concept}: {e}")

            if not fingerprints:
                continue

            # Average fingerprint across all samples of this concept
            avg_fp = torch.stack(fingerprints).mean(dim=0)

            # Find top-k synapses with highest activation
            top_vals, top_idxs = avg_fp.topk(min(top_k_synapses, len(avg_fp)))

            concept_map[concept] = [
                (idx.item(), val.item())
                for idx, val in zip(top_idxs, top_vals)
                if val.item() > 0  # only active synapses
            ]

            print(f"  {concept}: {len(concept_map[concept])} active synapses "
                  f"(max: {top_vals[0].item():.4f})")

        return concept_map

    def build_reverse_map(
        self, concept_map: Dict[str, List[Tuple[int, float]]]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Build reverse map: synapse_idx → list of (concept, activation).
        Useful for reading which concepts a synapse responds to.
        """
        reverse = defaultdict(list)
        for concept, synapses in concept_map.items():
            for syn_idx, activation in synapses:
                reverse[syn_idx].append((concept, activation))

        # Sort each synapse's concepts by activation strength
        for syn_idx in reverse:
            reverse[syn_idx].sort(key=lambda x: x[1], reverse=True)

        return dict(reverse)

    def diagnose_code(
        self, code: str, concept_map: Dict[str, List[Tuple[int, float]]]
    ) -> Dict[str, float]:
        """
        Given a piece of code, determine which concepts it contains
        by comparing its synapse fingerprint to the concept map.

        Returns:
            Dict mapping concept → confidence score (0-1)
        """
        fp = self.get_synapse_fingerprint(code)

        scores = {}
        for concept, synapses in concept_map.items():
            if not synapses:
                continue
            # Score = average activation at the concept's key synapses
            activations = [fp[idx].item() for idx, _ in synapses[:20]]
            scores[concept] = sum(activations) / len(activations) if activations else 0.0

        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def save_map(self, concept_map: Dict, path: str):
        """Save concept map to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(concept_map, f, indent=2)
        print(f"Concept map saved to {path}")

    def load_map(self, path: str) -> Dict:
        """Load concept map from JSON."""
        with open(path, "r") as f:
            return json.load(f)

    def print_report(self, concept_map: Dict[str, List[Tuple[int, float]]]):
        """Print a human-readable report of the concept map."""
        print("\n" + "=" * 60)
        print("  BDH Synapse → Concept Report")
        print("=" * 60)

        for concept, synapses in concept_map.items():
            active = [s for s in synapses if s[1] > 0]
            if not active:
                continue

            max_act = active[0][1] if active else 0
            bar_len = int(max_act * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)

            print(f"\n  {concept}")
            print(f"    Active synapses: {len(active)}")
            print(f"    Max activation:  {max_act:.4f} [{bar}]")
            print(f"    Top 5 synapses:  {[s[0] for s in active[:5]]}")

        print("\n" + "=" * 60)
