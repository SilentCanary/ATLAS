"""
Training data pipeline for Code-BDH.

Three data sources mixed during training:
1. Raw Python code (60%) — learns syntax, patterns, idioms
2. Structured representations (30%) — learns ATLAS-style graph relationships
3. Graph-annotated code (10%) — learns upstream/downstream context
"""

import os
import json
import random
from typing import List, Dict, Optional, Iterator


class CodeDataPipeline:
    """Prepares and mixes training data for Code-BDH."""

    def __init__(self, tokenizer, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.block_size = block_size

    # ------------------------------------------------------------------
    # Source 1: Raw Python code
    # ------------------------------------------------------------------
    def collect_raw_code(self, directory: str) -> List[str]:
        """Collect raw Python source files."""
        texts = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                            content = fp.read()
                            if 50 < len(content) < 100_000:  # skip tiny/huge files
                                texts.append(content)
                    except Exception:
                        continue
        return texts

    # ------------------------------------------------------------------
    # Source 2: Structured representations (from ATLAS parsed output)
    # ------------------------------------------------------------------
    def parsed_to_structured(self, parsed_repo: Dict[str, Dict]) -> List[str]:
        """
        Convert ATLAS parsed_repo.json into structured text.

        Format:
            [FILE] path/to/file.py
            [IMPORTS] os, sys, json
            [FUNC] function_name(args) -> calls: [a, b, c]
            [CLASS] ClassName inherits: [Base]
            [METHOD] method_name(self, args) -> calls: [x, y]
        """
        texts = []
        for file_path, data in parsed_repo.items():
            lines = [f"[FILE] {file_path}"]

            imports = data.get("imports", [])
            if imports:
                lines.append(f"[IMPORTS] {', '.join(imports)}")

            for func_name, func_data in data.get("functions", {}).items():
                args = ", ".join(func_data.get("args", []))
                calls = func_data.get("calls", [])
                call_str = f" -> calls: [{', '.join(calls)}]" if calls else ""
                lines.append(f"[FUNC] {func_name}({args}){call_str}")

            for class_name, class_data in data.get("classes", {}).items():
                bases = class_data.get("bases", [])
                base_str = f" inherits: [{', '.join(bases)}]" if bases else ""
                lines.append(f"[CLASS] {class_name}{base_str}")

                for method_name, method_data in class_data.get("methods", {}).items():
                    args = ", ".join(method_data.get("args", []))
                    calls = method_data.get("calls", [])
                    call_str = f" -> calls: [{', '.join(calls)}]" if calls else ""
                    lines.append(f"[METHOD] {method_name}({args}){call_str}")

            texts.append("\n".join(lines))
        return texts

    # ------------------------------------------------------------------
    # Source 3: Graph-annotated code (code with upstream/downstream context)
    # ------------------------------------------------------------------
    def annotate_with_graph(self, parsed_repo: Dict, graph) -> List[str]:
        """
        Add graph context as comments to code snippets.

        Format:
            # UPSTREAM: database.connect(), config.load()
            # DOWNSTREAM: api.respond(), logger.info()
            def process_request(req):
                ...
        """
        texts = []

        for file_path, data in parsed_repo.items():
            for func_name, func_data in data.get("functions", {}).items():
                node_id = f"{file_path}::{func_name}"
                code = func_data.get("code", "")
                if not code:
                    continue

                # Get upstream/downstream from graph
                upstream = []
                downstream = []
                if graph and node_id in graph.nodes:
                    upstream = [n for n in graph.predecessors(node_id)][:5]
                    downstream = [n for n in graph.successors(node_id)][:5]

                lines = []
                if upstream:
                    lines.append(f"# UPSTREAM: {', '.join(upstream)}")
                if downstream:
                    lines.append(f"# DOWNSTREAM: {', '.join(downstream)}")
                lines.append(code)
                texts.append("\n".join(lines))

            for class_name, class_data in data.get("classes", {}).items():
                for method_name, method_data in class_data.get("methods", {}).items():
                    node_id = f"{file_path}::{class_name}::{method_name}"
                    code = method_data.get("code", "")
                    if not code:
                        continue

                    upstream = []
                    downstream = []
                    if graph and node_id in graph.nodes:
                        upstream = [n for n in graph.predecessors(node_id)][:5]
                        downstream = [n for n in graph.successors(node_id)][:5]

                    lines = []
                    if upstream:
                        lines.append(f"# UPSTREAM: {', '.join(upstream)}")
                    if downstream:
                        lines.append(f"# DOWNSTREAM: {', '.join(downstream)}")
                    lines.append(code)
                    texts.append("\n".join(lines))

        return texts

    # ------------------------------------------------------------------
    # Mixing and tokenization
    # ------------------------------------------------------------------
    def prepare_dataset(
        self,
        code_dir: str,
        parsed_repo: Optional[Dict] = None,
        graph=None,
        mix_ratios: tuple = (0.6, 0.3, 0.1),
        output_path: str = "bdh/train_data.bin",
    ) -> str:
        """
        Prepare mixed training dataset.

        Args:
            code_dir: directory with Python source files
            parsed_repo: ATLAS parsed repo dict (for structured + annotated)
            graph: NetworkX graph (for annotated source)
            mix_ratios: (raw, structured, annotated) ratios
            output_path: where to save the tokenized data

        Returns:
            path to the saved binary file
        """
        import torch

        print("Collecting raw Python code...")
        raw_texts = self.collect_raw_code(code_dir)
        print(f"  {len(raw_texts)} raw code files")

        structured_texts = []
        annotated_texts = []

        if parsed_repo:
            print("Generating structured representations...")
            structured_texts = self.parsed_to_structured(parsed_repo)
            print(f"  {len(structured_texts)} structured entries")

            if graph:
                print("Generating graph-annotated code...")
                annotated_texts = self.annotate_with_graph(parsed_repo, graph)
                print(f"  {len(annotated_texts)} annotated entries")

        # Mix according to ratios
        all_texts = []
        r_raw, r_struct, r_annot = mix_ratios

        n_raw = int(len(raw_texts) * r_raw / (r_raw + 1e-9))
        n_struct = int(len(structured_texts) * r_struct / (r_struct + 1e-9))
        n_annot = int(len(annotated_texts) * r_annot / (r_annot + 1e-9))

        if raw_texts:
            all_texts.extend(random.sample(raw_texts, min(n_raw, len(raw_texts))))
        if structured_texts:
            all_texts.extend(random.sample(structured_texts, min(n_struct, len(structured_texts))))
        if annotated_texts:
            all_texts.extend(random.sample(annotated_texts, min(n_annot, len(annotated_texts))))

        random.shuffle(all_texts)

        print(f"Total training samples: {len(all_texts)}")

        # Tokenize everything into one big sequence
        all_ids = []
        for text in all_texts:
            ids = self.tokenizer.encode(text)
            all_ids.extend(ids)

        print(f"Total tokens: {len(all_ids):,}")

        # Save as binary tensor
        data = torch.tensor(all_ids, dtype=torch.long)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(data, output_path)
        print(f"Training data saved to {output_path} ({data.nbytes / 1e6:.1f} MB)")

        return output_path

    def load_dataset(self, path: str):
        """Load a pre-saved tokenized dataset."""
        import torch
        return torch.load(path)

    def get_batch(self, data, batch_size: int, device: str = "cpu"):
        """Get a random batch of (input, target) pairs."""
        import torch
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)
