"""
Interpretability Dashboard — real-time visibility into agent reasoning.

Shows BDH synapse activations, MemoryGraph hot paths, and retrieval context
in a terminal-friendly format.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    os.system("")

class C:
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    HEADER = "\033[1;96m"
    BOX = "\033[90m"  # gray for box drawing


class Dashboard:
    """Terminal-based interpretability dashboard for the ATLAS+BDH agent."""

    def __init__(self, memory_graph=None, working_memory=None, bdh_router=None):
        self.memory_graph = memory_graph
        self.working_memory = working_memory
        self.bdh_router = bdh_router

    def render(
        self,
        task: str = "",
        subtask: str = "",
        subtask_idx: int = 0,
        total_subtasks: int = 0,
        retrieved: List[Tuple[str, float]] = None,
        phase: str = "",
    ) -> str:
        """Render the full dashboard as a string."""
        W = 60
        lines = []

        # Header
        lines.append(f"{C.HEADER}{'=' * W}")
        lines.append(f"  ATLAS + BDH  |  Agent Dashboard")
        lines.append(f"{'=' * W}{C.RST}")

        # Current task
        if task:
            lines.append(f"\n  {C.WHITE}Task:{C.RST} {task}")
        if subtask:
            lines.append(f"  {C.WHITE}Subtask {subtask_idx}/{total_subtasks}:{C.RST} {subtask[:45]}...")
        if phase:
            lines.append(f"  {C.WHITE}Phase:{C.RST} {C.CYAN}{phase}{C.RST}")

        # BDH Working Memory
        if self.working_memory:
            lines.append(f"\n{C.BOX}{'─' * W}{C.RST}")
            lines.append(f"  {C.MAGENTA}BDH Working Memory{C.RST}")
            lines.append(f"{C.BOX}{'─' * W}{C.RST}")
            concepts = self.working_memory.get_active_concepts(top_k=6)
            if concepts:
                for name, score in concepts:
                    bar = _bar(score, 20)
                    lines.append(f"  {C.WHITE}{name:25s}{C.RST} {C.CYAN}{bar}{C.RST} {C.DIM}{score:.3f}{C.RST}")
            else:
                lines.append(f"  {C.DIM}(empty — no steps processed yet){C.RST}")

        # BDH Router (concept routing for current query)
        if self.bdh_router and subtask:
            lines.append(f"\n{C.BOX}{'─' * W}{C.RST}")
            lines.append(f"  {C.MAGENTA}BDH Concept Routing{C.RST}")
            lines.append(f"{C.BOX}{'─' * W}{C.RST}")
            try:
                concepts = self.bdh_router.get_active_concepts(subtask, top_k=5)
                for name, score in concepts:
                    bar = _bar(score, 20)
                    lines.append(f"  {C.WHITE}{name:25s}{C.RST} {C.GREEN}{bar}{C.RST} {C.DIM}{score:.3f}{C.RST}")
            except Exception:
                lines.append(f"  {C.DIM}(BDH model not available){C.RST}")

        # MemoryGraph
        if self.memory_graph and self.memory_graph.node_count > 0:
            lines.append(f"\n{C.BOX}{'─' * W}{C.RST}")
            lines.append(f"  {C.MAGENTA}Hebbian Memory (Hot Paths){C.RST}")
            lines.append(f"{C.BOX}{'─' * W}{C.RST}")
            lines.append(f"  {C.DIM}Nodes: {self.memory_graph.node_count}  "
                         f"Edges: {self.memory_graph.edge_count}  "
                         f"Updates: {self.memory_graph.stats['total_updates']}{C.RST}")

            # Show top paths
            try:
                import networkx as nx
                pagerank = nx.pagerank(self.memory_graph.graph, alpha=0.85)
                top = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
                for node_id, rank in top:
                    # Show strongest outgoing edge
                    edges = [
                        (tgt, self.memory_graph.graph[node_id][tgt].get("weight", 0))
                        for tgt in self.memory_graph.graph.successors(node_id)
                    ]
                    if edges:
                        edges.sort(key=lambda x: x[1], reverse=True)
                        best_tgt, best_w = edges[0]
                        short_src = _shorten(node_id, 25)
                        short_tgt = _shorten(best_tgt, 25)
                        lines.append(f"  {C.CYAN}{short_src}{C.RST} {C.YELLOW}--({best_w:.2f})-->{C.RST} {C.CYAN}{short_tgt}{C.RST}")
                    else:
                        lines.append(f"  {C.CYAN}{_shorten(node_id, 50)}{C.RST} {C.DIM}(rank: {rank:.4f}){C.RST}")
            except Exception:
                lines.append(f"  {C.DIM}(PageRank unavailable){C.RST}")

        # Retrieved Context
        if retrieved:
            lines.append(f"\n{C.BOX}{'─' * W}{C.RST}")
            lines.append(f"  {C.MAGENTA}Retrieved Context{C.RST}")
            lines.append(f"{C.BOX}{'─' * W}{C.RST}")
            for node, score in retrieved[:8]:
                lines.append(f"  {C.WHITE}{_shorten(node, 45)}{C.RST} {C.DIM}(score: {score:.3f}){C.RST}")

        lines.append(f"\n{C.HEADER}{'=' * W}{C.RST}")
        return "\n".join(lines)

    def print_dashboard(self, **kwargs):
        """Print the dashboard to terminal."""
        print(self.render(**kwargs))


def _bar(value: float, width: int = 20) -> str:
    """Render a simple progress bar with color."""
    filled = int(value * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def _shorten(text: str, max_len: int) -> str:
    """Shorten a string, keeping the end (usually the function name)."""
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3):]
