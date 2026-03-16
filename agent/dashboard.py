"""
Interpretability Dashboard — real-time visibility into agent reasoning.

Shows BDH synapse activations, MemoryGraph hot paths, and retrieval context
in a terminal-friendly format.
"""

from typing import Dict, List, Tuple, Optional


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
        lines.append("=" * W)
        lines.append("  ATLAS + BDH  |  Agent Dashboard")
        lines.append("=" * W)

        # Current task
        if task:
            lines.append(f"\n  Task: {task}")
        if subtask:
            lines.append(f"  Subtask {subtask_idx}/{total_subtasks}: {subtask}")
        if phase:
            lines.append(f"  Phase: {phase}")

        # BDH Working Memory
        if self.working_memory:
            lines.append(f"\n{'─' * W}")
            lines.append("  BDH Working Memory")
            lines.append(f"{'─' * W}")
            concepts = self.working_memory.get_active_concepts(top_k=6)
            if concepts:
                for name, score in concepts:
                    bar = _bar(score, 20)
                    lines.append(f"  {name:25s} {bar} {score:.3f}")
            else:
                lines.append("  (empty — no steps processed yet)")

        # BDH Router (concept routing for current query)
        if self.bdh_router and subtask:
            lines.append(f"\n{'─' * W}")
            lines.append("  BDH Concept Routing")
            lines.append(f"{'─' * W}")
            try:
                concepts = self.bdh_router.get_active_concepts(subtask, top_k=5)
                for name, score in concepts:
                    bar = _bar(score, 20)
                    lines.append(f"  {name:25s} {bar} {score:.3f}")
            except Exception:
                lines.append("  (BDH model not available)")

        # MemoryGraph
        if self.memory_graph and self.memory_graph.node_count > 0:
            lines.append(f"\n{'─' * W}")
            lines.append("  Hebbian Memory (Hot Paths)")
            lines.append(f"{'─' * W}")
            lines.append(f"  Nodes: {self.memory_graph.node_count}  "
                         f"Edges: {self.memory_graph.edge_count}  "
                         f"Updates: {self.memory_graph.stats['total_updates']}")

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
                        lines.append(f"  {short_src} --({best_w:.2f})--> {short_tgt}")
                    else:
                        lines.append(f"  {_shorten(node_id, 50)} (rank: {rank:.4f})")
            except Exception:
                lines.append("  (PageRank unavailable)")

        # Retrieved Context
        if retrieved:
            lines.append(f"\n{'─' * W}")
            lines.append("  Retrieved Context")
            lines.append(f"{'─' * W}")
            for node, score in retrieved[:8]:
                lines.append(f"  {_shorten(node, 45)} (score: {score:.3f})")

        lines.append("\n" + "=" * W)
        return "\n".join(lines)

    def print_dashboard(self, **kwargs):
        """Print the dashboard to terminal."""
        print(self.render(**kwargs))


def _bar(value: float, width: int = 20) -> str:
    """Render a simple progress bar."""
    filled = int(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _shorten(text: str, max_len: int) -> str:
    """Shorten a string, keeping the end (usually the function name)."""
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3):]
