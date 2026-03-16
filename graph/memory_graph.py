import json
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx


@dataclass
class NodeData:
    """Data associated with a graph node (concept/artifact)."""
    activation: float = 1.0
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    node_type: str = "concept"  # concept, code, task, fact, entity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "activation": self.activation,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "node_type": self.node_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NodeData":
        return cls(
            activation=data.get("activation", 1.0),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            node_type=data.get("node_type", "concept"),
            metadata=data.get("metadata", {})
        )


@dataclass
class EdgeData:
    """Data associated with a graph edge (relation/synapse)."""
    weight: float = 0.1
    relation_type: str = "related_to"  # related_to, depends_on, part_of, causes, etc.
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    co_activation_count: int = 1
    
    def to_dict(self) -> dict:
        return {
            "weight": self.weight,
            "relation_type": self.relation_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "co_activation_count": self.co_activation_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EdgeData":
        return cls(
            weight=data.get("weight", 0.1),
            relation_type=data.get("relation_type", "related_to"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            co_activation_count=data.get("co_activation_count", 1)
        )


class MemoryGraph:
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        decay_rate: float = 0.01,
        activation_threshold: float = 0.3,
        max_nodes: int = 1000,
        max_edges: int = 5000,
        prune_threshold: float = 0.01
    ):
        self.graph = nx.DiGraph()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.prune_threshold = prune_threshold
        
        # Statistics
        self.stats = {
            "total_updates": 0,
            "nodes_added": 0,
            "edges_added": 0,
            "edges_pruned": 0
        }
    
    @property
    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return self.graph.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return self.graph.number_of_edges()
    
    def add_node(
        self,
        node_id: str,
        activation: float = 1.0,
        node_type: str = "concept",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            activation: Initial activation level
            node_type: Type of node (concept, code, task, etc.)
            confidence: Confidence score for this node
            metadata: Additional metadata
            
        Returns:
            True if node was added, False if it already exists or limit reached
        """
        if self.graph.has_node(node_id):
            # Update existing node
            self._update_node_activation(node_id, activation)
            return False
        
        if self.node_count >= self.max_nodes:
            self._prune_least_important_nodes()
        
        node_data = NodeData(
            activation=activation,
            confidence=confidence,
            node_type=node_type,
            metadata=metadata or {}
        )
        
        self.graph.add_node(node_id, **node_data.to_dict())
        self.stats["nodes_added"] += 1
        return True
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 0.1,
        relation_type: str = "related_to"
    ) -> bool:
        """
        Add an edge (synapse) between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Initial connection weight
            relation_type: Type of relation
            
        Returns:
            True if edge was added/updated
        """
        # Ensure nodes exist
        if not self.graph.has_node(source):
            self.add_node(source)
        if not self.graph.has_node(target):
            self.add_node(target)
        
        if self.graph.has_edge(source, target):
            # Apply Hebbian update
            self._hebbian_update(source, target, weight)
            return False
        
        if self.edge_count >= self.max_edges:
            self._prune_weak_edges()
        
        edge_data = EdgeData(weight=weight, relation_type=relation_type)
        self.graph.add_edge(source, target, **edge_data.to_dict())
        self.stats["edges_added"] += 1
        return True
    
    def _hebbian_update(self, source: str, target: str, delta: float = None):
        """
        Apply Hebbian plasticity rule: "Neurons that fire together wire together"
        
        Update rule: weight += η * activation_source * activation_target
        
        Args:
            source: Source node ID
            target: Target node ID
            delta: Optional weight delta override
        """
        source_activation = self.graph.nodes[source].get("activation", 1.0)
        target_activation = self.graph.nodes[target].get("activation", 1.0)
        
        if source_activation < self.activation_threshold:
            return
        
        # Calculate weight update
        if delta is None:
            delta = self.learning_rate * source_activation * target_activation
        
        current_weight = self.graph[source][target].get("weight", 0.1)
        new_weight = min(current_weight + delta, 1.0)  # Cap at 1.0
        
        self.graph[source][target]["weight"] = new_weight
        self.graph[source][target]["updated_at"] = datetime.now().isoformat()
        self.graph[source][target]["co_activation_count"] = \
            self.graph[source][target].get("co_activation_count", 1) + 1
        
        self.stats["total_updates"] += 1
    
    def _update_node_activation(self, node_id: str, new_activation: float):
        """Update a node's activation level with smoothing."""
        current = self.graph.nodes[node_id].get("activation", 1.0)
        # Exponential moving average
        updated = 0.7 * current + 0.3 * new_activation
        self.graph.nodes[node_id]["activation"] = updated
        self.graph.nodes[node_id]["updated_at"] = datetime.now().isoformat()
    
    def decay_all(self):
        """Apply decay to all edges and prune weak ones."""
        edges_to_remove = []
        
        for source, target in self.graph.edges():
            current_weight = self.graph[source][target].get("weight", 0.1)
            new_weight = current_weight * (1 - self.decay_rate)
            
            if new_weight < self.prune_threshold:
                edges_to_remove.append((source, target))
            else:
                self.graph[source][target]["weight"] = new_weight
        
        for source, target in edges_to_remove:
            self.graph.remove_edge(source, target)
            self.stats["edges_pruned"] += 1
    
    def _prune_weak_edges(self, count: int = 100):
        """Remove weakest edges to make room for new ones."""
        edges_with_weights = [
            (s, t, self.graph[s][t].get("weight", 0))
            for s, t in self.graph.edges()
        ]
        edges_with_weights.sort(key=lambda x: x[2])
        
        for source, target, _ in edges_with_weights[:count]:
            self.graph.remove_edge(source, target)
            self.stats["edges_pruned"] += 1
    
    def _prune_least_important_nodes(self, count: int = 50):
        """Remove least important nodes based on pagerank."""
        if self.node_count == 0:
            return
        
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1])
            
            for node_id, _ in sorted_nodes[:count]:
                self.graph.remove_node(node_id)
        except:
            # Fallback: remove nodes with lowest activation
            sorted_nodes = sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get("activation", 0)
            )
            for node_id, _ in sorted_nodes[:count]:
                self.graph.remove_node(node_id)
    
    def update_from_concepts(
        self,
        concepts: List[str],
        relations: Optional[List[Tuple[str, str, str]]] = None,
        context_node: Optional[str] = None
    ):
        """
        Update graph from extracted concepts and relations.
        
        Args:
            concepts: List of concept strings
            relations: Optional list of (source, target, relation_type) tuples
            context_node: Optional context node to link all concepts to
        """
        # Add all concepts as nodes
        for concept in concepts:
            self.add_node(concept, node_type="concept")
        
        # Add explicit relations
        if relations:
            for source, target, rel_type in relations:
                self.add_edge(source, target, relation_type=rel_type)
        
        # Create implicit relations between co-occurring concepts
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                self.add_edge(concept1, concept2, weight=0.05, relation_type="co_occurs")
                self.add_edge(concept2, concept1, weight=0.05, relation_type="co_occurs")
        
        # Link to context if provided
        if context_node:
            if not self.graph.has_node(context_node):
                self.add_node(context_node, node_type="task")
            for concept in concepts:
                self.add_edge(context_node, concept, relation_type="context")
    
    def get_related_nodes(
        self,
        node_id: str,
        depth: int = 2,
        min_weight: float = 0.05,
        max_results: int = 20
    ) -> List[Tuple[str, float, str]]:
        """
        Get nodes related to a given node.
        
        Args:
            node_id: Starting node ID
            depth: How many hops to traverse
            min_weight: Minimum edge weight to follow
            max_results: Maximum number of results
            
        Returns:
            List of (node_id, relevance_score, node_type) tuples
        """
        if not self.graph.has_node(node_id):
            return []
        
        visited = {node_id: 1.0}
        frontier = [(node_id, 1.0)]
        
        for _ in range(depth):
            new_frontier = []
            for current, current_score in frontier:
                for neighbor in self.graph.successors(current):
                    edge_weight = self.graph[current][neighbor].get("weight", 0)
                    if edge_weight < min_weight:
                        continue
                    
                    new_score = current_score * edge_weight
                    if neighbor not in visited or visited[neighbor] < new_score:
                        visited[neighbor] = new_score
                        new_frontier.append((neighbor, new_score))
            
            frontier = new_frontier
        
        # Sort by relevance and return
        results = [
            (nid, score, self.graph.nodes[nid].get("node_type", "concept"))
            for nid, score in visited.items()
            if nid != node_id
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:max_results]
    
    def get_summary(
        self,
        max_nodes: int = 15,
        max_edges_per_node: int = 5,
        include_weights: bool = True
    ) -> str:
        """
        Get a text summary of the graph for prompt injection.
        
        Args:
            max_nodes: Maximum nodes to include
            max_edges_per_node: Maximum edges per node
            include_weights: Include edge weights in summary
            
        Returns:
            Formatted string summary
        """
        if self.node_count == 0:
            return "No prior state. This is a fresh session."
        
        # Get most important nodes by pagerank
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        except:
            # Fallback to activation-based ranking
            top_nodes = sorted(
                [(n, d.get("activation", 0)) for n, d in self.graph.nodes(data=True)],
                key=lambda x: x[1],
                reverse=True
            )[:max_nodes]
        
        lines = ["## Prior Knowledge State (Memory Graph)"]
        lines.append(f"Total: {self.node_count} concepts, {self.edge_count} relations\n")
        
        for node_id, importance in top_nodes:
            node_data = self.graph.nodes[node_id]
            node_type = node_data.get("node_type", "concept")
            activation = node_data.get("activation", 1.0)
            
            lines.append(f"### {node_id} [{node_type}] (relevance: {importance:.3f})")
            
            # Get top edges
            edges = [
                (tgt, self.graph[node_id][tgt].get("weight", 0),
                 self.graph[node_id][tgt].get("relation_type", "related_to"))
                for tgt in self.graph.successors(node_id)
            ]
            edges.sort(key=lambda x: x[1], reverse=True)
            
            for target, weight, rel_type in edges[:max_edges_per_node]:
                if include_weights:
                    lines.append(f"  → {target} ({rel_type}, w={weight:.3f})")
                else:
                    lines.append(f"  → {target} ({rel_type})")
        
        return "\n".join(lines)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Simple keyword search in the graph.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (node_id, score) tuples
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for node_id in self.graph.nodes():
            node_lower = node_id.lower()
            
            # Exact match
            if query_lower == node_lower:
                results.append((node_id, 1.0))
            # Contains query
            elif query_lower in node_lower:
                results.append((node_id, 0.8))
            # Word overlap
            else:
                node_words = set(node_lower.replace("_", " ").replace("-", " ").split())
                overlap = len(query_words & node_words) / max(len(query_words), 1)
                if overlap > 0:
                    results.append((node_id, overlap * 0.6))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def to_dict(self) -> dict:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {"id": n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ],
            "edges": [
                {"source": s, "target": t, **self.graph[s][t]}
                for s, t in self.graph.edges()
            ],
            "config": {
                "learning_rate": self.learning_rate,
                "decay_rate": self.decay_rate,
                "activation_threshold": self.activation_threshold,
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges,
                "prune_threshold": self.prune_threshold
            },
            "stats": self.stats
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGraph":
        """Deserialize graph from dictionary."""
        config = data.get("config", {})
        graph = cls(
            learning_rate=config.get("learning_rate", 0.05),
            decay_rate=config.get("decay_rate", 0.01),
            activation_threshold=config.get("activation_threshold", 0.3),
            max_nodes=config.get("max_nodes", 1000),
            max_edges=config.get("max_edges", 5000),
            prune_threshold=config.get("prune_threshold", 0.01)
        )
        
        # Add nodes
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            graph.graph.add_node(node_id, **node)
        
        # Add edges
        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            graph.graph.add_edge(source, target, **edge)
        
        # Restore stats
        graph.stats = data.get("stats", graph.stats)
        
        return graph
    
    def to_json(self) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "MemoryGraph":
        """Deserialize graph from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def clear(self):
        """Clear the entire graph."""
        self.graph.clear()
        self.stats = {
            "total_updates": 0,
            "nodes_added": 0,
            "edges_added": 0,
            "edges_pruned": 0
        }
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            **self.stats,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": nx.density(self.graph) if self.node_count > 0 else 0
        }
