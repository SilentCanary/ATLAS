from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from graph.graph_store import GraphStore
from typing import List, Optional


class CodeRetriever:
    def __init__(self, collection: Collection, graph_store: GraphStore, top_k: int = 5,
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 memory_graph=None):
        self.collection = collection
        self.graph_store = graph_store
        self.top_k = top_k
        self.embed_model = SentenceTransformer(embed_model_name)
        self.memory_graph = memory_graph

    def _build_node_id(self, metadata: dict) -> str:
        """Reconstruct full node ID to match graph builder naming."""
        node_type = metadata.get("type")
        path = metadata.get("path")
        name = metadata.get("name")
        class_name = metadata.get("class_name")

        if node_type == "method" and class_name:
            return f"{path}::{class_name}::{name}"
        elif node_type in ("function", "class"):
            return f"{path}::{name}"
        else:
            return path  # fallback for files/modules

    def retrieve(self, query: str, expand_depth: int = 2,
                 node_types: List[str] = ["function", "method", "class"]):
        query_vec = self.embed_model.encode(query).tolist()

        # Request more results than top_k so file-type docs don't consume all slots
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k * 3,
            include=["metadatas", "distances"]
        )

        sem_nodes = []
        sem_scores = {}

        for i, metadata in enumerate(results['metadatas'][0]):
            node_id = self._build_node_id(metadata)
            node_type = metadata.get("type")

            if node_type not in node_types:
                continue
            if node_id not in self.graph_store.graph.nodes:
                continue
            if node_id in sem_scores:
                continue  # deduplicate

            sem_nodes.append(node_id)
            # cosine distance from ChromaDB: distance = 1 - cosine_sim
            distance = results['distances'][0][i]
            sem_scores[node_id] = max(0.0, 1.0 - distance)

        # Merge Hebbian memory signals if available
        if self.memory_graph:
            for node in sem_nodes[:self.top_k]:
                related = self.memory_graph.get_related_nodes(
                    node, depth=2, min_weight=0.05, max_results=10
                )
                for rel_node, relevance, _ in related:
                    if rel_node in self.graph_store.graph.nodes:
                        node_t = self.graph_store.graph.nodes[rel_node].get("type")
                        if node_t in node_types and rel_node not in sem_scores:
                            # Hebbian score = 0.3 * relevance (lower weight than semantic)
                            sem_scores[rel_node] = 0.3 * relevance
                            sem_nodes.append(rel_node)

        # Expand nodes in the graph (upstream and downstream) with depth limit
        expanded_nodes = set()
        for node in sem_nodes:
            downstream = self.graph_store.get_full_downstream(node)
            upstream = self.graph_store.get_full_upstream(node)
            expanded_nodes.update(downstream)
            expanded_nodes.update(upstream)
        expanded_nodes.update(sem_nodes)

        # Filter by node types if requested
        if node_types:
            expanded_nodes = [
                n for n in expanded_nodes
                if self.graph_store.graph.nodes[n].get("type") in node_types
            ]

        # Graph-expanded nodes get a decaying score based on distance from semantic hits
        combined_results = []
        for node in expanded_nodes:
            if node in sem_scores:
                combined_results.append((node, sem_scores[node]))
            else:
                # Lower default for expanded nodes so semantic hits rank higher
                combined_results.append((node, 0.2))

        combined_results.sort(key=lambda x: x[1], reverse=True)

        return combined_results
