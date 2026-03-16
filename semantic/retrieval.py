from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from graph.graph_store import GraphStore
from typing import List
import numpy as np

class CodeRetriever:
    def __init__(self, collection: Collection, graph_store: GraphStore, top_k: int = 5,
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.collection = collection
        self.graph_store = graph_store
        self.top_k = top_k
        self.embed_model = SentenceTransformer(embed_model_name)

    def _build_node_id(self, metadata: dict) -> str:
        """Reconstruct full node ID to match graph builder naming."""
        node_type = metadata.get("type")
        path = metadata.get("path")
        name = metadata.get("name")

        if node_type == "function":
            return f"{path}::{name}"
        elif node_type == "class":
            return f"{path}::{name}"
        elif node_type == "method":
            # For methods, path should already include class name
            return f"{path}::{name}"
        else:
            return path  # fallback for files/modules

    def retrieve(self, query: str, expand_depth: int = 2,
                 node_types: List[str] = ["function", "method", "class"]):
        # Encode the query
        query_vec = self.embed_model.encode(query).tolist()

        # Query Chroma collection
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k
        )

        sem_nodes = []
        sem_scores = {}

        # Process each returned document
        for i, metadata in enumerate(results['metadatas'][0]):
            node_id = self._build_node_id(metadata)
            node_type = metadata.get("type")

            if node_type in node_types and node_id in self.graph_store.graph.nodes:
                sem_nodes.append(node_id)

                # If Chroma provides distances, use them
                if results['distances'] and results['distances'][0]:
                    distance = results['distances'][0][i]
                    sem_scores[node_id] = 1 - distance  # cosine distance -> similarity
                else:
                    # Fallback: compute cosine similarity manually
                    node_embedding_data = self.collection.get(ids=[results['ids'][0][i]])
                    node_embedding = np.array(node_embedding_data['embeddings'][0][0])
                    query_vec_np = np.array(query_vec)
                    cos_sim = np.dot(query_vec_np, node_embedding) / (
                        np.linalg.norm(query_vec_np) * np.linalg.norm(node_embedding)
                    )
                    sem_scores[node_id] = float(cos_sim)

        # Expand nodes in the graph (upstream and downstream)
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

        # Combine results with similarity scores (default 0.5 if missing)
        combined_results = [(node, sem_scores.get(node, 0.5)) for node in expanded_nodes]
        combined_results.sort(key=lambda x: x[1], reverse=True)

        return combined_results