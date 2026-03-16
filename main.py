import os
import json
import pickle
from parser.repo_cloner import RepoCloner
from parser.repo_parser import CodeParser
from graph.graph_builder import CodeGraphBuilder
from semantic.embeddings import build_semantic_memmory
from semantic.retrieval import CodeRetriever
from graph.graph_store import GraphStore
from graph.memory_graph import MemoryGraph
from utils.context_builder import build_context
from utils.executor import execute_plan
from utils.planner import plan_code

MEMORY_FILE = "data/memory_graph.json"

if __name__ == "__main__":
    repo_url = input("Enter the Github Repository URL: ")
    print("Cloning repository...")
    cloner = RepoCloner()
    repo_path = cloner.clone_repo(repo_url)

    print("Parsing repository...")
    parser = CodeParser()
    parsed = parser.parse_repository(repo_path)

    os.makedirs("data", exist_ok=True)
    parsed_file = os.path.join("data", "parsed_repo.json")
    with open(parsed_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"Parsed repo saved to {parsed_file}")

    print("Building repository graph...")
    builder = CodeGraphBuilder()
    repo_graph = builder.build_graph(parsed)  # just parsed, not parsed["files"]
    print("Nodes:", repo_graph.number_of_nodes())
    print("Edges:", repo_graph.number_of_edges())

    graph_file = "data/repo_graph.pkl"
    with open(graph_file, "wb") as f:
        pickle.dump(repo_graph, f)
    print(f"Graph saved to {graph_file}")

    print("Building semantic memory...")
    collection = build_semantic_memmory(parsed)  # just parsed, not parsed["files"]
    print(f"Collection info: {len(collection.get()['ids'])}")

    print("Loading or initializing Hebbian MemoryGraph...")
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = MemoryGraph.from_json(f.read())
        print("MemoryGraph loaded from previous session.")
    else:
        memory = MemoryGraph()
        print("Initialized new MemoryGraph.")

    # Add all nodes from repo graph to MemoryGraph
    for node in repo_graph.nodes():
        memory.add_node(node, node_type=repo_graph.nodes[node].get("type", "concept"))

    # Hebbian update: strengthen edges based on existing repo graph
    for source, target, data in repo_graph.edges(data=True):
        memory.add_edge(
            source,
            target,
            weight=data.get("weight", 0.1),
            relation_type=data.get("relation", "related_to")
        )

    # Initialize retriever with semantic memory
    graph_store = GraphStore(repo_graph)
    retriever = CodeRetriever(collection=collection, graph_store=graph_store, top_k=5)

    # ----------------------
    # Example query
    # ----------------------
    query = input("Enter your query: ")
    results = retriever.retrieve(query)

    print("\nTop retrieved nodes (combined semantic + Hebbian):")
    for node, score in results[:10]:
        node_type = graph_store.graph.nodes[node].get("type")
        folder = graph_store.graph.nodes[node].get("folder")
        module = graph_store.graph.nodes[node].get("module")
        print(f"{node_type}: {node} (folder: {folder}, module: {module}, score: {score:.3f})")

    # ----------------------
    # Auto-suggest folder and imports for new code
    # ----------------------
    top_nodes = [node for node, _ in results[:5]]
    folders = [graph_store.graph.nodes[n].get("folder") for n in top_nodes if graph_store.graph.nodes[n].get("folder")]
    target_folder = max(set(folders), key=folders.count) if folders else "new_folder"
    print("Suggested folder for new code:", target_folder)

    import_nodes = set()
    for node in top_nodes:
        import_nodes.update(graph_store.get_full_upstream(node, types=["file", "module"]))
    print("Suggested imports for new code:", import_nodes)

    # Save updated MemoryGraph
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write(memory.to_json())
    print(f"MemoryGraph saved to {MEMORY_FILE}")

    context=build_context(retriever,graph_store,query)
    new_code=plan_code(context,query)
    new_file_path=execute_plan(new_code,context['target_folder'],file_name="new_feature.py")
    print(f"New code created at: {new_file_path}")

