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
from context.repo_structure import build_repo_structure, format_structure
from context.clustering import cluster_by_files, cluster_scc
from context.local_context import build_context_for_nodes
from context.summarizer import summarize_clusters
from context.global_summary import generate_global_summary
from utils.planner import plan_code,explain_repo
MEMORY_FILE = "data/memory_graph.json"

def initialize_repo(repo_url):
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
    repo_graph = builder.build_graph(parsed)
    print("Nodes:", repo_graph.number_of_nodes())
    print("Edges:", repo_graph.number_of_edges())

    graph_file = "data/repo_graph.pkl"
    with open(graph_file, "wb") as f:
        pickle.dump(repo_graph, f)
    print(f"Graph saved to {graph_file}")

    print("Building semantic memory...")
    collection = build_semantic_memmory(parsed)
    print(f"Collection info: {len(collection.get()['ids'])}")

    print("Loading or initializing Hebbian MemoryGraph...")
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = MemoryGraph.from_json(f.read())
        print("MemoryGraph loaded from previous session.")
    else:
        memory = MemoryGraph()
        print("Initialized new MemoryGraph.")

    # Add nodes and edges
    for node in repo_graph.nodes():
        memory.add_node(node, node_type=repo_graph.nodes[node].get("type", "concept"))
    for src, tgt, data in repo_graph.edges(data=True):
        memory.add_edge(
            src, tgt,
            weight=data.get("weight", 0.1),
            relation_type=data.get("relation", "related_to")
        )

    graph_store = GraphStore(repo_graph)
    retriever = CodeRetriever(collection=collection, graph_store=graph_store, top_k=5)

    structure_dict = build_repo_structure(repo_path)
    repo_structure = format_structure(structure_dict)

    return parsed, repo_graph, memory, retriever, graph_store, repo_structure

def explore_repo(parsed, graph_store, retriever, repo_structure):
    query = input("What do you want to know about the repo? ")
    results = retriever.retrieve(query)

    print("\nTop retrieved nodes:")
    for node, score in results[:10]:
        node_type = graph_store.graph.nodes[node].get("type")
        folder = graph_store.graph.nodes[node].get("folder")
        module = graph_store.graph.nodes[node].get("module")
        print(f"{node_type}: {node} (folder: {folder}, module: {module}, score: {score:.3f})")

    # Generate clusters & summaries
    top_nodes = [node for node, _ in results[:5]]
    snippets = {n: parsed.get("functions", {}).get(n, {}).get("code", "") for n in top_nodes}
    contexts = build_context_for_nodes(graph_store, top_nodes)
    clusters = cluster_by_files(top_nodes)
    cluster_summaries = summarize_clusters(clusters, contexts, snippets)
    global_summary = generate_global_summary(cluster_summaries, repo_structure)

    print("\n🧠 Cluster Summaries:")
    for c in cluster_summaries:
        print(c['cluster'], "->", c['summary'])
    print("\n🌍 Global Summary:\n", global_summary)
    answer = explain_repo(context, query)
    print("\n🧠 Answer:\n", answer)
    

def generate_code(context):
    query = input("Enter the new feature or code you want to add: ")
    new_code = plan_code(context, query)
    print("\nGenerated Code:\n", new_code)

if __name__ == "__main__":
    repo_url = input("Enter the Github Repository URL: ")
    parsed, repo_graph, memory, retriever, graph_store, repo_structure = initialize_repo(repo_url)

    context = {
        "repo_summary": generate_global_summary([], format_structure(build_repo_structure(repo_url))),
        "module_summaries": [],
        "repo_structure": repo_structure,
        "retrieved_code": {},
        "target_folder": "new_folder",
        "imports": []
    }

    while True:
        print("\n--- MENU ---")
        print("1. Explore / Understand Repository")
        print("2. Generate Code")
        print("3. Exit")
        choice = input("Choose an option (1-3): ")

        if choice == "1":
            explore_repo(parsed, graph_store, retriever, repo_structure)
        elif choice == "2":
            generate_code(context)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")