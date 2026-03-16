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
from context.clustering import cluster_by_files
from context.local_context import build_context_for_nodes
from context.summarizer import summarize_clusters
from context.global_summary import generate_global_summary
from utils.planner import plan_code, explain_repo
from utils.executor import execute_plan
from agent.agent_loop import AgentLoop

MEMORY_FILE = "data/memory_graph.json"
BDH_CHECKPOINT = "bdh/checkpoints/best.pt"
BDH_TOKENIZER = "bdh/tokenizer.json"
BDH_CONCEPT_MAP = "bdh/concept_map.json"


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

    print("Loading or initializing Hebbian MemoryGraph...")
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = MemoryGraph.from_json(f.read())
        print("MemoryGraph loaded from previous session.")
    else:
        memory = MemoryGraph()
        print("Initialized new MemoryGraph.")

    # Populate memory with code graph nodes and edges
    for node in repo_graph.nodes():
        memory.add_node(node, node_type=repo_graph.nodes[node].get("type", "concept"))
    for src, tgt, data in repo_graph.edges(data=True):
        memory.add_edge(
            src, tgt,
            weight=data.get("weight", 0.1),
            relation_type=data.get("relation", "related_to")
        )

    graph_store = GraphStore(repo_graph)
    retriever = CodeRetriever(
        collection=collection, graph_store=graph_store,
        top_k=5, memory_graph=memory
    )

    structure_dict = build_repo_structure(repo_path)
    repo_structure = format_structure(structure_dict)

    return parsed, repo_graph, memory, retriever, graph_store, repo_structure, repo_path


def load_bdh_components(retriever):
    """Load BDH model, tokenizer, and concept map if available."""
    bdh_router = None
    working_memory = None

    if not all(os.path.exists(p) for p in [BDH_CHECKPOINT, BDH_TOKENIZER, BDH_CONCEPT_MAP]):
        print("BDH components not found. Train BDH first (option 6).")
        return None, None

    try:
        import torch
        from bdh.train import load_checkpoint
        from bdh.tokenizer import CodeTokenizer
        from agent.bdh_router import BDHRouter
        from agent.working_memory import BDHWorkingMemory

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BDH model on {device}...")

        model, config = load_checkpoint(BDH_CHECKPOINT, device)
        model.eval()

        tokenizer = CodeTokenizer()
        tokenizer.load(BDH_TOKENIZER)

        with open(BDH_CONCEPT_MAP, "r") as f:
            concept_map = json.load(f)

        bdh_router = BDHRouter(
            bdh_model=model,
            tokenizer=tokenizer,
            concept_map=concept_map,
            retriever=retriever,
            device=device,
        )

        working_memory = BDHWorkingMemory(
            bdh_model=model,
            tokenizer=tokenizer,
            concept_map=concept_map,
            device=device,
        )

        print("BDH components loaded successfully!")
        print(f"  Model: {config.n_embd}d, {config.n_head} heads, {model.get_neuron_count()} neurons")
        print(f"  Concepts mapped: {len(concept_map)}")

    except Exception as e:
        print(f"Failed to load BDH: {e}")

    return bdh_router, working_memory


def train_bdh(repo_path, parsed, repo_graph):
    """Train BDH model on the current repository."""
    try:
        import torch
        from bdh.train import train_code_bdh
        from bdh.bdh import CodeBDHConfig, CodeBDHConfigSmall
        from bdh.tokenizer import CodeTokenizer
        from bdh.synapse_inspector import SynapseInspector

        print("\n" + "=" * 50)
        print("  BDH Training")
        print("=" * 50)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use smaller config on CPU to avoid OOM
        if device == "cpu":
            print("CPU detected - using small model config (~3M params)")
            config = CodeBDHConfigSmall()
        else:
            config = CodeBDHConfig()

        # Configuration
        iters = input("Training iterations (default 500): ").strip()
        max_iters = int(iters) if iters.isdigit() else 500

        print(f"\nConfig: {config.n_embd}d, {config.n_head} heads, {config.n_layer} layers")
        print(f"Neurons: {config.mlp_internal_dim_multiplier * config.n_embd // config.n_head * config.n_head}")

        # Train
        model = train_code_bdh(
            code_dir=repo_path,
            parsed_repo=parsed,
            graph=repo_graph,
            config=config,
            max_iters=max_iters,
        )

        # Build concept map
        print("\nBuilding concept map...")
        tokenizer = CodeTokenizer()
        tokenizer.load(BDH_TOKENIZER)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        inspector = SynapseInspector(model, tokenizer, device)
        concept_map = inspector.build_concept_map()
        inspector.save_map(concept_map, BDH_CONCEPT_MAP)
        inspector.print_report(concept_map)

        print("\nBDH training complete!")

    except ImportError as e:
        print(f"Missing dependency for BDH training: {e}")
        print("Install with: pip install torch tokenizers")
    except Exception as e:
        print(f"Training failed: {e}")


def explore_repo(parsed, graph_store, retriever, repo_structure, memory):
    query = input("What do you want to know about the repo? ")
    results = retriever.retrieve(query)

    print("\nTop retrieved nodes:")
    for node, score in results[:10]:
        node_type = graph_store.graph.nodes[node].get("type")
        folder = graph_store.graph.nodes[node].get("folder")
        module = graph_store.graph.nodes[node].get("module")
        print(f"{node_type}: {node} (folder: {folder}, module: {module}, score: {score:.3f})")

    top_nodes = [node for node, _ in results[:5]]

    snippets = {}
    for n in top_nodes:
        parts = n.split("::")
        if len(parts) >= 2:
            file_data = parsed.get(parts[0], {})
            if len(parts) == 2:
                code = file_data.get("functions", {}).get(parts[1], {}).get("code", "")
            elif len(parts) == 3:
                code = (file_data.get("classes", {})
                        .get(parts[1], {})
                        .get("methods", {})
                        .get(parts[2], {})
                        .get("code", ""))
            else:
                code = ""
            if code:
                snippets[n] = code

    contexts = build_context_for_nodes(graph_store, top_nodes)
    clusters = cluster_by_files(top_nodes)
    cluster_summaries = summarize_clusters(clusters, contexts, snippets)
    global_summary = generate_global_summary(cluster_summaries, repo_structure)

    print("\n Cluster Summaries:")
    for c in cluster_summaries:
        print(c['cluster'], "->", c['summary'])
    print("\n Global Summary:\n", global_summary)

    folders = [graph_store.graph.nodes[n].get("folder") for n in top_nodes
               if graph_store.graph.nodes[n].get("folder")]
    target_folder = max(set(folders), key=folders.count) if folders else "new_folder"

    import_nodes = set()
    for n in top_nodes:
        import_nodes.update(graph_store.get_full_upstream(n, types=["file", "module"]))

    memory_summary = memory.get_summary(max_nodes=10)

    memory.update_from_concepts(
        concepts=top_nodes,
        context_node=f"query:{query[:50]}"
    )
    memory.decay_all()

    context = {
        "repo_summary": global_summary,
        "module_summaries": cluster_summaries,
        "repo_structure": repo_structure,
        "retrieved_code": snippets,
        "target_folder": target_folder,
        "imports": list(import_nodes),
        "memory_summary": memory_summary,
    }
    answer = explain_repo(context, query)
    print("\n Answer:\n", answer)
    return context


def generate_code(context):
    query = input("Enter the new feature or code you want to add: ")
    new_code = plan_code(context, query)
    print("\nGenerated Code:\n", new_code)

    save = input("Save to disk? (y/n): ").strip().lower()
    if save == "y":
        file_name = input("File name (default: new_feature.py): ").strip() or "new_feature.py"
        target = context.get("target_folder", "new_folder")
        path = execute_plan(new_code, target, file_name)
        print(f"Code saved to {path}")


if __name__ == "__main__":
    repo_url = input("Enter the Github Repository URL: ")
    parsed, repo_graph, memory, retriever, graph_store, repo_structure, repo_path = initialize_repo(repo_url)

    # Try loading BDH components
    bdh_router, working_memory = load_bdh_components(retriever)

    context = {
        "repo_summary": "",
        "module_summaries": [],
        "repo_structure": repo_structure,
        "retrieved_code": {},
        "target_folder": "new_folder",
        "imports": [],
        "memory_summary": "",
    }

    agent = AgentLoop(
        retriever=retriever, graph_store=graph_store,
        memory=memory, parsed=parsed, repo_structure=repo_structure,
        bdh_router=bdh_router, working_memory=working_memory,
    )

    while True:
        print("\n--- MENU ---")
        print("1. Explore / Understand Repository")
        print("2. Generate Code (single-shot)")
        print("3. Agent Mode (autonomous multi-step)")
        print("4. Memory Stats")
        print("5. Train BDH Model" + (" [TRAINED]" if bdh_router else ""))
        print("6. Load BDH Components" + (" [LOADED]" if bdh_router else ""))
        print("7. Exit")
        choice = input("Choose an option (1-7): ")

        if choice == "1":
            result = explore_repo(parsed, graph_store, retriever, repo_structure, memory)
            if result:
                context.update(result)
        elif choice == "2":
            generate_code(context)
        elif choice == "3":
            task = input("Describe the feature or task: ")
            auto = input("Auto-save generated files? (y/n): ").strip().lower() == "y"
            dash = input("Show dashboard? (y/n): ").strip().lower() == "y"
            agent.run(task, auto_save=auto, show_dashboard=dash)
        elif choice == "4":
            stats = memory.get_stats()
            print(f"\nMemory Stats:")
            print(f"  Nodes: {stats['node_count']}")
            print(f"  Edges: {stats['edge_count']}")
            print(f"  Total Hebbian updates: {stats['total_updates']}")
            print(f"  Edges pruned: {stats['edges_pruned']}")
            print(f"  Graph density: {stats['density']:.4f}")
            print(f"\nTop concepts (by PageRank):")
            print(memory.get_summary(max_nodes=5))
            if bdh_router:
                print("\nBDH Status: ACTIVE")
            else:
                print("\nBDH Status: Not loaded")
        elif choice == "5":
            train_bdh(repo_path, parsed, repo_graph)
            bdh_router, working_memory = load_bdh_components(retriever)
            agent.bdh_router = bdh_router
            agent.working_memory = working_memory
        elif choice == "6":
            bdh_router, working_memory = load_bdh_components(retriever)
            agent.bdh_router = bdh_router
            agent.working_memory = working_memory
        elif choice == "7":
            os.makedirs("data", exist_ok=True)
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write(memory.to_json())
            print(f"Memory saved to {MEMORY_FILE}")
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")
