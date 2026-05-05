import os
import sys
import json
import pickle
from collections import Counter
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
from utils.executor import execute_plan_bundle, parse_code_bundle
from agent.agent_loop import AgentLoop
from agent.validator import CodeValidator

# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    os.system("")  # enable ANSI escape codes on Windows

class C:
    """ANSI color codes."""
    RST    = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    # Foreground
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"
    # Special combos
    HEADER = "\033[1;96m"   # bold cyan
    OK     = "\033[92m"     # green
    WARN   = "\033[93m"     # yellow
    FAIL   = "\033[91m"     # red
    ACCENT = "\033[1;95m"   # bold magenta
    INFO   = "\033[94m"     # blue
    TITLE  = "\033[1;97m"   # bold white


MEMORY_FILE = "data/memory_graph.json"
BDH_CHECKPOINT = "bdh/checkpoints/best.pt"
BDH_TOKENIZER = "bdh/tokenizer.json"
BDH_CONCEPT_MAP = "bdh/concept_map.json"

LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "rust": ".rs",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "solidity": ".sol",
    "assembly": ".asm",
}


def detect_repo_language(parsed: dict):
    counts = Counter()
    for data in parsed.values():
        lang = data.get("language")
        if lang:
            counts[lang] += 1
    if not counts:
        return "python", {}
    primary = counts.most_common(1)[0][0]
    return primary, dict(counts)


def detect_target_language(parsed: dict, nodes: list, fallback: str) -> str:
    counts = Counter()
    for node in nodes:
        file_path = node.split("::", 1)[0]
        lang = parsed.get(file_path, {}).get("language")
        if lang:
            counts[lang] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return fallback


def default_extension_for_language(language: str) -> str:
    lang = (language or "").lower()
    return LANGUAGE_EXTENSIONS.get(lang, ".py")


def initialize_repo(repo_url):
    print(f"{C.CYAN}Cloning repository...{C.RST}")
    cloner = RepoCloner()
    repo_path = cloner.clone_repo(repo_url)

    print(f"{C.CYAN}Parsing repository...{C.RST}")
    parser = CodeParser()
    parsed = parser.parse_repository(repo_path)

    repo_language, language_stats = detect_repo_language(parsed)
    print(f"{C.DIM}Detected primary language: {repo_language}{C.RST}")

    os.makedirs("data", exist_ok=True)
    parsed_file = os.path.join("data", "parsed_repo.json")
    with open(parsed_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"{C.DIM}Parsed repo saved to {parsed_file}{C.RST}")

    print(f"{C.CYAN}Building repository graph...{C.RST}")
    builder = CodeGraphBuilder()
    repo_graph = builder.build_graph(parsed)
    print(f"{C.GREEN}Nodes: {C.BOLD}{repo_graph.number_of_nodes()}{C.RST}  "
          f"{C.GREEN}Edges: {C.BOLD}{repo_graph.number_of_edges()}{C.RST}")

    graph_file = "data/repo_graph.pkl"
    with open(graph_file, "wb") as f:
        pickle.dump(repo_graph, f)
    print(f"{C.DIM}Graph saved to {graph_file}{C.RST}")

    print(f"{C.CYAN}Building semantic memory...{C.RST}")
    collection = build_semantic_memmory(parsed)

    print(f"{C.CYAN}Loading Hebbian MemoryGraph...{C.RST}")
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = MemoryGraph.from_json(f.read())
        print(f"{C.GREEN}MemoryGraph loaded from previous session.{C.RST}")
    else:
        memory = MemoryGraph()
        print(f"{C.YELLOW}Initialized new MemoryGraph.{C.RST}")

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

    return (
        parsed,
        repo_graph,
        memory,
        retriever,
        graph_store,
        repo_structure,
        repo_path,
        repo_language,
        language_stats,
    )


def load_bdh_components(retriever):
    """Load BDH model, tokenizer, and concept map if available."""
    bdh_router = None
    working_memory = None

    if not all(os.path.exists(p) for p in [BDH_CHECKPOINT, BDH_TOKENIZER, BDH_CONCEPT_MAP]):
        print(f"{C.YELLOW}BDH components not found. Train BDH first (option 5).{C.RST}")
        return None, None

    try:
        import torch
        from bdh.train import load_checkpoint
        from bdh.tokenizer import CodeTokenizer
        from agent.bdh_router import BDHRouter
        from agent.working_memory import BDHWorkingMemory

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{C.CYAN}Loading BDH model on {C.BOLD}{device}{C.RST}{C.CYAN}...{C.RST}")

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

        print(f"{C.GREEN}BDH components loaded successfully!{C.RST}")
        print(f"  {C.DIM}Model: {config.n_embd}d, {config.n_head} heads, {model.get_neuron_count()} neurons{C.RST}")
        print(f"  {C.DIM}Concepts mapped: {len(concept_map)}{C.RST}")

    except Exception as e:
        print(f"{C.RED}Failed to load BDH: {e}{C.RST}")

    return bdh_router, working_memory


def train_bdh(repo_path, parsed, repo_graph):
    """Train BDH model on the current repository."""
    try:
        import torch
        from bdh.train import train_code_bdh
        from bdh.bdh import CodeBDHConfig, CodeBDHConfigSmall
        from bdh.tokenizer import CodeTokenizer
        from bdh.synapse_inspector import SynapseInspector

        print(f"\n{C.HEADER}{'=' * 50}")
        print(f"  BDH Training")
        print(f"{'=' * 50}{C.RST}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use smaller config on CPU or low-VRAM GPUs to avoid OOM
        force_full = os.getenv("ATLAS_BDH_FULL", "") == "1"
        force_small = os.getenv("ATLAS_BDH_SMALL", "") == "1"

        if device == "cpu":
            print(f"{C.YELLOW}CPU detected - using small model config (~3M params){C.RST}")
            config = CodeBDHConfigSmall()
        else:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if force_small or (vram_gb < 10 and not force_full):
                print(f"{C.YELLOW}GPU detected ({torch.cuda.get_device_name()}) - VRAM ~{vram_gb:.1f} GB{C.RST}")
                print(f"{C.YELLOW}Using small model config to avoid OOM (set ATLAS_BDH_FULL=1 to override){C.RST}")
                config = CodeBDHConfigSmall()
            else:
                print(f"{C.GREEN}GPU detected ({torch.cuda.get_device_name()}) - using full config{C.RST}")
                config = CodeBDHConfig()

        # Configuration
        iters = input(f"{C.CYAN}Training iterations (default 500): {C.RST}").strip()
        max_iters = int(iters) if iters.isdigit() else 500

        print(f"\n{C.DIM}Config: {config.n_embd}d, {config.n_head} heads, {config.n_layer} layers{C.RST}")
        print(f"{C.DIM}Neurons: {config.mlp_internal_dim_multiplier * config.n_embd // config.n_head * config.n_head}{C.RST}")

        # Train
        model = train_code_bdh(
            code_dir=repo_path,
            parsed_repo=parsed,
            graph=repo_graph,
            config=config,
            max_iters=max_iters,
        )

        # Build concept map
        print(f"\n{C.CYAN}Building concept map...{C.RST}")
        tokenizer = CodeTokenizer()
        tokenizer.load(BDH_TOKENIZER)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        inspector = SynapseInspector(model, tokenizer, device)
        concept_map = inspector.build_concept_map()
        inspector.save_map(concept_map, BDH_CONCEPT_MAP)
        inspector.print_report(concept_map)

        print(f"\n{C.GREEN}BDH training complete!{C.RST}")

    except ImportError as e:
        print(f"{C.RED}Missing dependency for BDH training: {e}{C.RST}")
        print(f"{C.YELLOW}Install with: pip install torch tokenizers{C.RST}")
    except Exception as e:
        print(f"{C.RED}Training failed: {e}{C.RST}")


def explore_repo(parsed, graph_store, retriever, repo_structure, memory, repo_language):
    query = input(f"{C.CYAN}What do you want to know about the repo? {C.RST}")
    results = retriever.retrieve(query)

    print(f"\n{C.HEADER}Top retrieved nodes:{C.RST}")
    for node, score in results[:10]:
        node_type = graph_store.graph.nodes[node].get("type")
        folder = graph_store.graph.nodes[node].get("folder")
        module = graph_store.graph.nodes[node].get("module")
        print(f"  {C.MAGENTA}{node_type}{C.RST}: {C.WHITE}{node}{C.RST} "
              f"{C.DIM}(folder: {folder}, module: {module}, score: {score:.3f}){C.RST}")

    top_nodes = [node for node, _ in results[:5]]
    target_language = detect_target_language(parsed, top_nodes, repo_language)

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
                snippets[n] = {
                    "code": code,
                    "language": file_data.get("language"),
                }

    contexts = build_context_for_nodes(graph_store, top_nodes)
    clusters = cluster_by_files(top_nodes)
    cluster_summaries = summarize_clusters(clusters, contexts, snippets)
    global_summary = generate_global_summary(cluster_summaries, repo_structure)

    print(f"\n{C.HEADER}Cluster Summaries:{C.RST}")
    for c in cluster_summaries:
        print(f"  {C.CYAN}{c['cluster']}{C.RST} {C.DIM}->{C.RST} {c['summary']}")
    print(f"\n{C.HEADER}Global Summary:{C.RST}\n{C.WHITE}{global_summary}{C.RST}")

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
        "repo_language": repo_language,
        "target_language": target_language,
    }
    answer = explain_repo(context, query)
    print(f"\n{C.GREEN}Answer:{C.RST}\n{answer}")
    return context


def generate_code(context, repo_path: str):
    query = input(f"{C.CYAN}Enter the new feature or code you want to add: {C.RST}")
    language = context.get("target_language") or context.get("repo_language") or "python"
    target = context.get("target_folder", "new_folder")
    default_name = f"new_feature{default_extension_for_language(language)}"

    validator = CodeValidator(repo_path=repo_path)

    max_attempts = 5
    issues = []
    new_code = ""
    file_map = {}

    for attempt in range(1, max_attempts + 1):
        attempt_query = query
        if issues:
            attempt_query = (
                f"{query}\n\nPrevious attempt failed validation: {issues}\n"
                "Fix the issues and regenerate the full bundle."
            )
        new_code = plan_code(context, attempt_query)
        file_map, _ = parse_code_bundle(new_code, target, default_name)

        valid, issues = validator.validate_bundle(file_map, language=language)
        if valid and issues:
            print(f"{C.YELLOW}Validation warnings: {issues}{C.RST}")
        if valid:
            break

    if issues:
        print(f"{C.RED}Not able to generate after {max_attempts} attempts{C.RST}")
        print(f"{C.DIM}Last issues: {issues}{C.RST}")
        return

    print(f"\n{C.HEADER}Generated Code:{C.RST}\n{C.WHITE}{new_code}{C.RST}")

    save = input(f"{C.CYAN}Save to disk? (y/n): {C.RST}").strip().lower()
    if save == "y":
        paths = execute_plan_bundle(file_map, repo_path)
        print(f"{C.GREEN}Saved {len(paths)} file(s).{C.RST}")


if __name__ == "__main__":
    print(f"\n{C.HEADER}{'=' * 60}")
    print(f"   ATLAS - Intelligent Codebase Reasoning Agent")
    print(f"{'=' * 60}{C.RST}\n")

    repo_url = input(f"{C.CYAN}Enter the Github Repository URL: {C.RST}")
    (parsed, repo_graph, memory, retriever, graph_store,
     repo_structure, repo_path, repo_language, language_stats) = initialize_repo(repo_url)

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
        "repo_language": repo_language,
        "target_language": repo_language,
        "language_stats": language_stats,
    }

    agent = AgentLoop(
        retriever=retriever, graph_store=graph_store,
        memory=memory, parsed=parsed, repo_structure=repo_structure,
        bdh_router=bdh_router, working_memory=working_memory,
        repo_language=repo_language,
        repo_path=repo_path,
    )

    while True:
        print(f"\n{C.HEADER}{'─' * 40}")
        print(f"   MENU")
        print(f"{'─' * 40}{C.RST}")
        print(f"  {C.CYAN}1{C.RST}  Explore / Understand Repository")
        print(f"  {C.CYAN}2{C.RST}  Generate Code (single-shot)")
        print(f"  {C.CYAN}3{C.RST}  Agent Mode (autonomous multi-step)")
        print(f"  {C.CYAN}4{C.RST}  Memory Stats")
        print(f"  {C.CYAN}5{C.RST}  Train BDH Model" + (f" {C.GREEN}[TRAINED]{C.RST}" if bdh_router else ""))
        print(f"  {C.CYAN}6{C.RST}  Load BDH Components" + (f" {C.GREEN}[LOADED]{C.RST}" if bdh_router else ""))
        print(f"  {C.CYAN}7{C.RST}  Exit")
        choice = input(f"{C.CYAN}Choose an option (1-7): {C.RST}")

        if choice == "1":
            result = explore_repo(parsed, graph_store, retriever, repo_structure, memory, repo_language)
            if result:
                context.update(result)
        elif choice == "2":
            generate_code(context, repo_path)
        elif choice == "3":
            task = input(f"{C.CYAN}Describe the feature or task: {C.RST}")
            auto = input(f"{C.CYAN}Auto-save generated files? (y/n): {C.RST}").strip().lower() == "y"
            dash = input(f"{C.CYAN}Show dashboard? (y/n): {C.RST}").strip().lower() == "y"
            agent.run(task, auto_save=auto, show_dashboard=dash)
        elif choice == "4":
            stats = memory.get_stats()
            print(f"\n{C.HEADER}Memory Stats:{C.RST}")
            print(f"  {C.WHITE}Nodes:{C.RST} {C.GREEN}{stats['node_count']}{C.RST}")
            print(f"  {C.WHITE}Edges:{C.RST} {C.GREEN}{stats['edge_count']}{C.RST}")
            print(f"  {C.WHITE}Total Hebbian updates:{C.RST} {C.YELLOW}{stats['total_updates']}{C.RST}")
            print(f"  {C.WHITE}Edges pruned:{C.RST} {stats['edges_pruned']}")
            print(f"  {C.WHITE}Graph density:{C.RST} {stats['density']:.4f}")
            print(f"\n{C.HEADER}Top concepts (by PageRank):{C.RST}")
            print(memory.get_summary(max_nodes=5))
            if bdh_router:
                print(f"\n{C.WHITE}BDH Status:{C.RST} {C.GREEN}ACTIVE{C.RST}")
            else:
                print(f"\n{C.WHITE}BDH Status:{C.RST} {C.YELLOW}Not loaded{C.RST}")
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
            print(f"{C.DIM}Memory saved to {MEMORY_FILE}{C.RST}")
            print(f"{C.MAGENTA}Exiting... Goodbye!{C.RST}")
            break
        else:
            print(f"{C.RED}Invalid choice. Try again.{C.RST}")
