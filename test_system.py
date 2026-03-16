#!/usr/bin/env python3
"""
ATLAS System Verification Test
Checks all components without AWS credentials required
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work"""
    print("\n" + "="*60)
    print("IMPORT VERIFICATION")
    print("="*60)

    modules_to_test = [
        ("networkx", "Graph Library"),
        ("chromadb", "Vector Database"),
        ("sentence_transformers", "Embeddings Model"),
        ("parser.repo_parser", "Code Parser"),
        ("graph.graph_builder", "Graph Builder"),
        ("graph.graph_store", "Graph Store"),
        ("semantic.embeddings", "Embeddings Generator"),
        ("semantic.retrieval", "Code Retriever"),
        ("context.clustering", "Context Clustering"),
        ("context.summarizer", "Context Summarizer"),
        ("utils.llm_call", "LLM Interface"),
        ("utils.planner", "Planning Module"),
        ("agent.agent_loop", "Agent Loop"),
        ("agent.validator", "Code Validator"),
        ("agent.dashboard", "Dashboard"),
    ]

    results = {"pass": 0, "fail": 0, "warn": 0}

    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"[PASS] {module_name:40s} - {description}")
            results["pass"] += 1
        except ImportError as e:
            if "torch" in str(e) or "tokenizers" in str(e):
                print(f"[WARN] {module_name:40s} - {description} (BDH optional)")
                results["warn"] += 1
            else:
                print(f"[FAIL] {module_name:40s} - {e}")
                results["fail"] += 1

    print(f"\nImport Results: {results['pass']} passed, {results['warn']} warned, {results['fail']} failed")
    return results["fail"] == 0


def test_core_classes():
    """Test that core classes can be instantiated"""
    print("\n" + "="*60)
    print("CORE CLASS INSTANTIATION")
    print("="*60)

    results = {"pass": 0, "fail": 0}

    # Test 1: GraphStore
    try:
        import networkx as nx
        from graph.graph_store import GraphStore

        G = nx.DiGraph()
        G.add_node("test_node", type="function")
        G.add_edge("test_node", "other_node")

        store = GraphStore(G)
        downstream = store.get_full_downstream("test_node")
        upstream = store.get_full_upstream("test_node")

        print("[PASS] GraphStore instantiation and BFS methods")
        results["pass"] += 1
    except Exception as e:
        print(f"[FAIL] GraphStore: {e}")
        results["fail"] += 1

    # Test 2: CodeValidator
    try:
        from agent.validator import CodeValidator

        validator = CodeValidator()

        valid_code = """
def hello():
    return "world"
"""
        valid, issues = validator.validate(valid_code)

        if valid:
            print("[PASS] CodeValidator - accepts valid code")
            results["pass"] += 1
        else:
            print(f"[FAIL] CodeValidator rejected valid code: {issues}")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] CodeValidator: {e}")
        results["fail"] += 1

    # Test 3: MemoryGraph
    try:
        from graph.memory_graph import MemoryGraph

        memory = MemoryGraph()
        memory.add_node("func_a", node_type="function")
        memory.add_node("func_b", node_type="function")
        memory.add_edge("func_a", "func_b", weight=0.5, relation_type="calls")

        memory.update_from_concepts(["func_a", "func_b"], context_node="test")
        memory.decay_all()

        stats = memory.get_stats()
        summary = memory.get_summary(max_nodes=5)

        # Check that nodes and edges exist and memory operations work
        if stats["node_count"] >= 2 and stats["edge_count"] >= 1:
            print("[PASS] MemoryGraph - add nodes, update, decay")
            results["pass"] += 1
        else:
            print(f"[FAIL] MemoryGraph - nodes: {stats['node_count']}, edges: {stats['edge_count']}")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] MemoryGraph: {e}")
        results["fail"] += 1

    # Test 4: LLM Call (mock test, no actual API call)
    try:
        from utils.llm_call import ask_llm

        # Just verify the function exists and signature is correct
        import inspect
        sig = inspect.signature(ask_llm)
        params = list(sig.parameters.keys())

        if "prompt" in params and "max_tokens" in params:
            print("[PASS] ask_llm - correct signature (max_tokens ready)")
            results["pass"] += 1
        else:
            print(f"[FAIL] ask_llm signature mismatch: {params}")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] ask_llm: {e}")
        results["fail"] += 1

    # Test 5: Planner module
    try:
        from utils.planner import explain_repo, plan_code
        from utils.planner import _format_code_snippets

        snippets = {"func_a": "def func_a():\n    return 1"}
        formatted = _format_code_snippets(snippets)

        if "func_a" in formatted and "```python" in formatted:
            print("[PASS] Planner - formats code snippets correctly")
            results["pass"] += 1
        else:
            print(f"[FAIL] Code snippet formatting error")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] Planner module: {e}")
        results["fail"] += 1

    print(f"\nClass Instantiation Results: {results['pass']} passed, {results['fail']} failed")
    return results["fail"] == 0


def test_data_structures():
    """Test key data structure operations"""
    print("\n" + "="*60)
    print("DATA STRUCTURE TESTS")
    print("="*60)

    results = {"pass": 0, "fail": 0}

    # Test graph depth limits
    try:
        import networkx as nx
        from graph.graph_store import GraphStore

        G = nx.DiGraph()
        # Create a chain: 0 -> 1 -> 2 -> 3 -> 4 -> 5
        for i in range(5):
            G.add_node(f"node_{i}", type="function")
            G.add_edge(f"node_{i}", f"node_{i+1}")

        store = GraphStore(G)

        # With depth=3, should get nodes 1, 2, 3 (not 4, 5)
        downstream = store.get_full_downstream("node_0", max_depth=3)

        if len(downstream) <= 3:  # Within depth limit
            print("[PASS] GraphStore respects max_depth=3 limit")
            results["pass"] += 1
        else:
            print(f"[FAIL] Depth limit not respected: got {len(downstream)} nodes, expected <= 3")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] Depth limit test: {e}")
        results["fail"] += 1

    # Test Hebbian weight accumulation
    try:
        from graph.memory_graph import MemoryGraph

        memory = MemoryGraph()
        nodes = ["a", "b", "c"]
        for n in nodes:
            memory.add_node(n, node_type="function")

        # First update
        memory.update_from_concepts(nodes, context_node="ctx1")

        # Get weights
        edges1 = list(memory.graph.edges(data=True))
        weight1 = edges1[0][2]["weight"] if edges1 else 0

        # Second update (should increase weights)
        memory.update_from_concepts(nodes, context_node="ctx2")

        edges2 = list(memory.graph.edges(data=True))
        weight2 = edges2[0][2]["weight"] if edges2 else 0

        if weight2 > weight1:
            print("[PASS] MemoryGraph strengthens co-activated edges")
            results["pass"] += 1
        else:
            print(f"[FAIL] Weights not increasing: {weight1} -> {weight2}")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] Hebbian weight test: {e}")
        results["fail"] += 1

    print(f"\nData Structure Results: {results['pass']} passed, {results['fail']} failed")
    return results["fail"] == 0


def test_integration():
    """Test integration between components"""
    print("\n" + "="*60)
    print("INTEGRATION TESTS")
    print("="*60)

    results = {"pass": 0, "fail": 0}

    # Test: Retriever uses MemoryGraph
    try:
        import networkx as nx
        from graph.graph_store import GraphStore
        from graph.memory_graph import MemoryGraph
        from semantic.retrieval import CodeRetriever
        from unittest.mock import MagicMock

        # Create mock collection
        collection = MagicMock()
        collection.query.return_value = {
            'metadatas': [[]],
            'distances': [[]],
            'ids': [[]]
        }

        # Create graph and memory
        G = nx.DiGraph()
        G.add_node("test::func", type="function")
        store = GraphStore(G)
        memory = MemoryGraph()
        memory.add_node("test::func", node_type="function")

        # Create retriever with memory
        retriever = CodeRetriever(
            collection=collection,
            graph_store=store,
            top_k=5,
            memory_graph=memory
        )

        if retriever.memory_graph is not None:
            print("[PASS] CodeRetriever integrates with MemoryGraph")
            results["pass"] += 1
        else:
            print("[FAIL] MemoryGraph not attached to retriever")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] Retriever integration: {e}")
        results["fail"] += 1

    # Test: Agent accesses all components
    try:
        from agent.agent_loop import AgentLoop
        from unittest.mock import MagicMock

        retriever = MagicMock()
        graph_store = MagicMock()
        memory = MagicMock()
        parsed = {}
        repo_structure = ""

        agent = AgentLoop(
            retriever=retriever,
            graph_store=graph_store,
            memory=memory,
            parsed=parsed,
            repo_structure=repo_structure,
        )

        if hasattr(agent, 'validator') and hasattr(agent, 'dashboard'):
            print("[PASS] AgentLoop integrates validator, dashboard, memory")
            results["pass"] += 1
        else:
            print("[FAIL] AgentLoop missing components")
            results["fail"] += 1
    except Exception as e:
        print(f"[FAIL] Agent integration: {e}")
        results["fail"] += 1

    print(f"\nIntegration Results: {results['pass']} passed, {results['fail']} failed")
    return results["fail"] == 0


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ATLAS SYSTEM VERIFICATION")
    print("="*60)
    print("Testing all components (excluding AWS credentials)")

    test_results = []

    test_results.append(("Imports", test_imports()))
    test_results.append(("Core Classes", test_core_classes()))
    test_results.append(("Data Structures", test_data_structures()))
    test_results.append(("Integration", test_integration()))

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    all_passed = all(result for _, result in test_results)

    for name, result in test_results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    if all_passed:
        print("\n[SUCCESS] All system tests passed!")
        print("\nYou can now run: python main2.py")
        return 0
    else:
        print("\n[ERROR] Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
